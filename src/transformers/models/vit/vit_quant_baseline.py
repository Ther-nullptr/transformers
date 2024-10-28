import math
import torch
import typing
import bitsandbytes as bnb
import torch.nn.functional as F
import bitsandbytes.functional as BF

from .layernorm_kernels import layernorm_forward, layernorm_backward
from .gelu_kernels import gelu_backward
from .softmax_kernels import softmax_backward

from .compress_function import (
    compress_pack_channel_base,
    compress_pack_quant_base,
    compress_pack_softmax_base,
    outlier_addition_fuse_decompression_dequantization,
    decompression_dequantization,
    update_dict,
)


def hidden_to_head_shape(x: torch.Tensor, num_heads: int):
    bsz, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)


def head_to_hidden_shape(x: torch.Tensor):
    bsz, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).reshape(bsz, seq_len, -1)


def lora_forward(w, w_quant_state, w_lora_a, w_lora_b, b, x):
    w_dequant = w.T
    x = x.to(w_dequant.dtype)
    x_main = x @ w_dequant + b.to(w_dequant.dtype) if b is not None else x @ w_dequant
    x_lora_a = x @ w_lora_a.to(w_dequant.dtype)
    x_lora = x_lora_a @ w_lora_b.to(w_dequant.dtype)
    x = x_main + x_lora
    return x, x_main, x_lora_a


def lora_backward(w, w_quant_state, w_lora_a, w_lora_b, x, x_lora_a, grad_y):
    w_dequant = w.T
    w_lora_a, w_lora_b = w_lora_a.to(w_dequant.dtype), w_lora_b.to(w_dequant.dtype)
    grad_w_lora_a = x.to(w_dequant.dtype).mT @ (grad_y.to(w_dequant.dtype) @ w_lora_b.mT)
    grad_w_lora_b = x_lora_a.mT @ grad_y.to(w_lora_b.dtype)
    grad_x = grad_y.to(w_dequant.dtype) @ w_dequant.T 
    grad_x += (grad_y.to(w_lora_b.dtype) @ w_lora_b.T @ w_lora_a.T)
    return grad_w_lora_a, grad_w_lora_b, grad_x


class FusedViTLayerFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        #############attention part#############
        w_q: torch.Tensor,
        b_q: torch.Tensor,
        w_q_quant_state: typing.Tuple,
        w_q_lora_a: torch.Tensor,
        w_q_lora_b: torch.Tensor,
        ####################################
        w_k: torch.Tensor,
        b_k: torch.Tensor,
        w_k_quant_state: typing.Tuple,
        w_k_lora_a: torch.Tensor,
        w_k_lora_b: torch.Tensor,
        ####################################
        w_v: torch.Tensor,
        b_v: torch.Tensor,
        w_v_quant_state: typing.Tuple,
        w_v_lora_a: torch.Tensor,
        w_v_lora_b: torch.Tensor,
        ####################################
        w_o: torch.Tensor,
        b_o: torch.Tensor,
        w_o_quant_state: typing.Tuple,
        w_o_lora_a: torch.Tensor,
        w_o_lora_b: torch.Tensor,
        ####################################
        norm_weight_1: torch.Tensor,
        norm_bias_1: torch.Tensor,
        #############mlp part#############
        w_up: torch.Tensor,
        b_up: torch.Tensor,
        w_up_quant_state: typing.Tuple,
        w_up_lora_a: torch.Tensor,
        w_up_lora_b: torch.Tensor,
        ####################################
        w_down: torch.Tensor,
        b_down: torch.Tensor,
        w_down_quant_state: typing.Tuple,
        w_down_lora_a: torch.Tensor,
        w_down_lora_b: torch.Tensor,
        ####################################
        norm_weight_2: torch.Tensor,
        norm_bias_2: torch.Tensor,
        ###############other################
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        ###############about statistics################
        iteration: int,
        iteration_threshold: int,
        static_value: dict,
        softmax_outlier_ratio: float,
        layernorm_outlier_ratio: float,
        q_bit: int,
    ): 
        x_norm_1, mean_1, rstd_1, _, _ = layernorm_forward(x, norm_weight_1, norm_bias_1, eps = 1e-5) 
        
        #* compress the (copy of) x
        x_copy = x.clone()
        x_o, x_q, x_rest, x_channel_idx, x_scale = compress_pack_channel_base(
            x=x_copy, o_ratio=layernorm_outlier_ratio, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x']
        )
        
        # compute q,k,v
        # forward process: q_proj
        q, _, q_lora_a = lora_forward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, b_q, x_norm_1)
        
        # forward process: k_proj
        k, _, k_lora_a = lora_forward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, b_k, x_norm_1)

        # forward process: v_proj
        v, _, v_lora_a = lora_forward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, b_v, x_norm_1)
        
        #* compress x_norm_1
        x_norm_1_q, x_norm_1_rest, x_norm_1_scale = compress_pack_quant_base(
            x=x_norm_1, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x_norm_1']
        )
        del x_norm_1
        
        # reshape
        q = hidden_to_head_shape(q, num_heads)
        k = hidden_to_head_shape(k, num_heads)
        v = hidden_to_head_shape(v, num_heads)
        
        ctx.q_shape = q.shape

        # forward: S = Q @ K.T / sqrt(d_k)
        s = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        # apply mask
        if attention_mask is not None:
            s = s + attention_mask
            
        #* compress q, k
        q_q, q_rest, q_scale = compress_pack_quant_base(
            x=q, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['q']
        )
        k_q, k_rest, k_scale = compress_pack_quant_base(
            x=k, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['k']
        )
        del q, k

        # forward: softmax
        a = torch.softmax(s, dim=-1, dtype=v.dtype)  # [bsz, num_heads, q_len, q_len]
        
        # forward: O = A @ V
        o = a @ v
        
        #* compress a
        a_o, a_threshold = compress_pack_softmax_base(
            x=a, o_ratio=softmax_outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['a']
        )
        del a
        
        #* compress v
        v_q, v_rest, v_scale = compress_pack_quant_base(
            x=v, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['v']
        )
        del v
        
        # reshape
        o = head_to_hidden_shape(o)

        # forward process: o_proj
        o_final, _, o_final_lora_a = lora_forward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, b_o, o)
        
        #* compress o
        o_q, o_rest, o_scale = compress_pack_quant_base(
            x=o, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['o']
        )
        del o
        
        # layernorm or rmsnorm (with residual connection)
        x_medium = x + o_final
        
        x_norm_2, mean_2, rstd_2, block_size, num_warps = layernorm_forward(x_medium, norm_weight_2, norm_bias_2, eps = 1e-5)
        #* compress the (copy of) x_medium
        x_medium_copy = x_medium.clone()
        x_medium_o, x_medium_q, x_medium_rest, x_medium_channel_idx, x_medium_scale = compress_pack_channel_base(
            x=x_medium_copy, o_ratio=layernorm_outlier_ratio, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x_medium']
        )
        
        # forward process: up_proj
        up, _, up_lora_a = lora_forward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, b_up, x_norm_2)
        
        #* compress the x_norm_2
        x_norm_2_q, x_norm_2_rest, x_norm_2_scale = compress_pack_quant_base(
            x=x_norm_2, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x_norm_2']
        )
        del x_norm_2
        
        # activation function
        fn = torch.nn.functional.gelu(up)
        
        up_q, up_rest, up_scale = compress_pack_quant_base(
            x=up, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['up']
        )

        # forward process: down_proj
        down, _, down_lora_a = lora_forward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, b_down, fn)
        x_out = x_medium + down
        
        fn_q, fn_rest, fn_scale = compress_pack_quant_base(
            x=fn, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['fn']
        )
        
        ctx.save_for_backward(
            ### activations (attention) ###
            x_o, x_q, x_rest, x_scale,
            mean_1, # buffer for layernorm
            rstd_1, # buffer for layernorm
            x_norm_1_q, x_norm_1_rest, x_norm_1_scale,
            q_lora_a, # buffer for lora (qkv)
            k_lora_a, # buffer for lora (qkv)
            v_lora_a, # buffer for lora (qkv)
            q_q, q_rest, q_scale, # buffer for attn
            k_q, k_rest, k_scale, # buffer for attn
            v_q, v_rest, v_scale, # buffer for attn
            a_o, a_threshold, # buffer for attn
            o_q, o_rest, o_scale, # buffer for o
            o_final_lora_a, # buffer for lora (o)
            ### activations (mlp) ###
            x_medium_o, x_medium_q, x_medium_rest, x_medium_scale, # buffer for up
            x_norm_2_q, x_norm_2_rest, x_norm_2_scale,
            mean_2, # buffer for layernorm
            rstd_2, # buffer for layernorm
            up_lora_a, # buffer for lora (up)
            up_q, up_rest, up_scale, # up
            fn_q, fn_rest, fn_scale, # fn
            down_lora_a, # buffer for lora (down)
            ### weights (attention) ###
            w_q,
            b_q,
            w_q_lora_a,
            w_q_lora_b,
            #**********************
            w_k,
            b_k,
            w_k_lora_a,
            w_k_lora_b,
            #**********************
            w_v,
            b_v,
            w_v_lora_a,
            w_v_lora_b,
            #**********************
            w_o,
            b_o,
            w_o_lora_a,
            w_o_lora_b,
            #**********************
            norm_weight_1, 
            norm_bias_1,
            ### weights (mlp) ###
            #**********************
            w_up,
            b_up,
            w_up_lora_a,
            w_up_lora_b,
            #**********************
            w_down,
            b_down,
            w_down_lora_a,
            w_down_lora_b,
            #**********************
            norm_weight_2,
            norm_bias_2,
        )
        ctx.quant_state = (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
            w_up_quant_state,
            w_down_quant_state,
        )
        ctx.num_heads = num_heads
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        ctx.head_dim = head_dim
        
        ctx.input_layernorm_channel = x_channel_idx
        ctx.post_layernorm_channel = x_medium_channel_idx
        ctx.q_bit = q_bit

        return x_out, x_scale, x_channel_idx, x_norm_1_scale, \
            q_scale, k_scale, v_scale, \
            a_threshold, \
            o_scale, \
            x_medium_scale, x_medium_channel_idx, x_norm_2_scale, \
            up_scale, fn_scale
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args):
        (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
            w_up_quant_state,
            w_down_quant_state,
        ) = ctx.quant_state
        
        (
            ### activations (attention) ###
            x_o, x_q, x_rest, x_scale,
            mean_1, # buffer for layernorm
            rstd_1, # buffer for layernorm
            x_norm_1_q, x_norm_1_rest, x_norm_1_scale,
            q_lora_a, # buffer for lora (qkv)
            k_lora_a, # buffer for lora (qkv)
            v_lora_a, # buffer for lora (qkv)
            q_q, q_rest, q_scale, # buffer for attn
            k_q, k_rest, k_scale, # buffer for attn
            v_q, v_rest, v_scale, # buffer for attn
            a_o, a_threshold, # buffer for attn
            o_q, o_rest, o_scale, # buffer for o
            o_final_lora_a, # buffer for lora (o)
            ### activations (mlp) ###
            x_medium_o, x_medium_q, x_medium_rest, x_medium_scale, # buffer for up
            x_norm_2_q, x_norm_2_rest, x_norm_2_scale,
            mean_2, # buffer for layernorm
            rstd_2, # buffer for layernorm
            up_lora_a, # buffer for lora (up)
            up_q, up_rest, up_scale, # up
            fn_q, fn_rest, fn_scale, # fn
            down_lora_a, # buffer for lora (down)
            ### weights (attention) ###
            w_q,
            b_q,
            w_q_lora_a,
            w_q_lora_b,
            #**********************
            w_k,
            b_k,
            w_k_lora_a,
            w_k_lora_b,
            #**********************
            w_v,
            b_v,
            w_v_lora_a,
            w_v_lora_b,
            #**********************
            w_o,
            b_o,
            w_o_lora_a,
            w_o_lora_b,
            #**********************
            norm_weight_1, 
            norm_bias_1,
            ### weights (mlp) ###
            #**********************
            w_up,
            b_up,
            w_up_lora_a,
            w_up_lora_b,
            #**********************
            w_down,
            b_down,
            w_down_lora_a,
            w_down_lora_b,
            #**********************
            norm_weight_2,
            norm_bias_2,
        ) = ctx.saved_tensors

        # down proj part
        fn = decompression_dequantization(fn_q, fn_rest, fn_scale, ctx.q_bit)
        grad_w_down_lora_a, grad_w_down_lora_b, grad_down = lora_backward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, fn, down_lora_a, grad_output)
        
        # TODO: activation backward
        # activation part
        up = decompression_dequantization(up_q, up_rest, up_scale, ctx.q_bit)
        grad_fn = gelu_backward(up, grad_down)
        
        # up proj part
        x_norm_2 = decompression_dequantization(x_norm_2_q, x_norm_2_rest, x_norm_2_scale, ctx.q_bit)
        grad_w_up_lora_a, grad_w_up_lora_b, grad_up = lora_backward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, x_norm_2, up_lora_a, grad_fn)
        
        # layer norm
        x_medium = outlier_addition_fuse_decompression_dequantization(x_medium_q, x_medium_rest, x_medium_scale, x_medium_o, ctx.post_layernorm_channel, ctx.q_bit)
        grad_norm_2, _, _ = layernorm_backward(
            grad_up, x_medium, norm_weight_2, norm_bias_2, mean_2, rstd_2, # TODO: other params
            True, 1e-5, ctx.num_warps, ctx.block_size
        )
        
        # residual connection
        grad_medium = grad_norm_2 + grad_output
        
        # o part
        o = decompression_dequantization(o_q, o_rest, o_scale, ctx.q_bit)
        grad_w_o_lora_a, grad_w_o_lora_b, grad_o = lora_backward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, o, o_final_lora_a, grad_medium)
        
        # reshape
        grad_o = hidden_to_head_shape(grad_o, ctx.num_heads)

        # backward of second GEMM: O = A @ V
        # d L / d V = A.T @ d L / d O
        a = a_o.to_dense()
        v = decompression_dequantization(v_q, v_rest, v_scale, ctx.q_bit, is_head=True, num_heads=ctx.num_heads)
        grad_v = a.transpose(-2, -1) @ grad_o
        grad_a = grad_o @ v.transpose(-2, -1)

        # backward of softmax
        grad_s = softmax_backward(a, grad_a)
        
        q = decompression_dequantization(q_q, q_rest, q_scale, ctx.q_bit, is_head=True, num_heads=ctx.num_heads)
        k = decompression_dequantization(k_q, k_rest, k_scale, ctx.q_bit, is_head=True, num_heads=ctx.num_heads)
        # backward of first GEMM: S = Q @ K.T / sqrt(d_k)
        grad_s = grad_s / math.sqrt(ctx.head_dim)
        # d L / d K = (d L / d S)^T @ Q
        grad_k = grad_s.transpose(-2, -1) @ q
        # d L / d Q = d L / d S @ K
        grad_q = grad_s @ k

        grad_q = head_to_hidden_shape(grad_q)
        grad_k = head_to_hidden_shape(grad_k)
        grad_v = head_to_hidden_shape(grad_v)
        
        x_norm_1 = decompression_dequantization(x_norm_1_q, x_norm_1_rest, x_norm_1_scale, ctx.q_bit)
        del x_norm_1_q, x_norm_1_scale
        
        # backward of q_proj
        grad_w_q_lora_a, grad_w_q_lora_b, grad_x = lora_backward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, x_norm_1, q_lora_a, grad_q)

        # backward of k_proj
        grad_w_k_lora_a, grad_w_k_lora_b, grad_x_temp = lora_backward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, x_norm_1, k_lora_a, grad_k)
        grad_x += grad_x_temp

        # backward of v_proj
        grad_w_v_lora_a, grad_w_v_lora_b, grad_x_temp = lora_backward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, x_norm_1, v_lora_a, grad_v)
        grad_x += grad_x_temp
        
        #* dequantize x
        x = outlier_addition_fuse_decompression_dequantization(x_q, x_rest, x_scale, x_o, ctx.input_layernorm_channel, ctx.q_bit)
        del x_q, x_scale, x_o
        
        # layernorm or rmsnorm backward
        grad_norm_1, _, _ = layernorm_backward(
            grad_x, x, norm_weight_1, norm_bias_1, mean_1, rstd_1, # TODO: other params
            True, 1e-5, ctx.num_warps, ctx.block_size
        )
          
        # residual connection
        grad_input = grad_medium + grad_norm_1
        
        return (
            grad_input,
            #############attention part#############
            None,
            None,
            None,
            grad_w_q_lora_a,
            grad_w_q_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_k_lora_a,
            grad_w_k_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_v_lora_a,
            grad_w_v_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_o_lora_a,
            grad_w_o_lora_b,
            ####################################
            None,
            None,
            ####################################
            None,
            None,
            None,
            grad_w_up_lora_a,
            grad_w_up_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_down_lora_a,
            grad_w_down_lora_b,
            ####################################
            None,
            None,
        ) + (None,) * 10


class FusedViTLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
    ):
        super(FusedViTLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.iteration = 0
        self.iteration_threshold = 5
        self.softmax_outlier_ratio = 0.05
        self.layernorm_outlier_ratio = 0
        self.q_bit = 4
        self.static_value = {
            'x': {'outlier_channel_index': None, 'scale': None},
            'x_norm_1': {'scale': None},
            'q': {'scale': None},
            'k': {'scale': None},
            'v': {'scale': None},
            'a': {'outlier': None},
            'o': {'scale': None},
            'x_medium': {'outlier_channel_index': None, 'scale': None},
            'x_norm_2': {'scale': None},
            'up': {'scale': None},
            'fn': {'scale': None},
        }
        print(f'FusedViTLayer(no reorder): softmax_outlier_ratio={self.softmax_outlier_ratio}, layernorm_outlier_ratio={self.layernorm_outlier_ratio}, q_bit={self.q_bit}')

    def forward(
        self,
        input: torch.Tensor,
        ############################################
        q_proj_base: bnb.nn.modules.Linear4bit,
        q_proj_lora_a: torch.nn.Linear,
        q_proj_lora_b: torch.nn.Linear,
        k_proj_base: bnb.nn.modules.Linear4bit,
        k_proj_lora_a: torch.nn.Linear,
        k_proj_lora_b: torch.nn.Linear,
        v_proj_base: bnb.nn.modules.Linear4bit,
        v_proj_lora_a: torch.nn.Linear,
        v_proj_lora_b: torch.nn.Linear,
        o_proj_base: bnb.nn.modules.Linear4bit,
        o_proj_lora_a: torch.nn.Linear,
        o_proj_lora_b: torch.nn.Linear,
        norm_weight_1: torch.Tensor,
        norm_bias_1: torch.Tensor,
        ############################################
        up_proj_base: bnb.nn.modules.Linear4bit,
        up_proj_lora_a: torch.nn.Linear,
        up_proj_lora_b: torch.nn.Linear,
        down_proj_base: bnb.nn.modules.Linear4bit,
        down_proj_lora_a: torch.nn.Linear,
        down_proj_lora_b: torch.nn.Linear,
        norm_weight_2: torch.Tensor,
        norm_bias_2: torch.Tensor,
        ############################################
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        ############################################
    ):
        x_out, x_scale, x_channel_idx, x_norm_1_scale, \
        q_scale, k_scale, v_scale, \
        a_threshold, \
        o_scale, \
        x_medium_scale, x_medium_channel_idx, x_norm_2_scale, \
        up_scale, fn_scale = FusedViTLayerFunc.apply(
            input,
            #############attention part#############
            q_proj_base.weight,
            q_proj_base.bias,
            None,
            q_proj_lora_a.default.weight.T,
            q_proj_lora_b.default.weight.T,
            ####################################
            k_proj_base.weight,
            k_proj_base.bias,
            None,
            k_proj_lora_a.default.weight.T,
            k_proj_lora_b.default.weight.T,
            ####################################
            v_proj_base.weight,
            v_proj_base.bias,
            None,
            v_proj_lora_a.default.weight.T,
            v_proj_lora_b.default.weight.T,
            ####################################
            o_proj_base.weight,
            o_proj_base.bias,
            None,
            o_proj_lora_a.default.weight.T,
            o_proj_lora_b.default.weight.T,
            ####################################
            norm_weight_1,
            norm_bias_1,
            #############mlp part#############
            up_proj_base.weight,
            up_proj_base.bias,
            None,
            up_proj_lora_a.default.weight.T,
            up_proj_lora_b.default.weight.T,
            ####################################
            down_proj_base.weight,
            down_proj_base.bias,
            None,
            down_proj_lora_a.default.weight.T,
            down_proj_lora_b.default.weight.T,
            ####################################
            norm_weight_2,
            norm_bias_2,
            ####################################
            attention_mask,
            num_heads,
            head_dim,
            ####################################
            self.iteration,
            self.iteration_threshold,
            self.static_value,
            self.softmax_outlier_ratio,
            self.layernorm_outlier_ratio,
            self.q_bit,
        )
        
        if self.iteration < self.iteration_threshold:
            self.static_value['x'] = update_dict(self.static_value['x'], {'outlier_channel_index': x_channel_idx, 'scale': x_scale}, self.iteration)
            self.static_value['x_norm_1'] = update_dict(self.static_value['x_norm_1'], {'scale': x_norm_1_scale}, self.iteration)
            self.static_value['q'] = update_dict(self.static_value['q'], {'scale': q_scale}, self.iteration)
            self.static_value['k'] = update_dict(self.static_value['k'], {'scale': k_scale}, self.iteration)
            self.static_value['v'] = update_dict(self.static_value['v'], {'scale': v_scale}, self.iteration)
            self.static_value['a'] = update_dict(self.static_value['a'], {'outlier': a_threshold}, self.iteration)
            self.static_value['o'] = update_dict(self.static_value['o'], {'scale': o_scale}, self.iteration)
            self.static_value['x_medium'] = update_dict(self.static_value['x_medium'], {'outlier_channel_index': x_medium_channel_idx, 'scale': x_medium_scale}, self.iteration)
            self.static_value['x_norm_2'] = update_dict(self.static_value['x_norm_2'], {'scale': x_norm_2_scale}, self.iteration)
            self.static_value['up'] = update_dict(self.static_value['up'], {'scale': up_scale}, self.iteration)
            self.static_value['fn'] = update_dict(self.static_value['fn'], {'scale': fn_scale}, self.iteration)
        self.iteration += 1
        
        return x_out