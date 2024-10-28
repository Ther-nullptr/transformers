import math
import torch
import typing
import bitsandbytes as bnb
import torch.nn.functional as F
import bitsandbytes.functional as BF

from .rmsnorm_kernels import rmsnorm_forward, rmsnorm_backward
from .softmax_kernels import softmax_backward
from .rope_kernels import calculate_settings, rope_forward, rope_backward
from .silu_kernels import silu_backward

from .compress_function import (
    compress_pack_softmax_base,
    compress_pack_structed_pruning_base,
    decompress_structed_pruning,
    update_dict,
)


def hidden_to_head_shape(x: torch.Tensor, num_heads: int):
    bsz, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)


def head_to_hidden_shape(x: torch.Tensor):
    bsz, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).reshape(bsz, seq_len, -1)


def lora_forward(w, w_quant_state, w_lora_a, w_lora_b, b, x, lora_scale):
    w_dequant = BF.dequantize_nf4(w, w_quant_state).t()
    x = x.to(w_dequant.dtype)
    x_main = x @ w_dequant + b.to(w_dequant.dtype) if b is not None else x @ w_dequant
    x_lora_a = x @ w_lora_a.to(w_dequant.dtype)
    x_lora = x_lora_a @ w_lora_b.to(w_dequant.dtype)
    x = x_main + x_lora * lora_scale
    return x, x_main, x_lora_a


def lora_backward(w, w_quant_state, w_lora_a, w_lora_b, x, x_lora_a, grad_y, lora_scale):
    w_dequant = BF.dequantize_nf4(w, w_quant_state).t()
    w_lora_a, w_lora_b = w_lora_a.to(w_dequant.dtype), w_lora_b.to(w_dequant.dtype)
    grad_w_lora_a = x.to(w_dequant.dtype).mT @ (grad_y.to(w_dequant.dtype) @ w_lora_b.mT) * lora_scale
    grad_w_lora_b = (x_lora_a.mT @ grad_y.to(w_lora_b.dtype)) * lora_scale
    grad_x = grad_y.to(w_dequant.dtype) @ w_dequant.T 
    grad_x += (((grad_y.to(w_lora_b.dtype) @ w_lora_b.T) * lora_scale) @ w_lora_a.T)
    return grad_w_lora_a, grad_w_lora_b, grad_x


class FusedLlamaLayerFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        #############attention part#############
        norm_weight_1: torch.Tensor,
        norm_bias_1: torch.Tensor,
        ####################################
        cos: torch.Tensor,
        sin: torch.Tensor,
        ####################################
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
        #############mlp part#############
        norm_weight_2: torch.Tensor,
        norm_bias_2: torch.Tensor,
        ####################################
        w_gate: torch.Tensor,
        b_gate: torch.Tensor,
        w_gate_quant_state: typing.Tuple,
        w_gate_lora_a: torch.Tensor,
        w_gate_lora_b: torch.Tensor,
        ####################################
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
        ###############other################
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        lora_scale: float,
        ###############about statistics################
        iteration: int,
        iteration_threshold: int,
        static_value: dict,
        softmax_outlier_ratio: float,
        outlier_ratio: float,
    ):
        # layernorm or rmsnorm
        x_norm_1, mean_1, rstd_1, _, _ = rmsnorm_forward(x, norm_weight_1, eps = 1e-5)
        
        #* compress the (copy of) x
        x_copy = x.clone()
        x_o, x_threshold = compress_pack_structed_pruning_base(
            x=x_copy, channel_ratio=outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x']
        )

        # compute q,k,v
        # forward process: q_proj
        q, _, q_lora_a = lora_forward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, b_q, x_norm_1, lora_scale)

        # forward process: k_proj
        k, _, k_lora_a = lora_forward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, b_k, x_norm_1, lora_scale)

        # forward process: v_proj
        v, _, v_lora_a = lora_forward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, b_v, x_norm_1, lora_scale)
        
        #* compress x_norm_1
        x_norm_1_o, x_norm_1_threshold = compress_pack_structed_pruning_base(
            x=x_norm_1, channel_ratio=outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x_norm_1']
        )
        del x_norm_1
        
        # reshape
        q = hidden_to_head_shape(q, num_heads)
        k = hidden_to_head_shape(k, num_heads)
        v = hidden_to_head_shape(v, num_heads)
        
        ctx.q_shape = q.shape

        q = rope_forward(q.transpose(1, 2), cos, sin).transpose(1, 2)
        k = rope_forward(k.transpose(1, 2), cos, sin).transpose(1, 2)

        # forward: S = Q @ K.T / sqrt(d_k)
        s = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        
        #* compress q, k
        q = head_to_hidden_shape(q)
        q_o, q_threshold = compress_pack_structed_pruning_base(
            x=q, channel_ratio=outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['q']
        )
        k = head_to_hidden_shape(k)
        k_o, k_threshold = compress_pack_structed_pruning_base(
            x=k, channel_ratio=outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['k']
        )
        del q, k
        
        # apply mask
        if attention_mask is not None:
            s = s + attention_mask

        # forward: softmax
        a = torch.softmax(s, dim=-1, dtype=v.dtype)  # [bsz, num_heads, q_len, q_len]
        del s

        # forward: O = A @ V
        o = a @ v
        
        #* compress a
        a_o, a_threshold = compress_pack_softmax_base(
            x=a, o_ratio=softmax_outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['a']
        )
        del a
        
        #* compress v
        v = head_to_hidden_shape(v)
        v_o, v_threshold = compress_pack_structed_pruning_base(
            x=v, channel_ratio=outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['v']
        )
        del v
        
        # reshape
        o = head_to_hidden_shape(o)

        # forward process: o_proj
        o_final, _, o_final_lora_a = lora_forward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, b_o, o, lora_scale)
        
        #* compress o
        o_o, o_threshold = compress_pack_structed_pruning_base(
            x=o, channel_ratio=outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['o']
        )
        del o
        
        # residual connection
        x_medium = x + o_final
        del x, o_final
        
        # layernorm or rmsnorm
        x_norm_2, mean_2, rstd_2, block_size, num_warps = rmsnorm_forward(x_medium, norm_weight_2, eps = 1e-5)
        
        #* compress the (copy of) x_medium
        x_medium_copy = x_medium.clone()
        x_medium_o, x_medium_threshold = compress_pack_structed_pruning_base(
            x=x_medium_copy, channel_ratio=outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x_medium']
        )
        
        # forward process: gate_proj
        gate, _, gate_lora_a = lora_forward(w_gate, w_gate_quant_state, w_gate_lora_a, w_gate_lora_b, b_gate, x_norm_2, lora_scale)
        
        # forward process: up_proj
        up, _, up_lora_a = lora_forward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, b_up, x_norm_2, lora_scale)

        #* compress the x_norm_2
        x_norm_2_o, x_norm_2_threshold = compress_pack_structed_pruning_base(
            x=x_norm_2, channel_ratio=outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x_norm_2']
        )
        del x_norm_2
        
        # apply activation function (for gate)
        fn = torch.nn.functional.silu(gate)
        
        #* compress the gate
        gate_o, gate_threshold = compress_pack_structed_pruning_base(
            x=gate, channel_ratio=outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['gate']
        )
        ctx.shape_2 = gate.shape
        del gate
            
        # hadamard
        hadamard = up * fn
        
        #* compress the up / fn
        up_o, up_threshold = compress_pack_structed_pruning_base(
            x=up, channel_ratio=outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['up']
        )
        del up
        fn_o, fn_threshold = compress_pack_structed_pruning_base(
            x=fn, channel_ratio=outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['fn']
        )
        del fn
            
        # forward process: down_proj
        down, _, down_lora_a = lora_forward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, b_down, hadamard, lora_scale)
        
        #* compress the hadamard
        hadamard_o, hadamard_threshold = compress_pack_structed_pruning_base(
            x=hadamard, channel_ratio=outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['hadamard']
        )
        del hadamard
        
        # residual connection
        x_out = x_medium + down
        del x_medium, down
        
        ctx.save_for_backward(
            ### buffered activation (attention) ###
            x_o, x_threshold, # x
            mean_1, rstd_1, # buffer for rmsnorm
            x_norm_1_o, x_norm_1_threshold, # x_norm_1
            cos, sin, # buffer for rope
            q_o, q_threshold, # q
            k_o, k_threshold, # k
            v_o, v_threshold, # v
            a_o, a_threshold, # a
            o_o, o_threshold, # o
            q_lora_a, k_lora_a, v_lora_a, # buffer for lora (qkv)
            o_final_lora_a, # buffer for lora (o)
            ### buffered activation (mlp) ###
            mean_2, rstd_2, # buffer for rmsnorm
            x_medium_o, x_medium_threshold, # x_medium
            x_norm_2_o, x_norm_2_threshold, # x_norm_2
            gate_o, gate_threshold, # gate
            up_o, up_threshold, # up
            fn_o, fn_threshold, # fn
            hadamard_o, hadamard_threshold, # hadamard
            gate_lora_a, up_lora_a, down_lora_a,
            ### weights (attention) ###
            norm_weight_1, norm_bias_1,
            w_q, b_q, w_q_lora_a, w_q_lora_b,
            w_k, b_k, w_k_lora_a, w_k_lora_b,
            w_v, b_v, w_v_lora_a, w_v_lora_b,
            w_o, b_o, w_o_lora_a, w_o_lora_b,
            ### weights (mlp) ###
            norm_weight_2, norm_bias_2,
            w_gate, b_gate, w_gate_lora_a, w_gate_lora_b,
            w_up, b_up, w_up_lora_a, w_up_lora_b,
            w_down, b_down, w_down_lora_a, w_down_lora_b,
        )
        ctx.quant_state = (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
            w_gate_quant_state,
            w_up_quant_state,
            w_down_quant_state,
        )
        ctx.num_heads = num_heads
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        ctx.head_dim = head_dim
        ctx.lora_scale = lora_scale
        ctx.shape_1 = x_out.shape

        return x_out, x_threshold, \
            x_norm_1_threshold, \
            q_threshold, k_threshold, v_threshold, \
            a_threshold, o_threshold, \
            x_medium_threshold, x_medium_threshold, \
            x_norm_2_threshold, \
            gate_threshold, up_threshold, fn_threshold, hadamard_threshold
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args):
        (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
            w_gate_quant_state,
            w_up_quant_state,
            w_down_quant_state,
        ) = ctx.quant_state
        
        (
            ### buffered activation (attention) ###
            x_o, x_threshold, # x
            mean_1, rstd_1, # buffer for rmsnorm
            x_norm_1_o, x_norm_1_threshold, # x_norm_1
            cos, sin, # buffer for rope
            q_o, q_threshold, # q
            k_o, k_threshold, # k
            v_o, v_threshold, # v
            a_o, a_threshold, # a
            o_o, o_threshold, # o
            q_lora_a, k_lora_a, v_lora_a, # buffer for lora (qkv)
            o_final_lora_a, # buffer for lora (o)
            ### buffered activation (mlp) ###
            mean_2, rstd_2, # buffer for rmsnorm
            x_medium_o, x_medium_threshold, # x_medium
            x_norm_2_o, x_norm_2_threshold, # x_norm_2
            gate_o, gate_threshold, # gate
            up_o, up_threshold, # up
            fn_o, fn_threshold, # fn
            hadamard_o, hadamard_threshold, # hadamard
            gate_lora_a, up_lora_a, down_lora_a,
            ### weights (attention) ###
            norm_weight_1, norm_bias_1,
            w_q, b_q, w_q_lora_a, w_q_lora_b,
            w_k, b_k, w_k_lora_a, w_k_lora_b,
            w_v, b_v, w_v_lora_a, w_v_lora_b,
            w_o, b_o, w_o_lora_a, w_o_lora_b,
            ### weights (mlp) ###
            norm_weight_2, norm_bias_2,
            w_gate, b_gate, w_gate_lora_a, w_gate_lora_b,
            w_up, b_up, w_up_lora_a, w_up_lora_b,
            w_down, b_down, w_down_lora_a, w_down_lora_b,
        ) = ctx.saved_tensors
        
        #* dequantize hadamard
        hadamard = decompress_structed_pruning(hadamard_o, hadamard_threshold, ctx.shape_2)
        del hadamard_o
        
        # down proj part
        grad_w_down_lora_a, grad_w_down_lora_b, grad_down = lora_backward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, hadamard, down_lora_a, grad_output, ctx.lora_scale)
        del hadamard
        
        #* dequantize up
        up = decompress_structed_pruning(up_o, up_threshold, ctx.shape_2)
        fn = decompress_structed_pruning(fn_o, fn_threshold, ctx.shape_2)
        del up_o, fn_o
        
        # hadamard
        grad_hadamard_1 = grad_down * up
        grad_hadamard_2 = grad_down * fn
        del grad_down, up, fn
        
        #* dequantize gate
        gate = decompress_structed_pruning(gate_o, gate_threshold, ctx.shape_2)
        grad_fn = silu_backward(gate, grad_hadamard_1)
        del gate_o, gate, grad_hadamard_1
        
        #* dequantize x_norm_2
        x_norm_2 = decompress_structed_pruning(x_norm_2_o, x_norm_2_threshold, ctx.shape_1)
        del x_norm_2_o
        
        # gate proj part
        grad_w_gate_lora_a, grad_w_gate_lora_b, grad_gate = lora_backward(w_gate, w_gate_quant_state, w_gate_lora_a, w_gate_lora_b, x_norm_2, gate_lora_a, grad_fn, ctx.lora_scale)
        del grad_fn
        
        # up proj part
        grad_w_up_lora_a, grad_w_up_lora_b, grad_up = lora_backward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, x_norm_2, up_lora_a, grad_hadamard_2, ctx.lora_scale)
        grad_gate_up = grad_up + grad_gate
        del grad_up, grad_gate, grad_hadamard_2
        
        #* dequantize x_medium
        x_medium = decompress_structed_pruning(x_medium_o, x_medium_threshold, ctx.shape_1)
        del x_medium_o
        
        # layernorm & rmsnorm backward
        grad_norm_2, _ = rmsnorm_backward(
            grad_gate_up, x_medium, norm_weight_2, mean_2, rstd_2, # TODO: other params
            True, 1e-5, ctx.num_warps, ctx.block_size
        )
        del x_medium, grad_gate_up
        
        # residual connection
        grad_medium = grad_norm_2 + grad_output
        
        #* dequantize o
        o = decompress_structed_pruning(o_o, o_threshold, ctx.shape_1)
        del o_o
        
        # o part
        grad_w_o_lora_a, grad_w_o_lora_b, grad_o = lora_backward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, o, o_final_lora_a, grad_medium, ctx.lora_scale)
        del o
        
        # reshape
        grad_o = hidden_to_head_shape(grad_o, ctx.num_heads)
        
        #* dequantize a
        a = a_o.to_dense()
        v = decompress_structed_pruning(v_o, v_threshold, ctx.shape_1)
        v = hidden_to_head_shape(v, ctx.num_heads)
        del a_o, v_o
        
        # backward of second GEMM: O = A @ V
        # d L / d V = A.T @ d L / d O
        grad_v = a.transpose(-2, -1) @ grad_o
        grad_a = grad_o @ v.transpose(-2, -1)
        del grad_o, v

        # backward of softmax
        grad_s = softmax_backward(a, grad_a)
        del a, grad_a

        # backward of first GEMM: S = Q @ K.T / sqrt(d_k)
        grad_s = grad_s / math.sqrt(ctx.head_dim)
        
        #* dequantize q
        q = decompress_structed_pruning(q_o, q_threshold, ctx.shape_1)
        q = hidden_to_head_shape(q, ctx.num_heads)
        k = decompress_structed_pruning(k_o, k_threshold, ctx.shape_1)
        k = hidden_to_head_shape(k, ctx.num_heads)
        # d L / d K = (d L / d S)^T @ Q
        grad_k = grad_s.transpose(-2, -1) @ q
        # d L / d Q = d L / d S @ K
        grad_q = grad_s @ k
        del grad_s, q, k

        BLOCK_SIZE, num_warps = calculate_settings(ctx.head_dim // 2)
        N_GROUPS = 128
        grad_q = rope_backward(grad_q.transpose(1, 2), cos, sin, N_GROUPS, BLOCK_SIZE, num_warps).transpose(1, 2) # TODO: other params
        grad_k = rope_backward(grad_k.transpose(1, 2), cos, sin, N_GROUPS, BLOCK_SIZE, num_warps).transpose(1, 2)

        grad_q = head_to_hidden_shape(grad_q)
        grad_k = head_to_hidden_shape(grad_k)
        grad_v = head_to_hidden_shape(grad_v)
        
        #* dequantize x_norm_1
        x_norm_1 = decompress_structed_pruning(x_norm_1_o, x_norm_1_threshold, ctx.shape_1)
        del x_norm_1_o
        
        # backward of q_proj
        grad_w_q_lora_a, grad_w_q_lora_b, grad_x = lora_backward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, x_norm_1, q_lora_a, grad_q, ctx.lora_scale)

        # backward of k_proj
        grad_w_k_lora_a, grad_w_k_lora_b, grad_x_temp = lora_backward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, x_norm_1, k_lora_a, grad_k, ctx.lora_scale)
        grad_x += grad_x_temp

        # backward of v_proj
        grad_w_v_lora_a, grad_w_v_lora_b, grad_x_temp = lora_backward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, x_norm_1, v_lora_a, grad_v, ctx.lora_scale)
        grad_x += grad_x_temp
        
        #* dequantize x
        x = decompress_structed_pruning(x_o, x_threshold, ctx.shape_1)
        del x_o
        
        # layernorm or rmsnorm backward
        grad_norm_1, _ = rmsnorm_backward(
            grad_x, x, norm_weight_1, mean_1, rstd_1, # TODO: other params
            True, 1e-5, ctx.num_warps, ctx.block_size
        )
            
        # residual connection
        grad_input = grad_norm_1 + grad_medium
        
        return (
            grad_input,
            #############attention part#############
            None,
            None,
            ####################################
            None,
            None,
            ####################################
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
            grad_w_gate_lora_a,
            grad_w_gate_lora_b,
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
            grad_w_down_lora_b
        ) + (None,) * 9


class FusedLlamaLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
    ):
        super(FusedLlamaLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.iteration = 0
        self.iteration_threshold = 5
        self.outlier_ratio = 0.25
        self.softmax_outlier_ratio = 0.25
        self.static_value = {
            'x': {'outlier_channel_index': None},
            'x_norm_1': {'outlier_channel_index': None},
            'q': {'outlier_channel_index': None},
            'k': {'outlier_channel_index': None},
            'v': {'outlier_channel_index': None},
            'a': {'outlier': None},
            'o': {'outlier_channel_index': None},
            'x_medium': {'outlier_channel_index': None},
            'x_norm_2': {'outlier_channel_index': None},
            'gate': {'outlier_channel_index': None},
            'up': {'outlier_channel_index': None},
            'fn': {'outlier_channel_index': None},
            'hadamard': {'outlier_channel_index': None},
        }
        print(f'FusedLlamaLayer(no reorder): softmax_outlier_ratio={self.softmax_outlier_ratio}')
        
    def forward(
        self,
        input: torch.Tensor,
        ############################################
        norm_weight_1: torch.Tensor,
        norm_bias_1: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
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
        ############################################
        norm_weight_2: torch.Tensor,
        norm_bias_2: torch.Tensor,
        gate_proj_base: bnb.nn.modules.Linear4bit,
        gate_proj_lora_a: torch.nn.Linear,
        gate_proj_lora_b: torch.nn.Linear,
        up_proj_base: bnb.nn.modules.Linear4bit,
        up_proj_lora_a: torch.nn.Linear,
        up_proj_lora_b: torch.nn.Linear,
        down_proj_base: bnb.nn.modules.Linear4bit,
        down_proj_lora_a: torch.nn.Linear,
        down_proj_lora_b: torch.nn.Linear,
        ############################################
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        lora_scale: float,
    ):
        y, x_threshold, \
        x_norm_1_threshold, \
        q_threshold, k_threshold, v_threshold, \
        a_threshold, o_threshold, \
        x_medium_threshold, x_medium_threshold, \
        x_norm_2_threshold, \
        gate_threshold, up_threshold, fn_threshold, hadamard_threshold = FusedLlamaLayerFunc.apply(
            input,
            #############attention part#############
            norm_weight_1,
            norm_bias_1,
            ####################################
            cos,
            sin,
            ####################################
            q_proj_base.weight,
            q_proj_base.bias,
            q_proj_base.weight.quant_state,
            q_proj_lora_a.default.weight.T,
            q_proj_lora_b.default.weight.T,
            ####################################
            k_proj_base.weight,
            k_proj_base.bias,
            k_proj_base.weight.quant_state,
            k_proj_lora_a.default.weight.T,
            k_proj_lora_b.default.weight.T,
            ####################################
            v_proj_base.weight,
            v_proj_base.bias,
            v_proj_base.weight.quant_state,
            v_proj_lora_a.default.weight.T,
            v_proj_lora_b.default.weight.T,
            ####################################
            o_proj_base.weight,
            o_proj_base.bias,
            o_proj_base.weight.quant_state,
            o_proj_lora_a.default.weight.T,
            o_proj_lora_b.default.weight.T,
            #############mlp part#############
            norm_weight_2,
            norm_bias_2,
            ####################################
            gate_proj_base.weight,
            gate_proj_base.bias,
            gate_proj_base.weight.quant_state,
            gate_proj_lora_a.default.weight.T,
            gate_proj_lora_b.default.weight.T,
            ####################################
            up_proj_base.weight,
            up_proj_base.bias,
            up_proj_base.weight.quant_state,
            up_proj_lora_a.default.weight.T,
            up_proj_lora_b.default.weight.T,
            ####################################
            down_proj_base.weight,
            down_proj_base.bias,
            down_proj_base.weight.quant_state,
            down_proj_lora_a.default.weight.T,
            down_proj_lora_b.default.weight.T,
            ####################################
            attention_mask,
            num_heads,
            head_dim,
            lora_scale,
            ####################################
            self.iteration,
            self.iteration_threshold,
            self.static_value,
            self.softmax_outlier_ratio,
            self.outlier_ratio,
        )
        
        if self.iteration < self.iteration_threshold:
            self.static_value['x'] = update_dict(self.static_value['x'], {'outlier_channel_index': x_threshold}, self.iteration)
            self.static_value['x_norm_1'] = update_dict(self.static_value['x_norm_1'], {'outlier_channel_index': x_norm_1_threshold}, self.iteration)
            self.static_value['q'] = update_dict(self.static_value['q'], {'outlier_channel_index': q_threshold}, self.iteration)
            self.static_value['k'] = update_dict(self.static_value['k'], {'outlier_channel_index': k_threshold}, self.iteration)
            self.static_value['v'] = update_dict(self.static_value['v'], {'outlier_channel_index': v_threshold}, self.iteration)
            self.static_value['a'] = update_dict(self.static_value['a'], {'outlier': a_threshold}, self.iteration)
            self.static_value['o'] = update_dict(self.static_value['o'], {'outlier_channel_index': o_threshold}, self.iteration)
            self.static_value['x_medium'] = update_dict(self.static_value['x_medium'], {'outlier_channel_index': x_medium_threshold}, self.iteration)
            self.static_value['x_norm_2'] = update_dict(self.static_value['x_norm_2'], {'outlier_channel_index': x_norm_2_threshold}, self.iteration)
            self.static_value['gate'] = update_dict(self.static_value['gate'], {'outlier_channel_index': gate_threshold}, self.iteration)
            self.static_value['up'] = update_dict(self.static_value['up'], {'outlier_channel_index': up_threshold}, self.iteration)
            self.static_value['fn'] = update_dict(self.static_value['fn'], {'outlier_channel_index': fn_threshold}, self.iteration)
            self.static_value['hadamard'] = update_dict(self.static_value['hadamard'], {'outlier_channel_index': hadamard_threshold}, self.iteration)
        self.iteration += 1
        
        return y
        