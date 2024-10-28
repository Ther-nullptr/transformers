import torch
from .compress_function_kernel import *


def hidden_to_head_shape(x: torch.Tensor, num_heads: int):
    bsz, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)


def head_to_hidden_shape(x: torch.Tensor):
    bsz, _, seq_len, _ = x.shape
    return x.transpose(1, 2).reshape(bsz, seq_len, -1)


@torch.no_grad
def compress_softmax(x: torch.Tensor, outlier: float):
    mask = (x > outlier)
    x_outlier = x * mask
    x_outlier_sparse = x_outlier.to_sparse()
    return x_outlier_sparse


@torch.no_grad
def decompress_softmax(x_sparse: torch.Tensor):
    return x_sparse.to_dense()


@torch.no_grad
def get_statistics_softmax(x: torch.Tensor, outlier_ratio: float):
    outlier = torch.kthvalue(x.float().flatten(), int(x.numel() * (1 - outlier_ratio))).values
    return outlier


@torch.no_grad
def compress_unstructed_pruning(x: torch.Tensor, outlier: float):
    mask = (x.abs() > outlier)
    x_outlier = x * mask
    x_outlier_sparse = x_outlier.to_sparse()
    return x_outlier_sparse


@torch.no_grad
def decompress_unstructed_pruning(x_sparse: torch.Tensor):
    return x_sparse.to_dense()


@torch.no_grad
def get_statistics_outlier(x: torch.Tensor, outlier_ratio: float):
    outlier = torch.kthvalue(x.float().abs().flatten(), int(x.numel() * (1 - outlier_ratio))).values
    return outlier


@torch.no_grad
def compress_structed_pruning(x: torch.Tensor, channel_idx: torch.Tensor):
    x_outlier = x[:, :, channel_idx]
    return x_outlier, channel_idx


@torch.no_grad
def decompress_structed_pruning(x_outlier: torch.Tensor, channel_idx: torch.Tensor, x_shape: torch.Tensor):
    x = torch.zeros(x_shape, device=x_outlier.device, dtype=x_outlier.dtype)
    x[:, :, channel_idx] = x_outlier
    return x


@torch.no_grad
def get_statistics_structed_pruning(x: torch.Tensor, outlier_ratio: float):
    channel_norm = x.abs().norm(dim=-2)
    outlier_channel_index = torch.topk(channel_norm, int(x.shape[-1] * outlier_ratio), largest=True).indices
    return outlier_channel_index


@torch.no_grad
def pad_cut_L(src_L, tgt_L_len):
    seq_len_1, r = src_L.shape
    seq_len_2 = tgt_L_len
    if seq_len_1 < seq_len_2:
        src_L = torch.cat((src_L, torch.zeros(seq_len_2 - seq_len_1, r).to(src_L.dtype).to(src_L.device)), dim=0)
    elif seq_len_1 > seq_len_2:
        src_L = src_L[0:seq_len_2, :]
    return src_L.contiguous()


def update_dict(old_dict, new_dict, iteration):
    for name in old_dict.keys():
        if name == 'L':
            if old_dict[name] is None:
                old_dict[name] = new_dict[name]
            else:
                old_dict[name] = (iteration * old_dict[name] + pad_cut_L(new_dict[name], old_dict[name].shape[0])) / (iteration + 1)
        elif name == 'outlier_channel_index':
            if old_dict[name] is None:
                old_dict[name] = new_dict[name]
            else: # TODO merge logic
                old_dict[name] = new_dict[name]
        else:
            if old_dict[name] is None:
                old_dict[name] = new_dict[name]
            else:
                old_dict[name] = (iteration * old_dict[name] + new_dict[name]) / (iteration + 1)
    return old_dict


@torch.no_grad
def get_statistics_only_quant(x: torch.Tensor, q_bit: int = 8, q_method: str = 'per-tensor'):
    if len(x.shape) == 4:
        batch, num_head, seq_len, sep_dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, seq_len, num_head * sep_dim)

    x_sample = x[0]
    if q_method == 'per-tensor':
        # TODO: set the scale factor to per channel or per tensor?
        scale = (x_sample.max() - x_sample.min()) / (2 ** q_bit - 1)
        # make the shape of scale same as 
    elif q_method == 'per-channel':
        # channel dimension: -2
        scale = (x_sample.max(dim=-2, keepdim=True).values - x_sample.min(dim=-2, keepdim=True).values) / (2 ** q_bit - 1)
    else:
        raise "Unsupport Quantize Method"
    
    del x

    return scale.to(torch.bfloat16)


@torch.no_grad
def get_statistics_channel_base(x: torch.Tensor, outlier_ratio: float, q_bit: int = 8, q_method: str = 'per-tensor'):
    x_ = x.clone()
    if len(x_.shape) == 4:
        batch, num_head, seq_len, sep_dim = x_.shape
        x_ = x_.permute(0, 2, 1, 3).reshape(batch, seq_len, num_head * sep_dim)
    
    channel_norm = x_[0].abs().norm(dim=-2)

    outlier_channel_index = torch.topk(channel_norm, int(x_[0].shape[-1] * outlier_ratio), largest=True).indices

    x_outlier = x_[:, :, outlier_channel_index]
    x_outlier = x_outlier.to(torch.bfloat16)
    x_[:, :, outlier_channel_index] = 0

    x_sub_outlier = x_[0]
    if q_method == 'per-tensor':
        # TODO: set the scale factor to per channel or per tensor?
        scale = (x_sub_outlier.max() - x_sub_outlier.min()) / (2 ** q_bit - 1)
    elif q_method == 'per-channel':
        # channel dimension: -2
        scale = (x_sub_outlier.max(dim=-2, keepdim=True).values - x_sub_outlier.min(dim=-2, keepdim=True).values) / (2 ** q_bit - 1)
    else:
        raise "Unsupport Quantize Method"
    
    del x_
    
    scale += (scale == 0) * 1e-3

    return outlier_channel_index, scale.to(torch.bfloat16)


@torch.no_grad
def get_statistics_rank_base(x: torch.Tensor, q_bit: int = 8, q_method: str = 'per-tensor', svd_rank: int = 16):
    if len(x.shape) == 4:
        batch, num_head, seq_len, sep_dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, seq_len, num_head * sep_dim)
    if svd_rank > 0:
        x0 = x[0].to(torch.float32)
        U, S, V = torch.svd(x0)
        U, S, V = U[:, :svd_rank], S[:svd_rank], V[:, :svd_rank]
        L = U
        L = L.contiguous()
        R = torch.diag(S) @ V.T
        R = R.contiguous()
        x = x - L @ R
        del U, S, V
    else:
        L = torch.zeros((x.shape[-2], 16)).to(x.device).to(x.dtype)
        R = torch.zeros((16, x.shape[-1])).to(x.device).to(x.dtype)

    x_sub_outlier = x[0]
    if q_method == 'per-tensor':
        scale = (x_sub_outlier.max() - x_sub_outlier.min()) / (2 ** q_bit - 1)
    elif q_method == 'per-channel':
        scale = (x_sub_outlier.max(dim=-2, keepdim=True).values - x_sub_outlier.min(dim=-2, keepdim=True).values) / (2 ** q_bit - 1)
    else:
        raise "Unsupport Quantize Method"
    # clear the cache
    torch.cuda.empty_cache()
    
    scale += (scale == 0) * 1e-3
    
    return L.to(torch.bfloat16), R.to(torch.bfloat16), scale.to(torch.bfloat16)


@torch.no_grad
def low_rank_subtraction_fuse_compression_quantization(l, r, x, s, quantize_bit=8, dtype=torch.bfloat16):
    # Change shape if need
    is_head = len(x.shape) == 4
    if is_head:
        x = head_to_hidden_shape(x)
        
    # decide dtype
    l, r, x, s = l.to(dtype), r.to(dtype), x.to(dtype), s.to(dtype)
    
    ML, K = l.shape
    K, N = r.shape
    B, M, _ = x.shape
    
    if ML != M: # adjust
        l = pad_cut_L(l, M)
        
    if K < 16:
        l = torch.cat([l, torch.zeros((M, 16 - K), device=l.device, dtype=l.dtype)], dim=1).contiguous()
        r = torch.cat([r, torch.zeros((16 - K, N), device=r.device, dtype=r.dtype)], dim=0).contiguous()
        K = 16
    
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    elem_per_position = 8 // quantize_bit
    o = torch.empty((B, M, N), device=x.device, dtype=torch.bfloat16)
    q = torch.empty((B, M, N // elem_per_position), device=x.device, dtype=torch.uint8)
    low_rank_subtraction_fuse_compression_quantization_kernel[grid](
        l, r, x, o, q, s,
        B, M, N, K,
        l.stride(0), l.stride(1),
        r.stride(0), r.stride(1),
        x.stride(0), x.stride(1), x.stride(2), 
        o.stride(0), o.stride(1), o.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(0), s.stride(1),
        quantize_bit, elem_per_position,
        BLOCK_SIZE_K=K
    )
    
    del o, x
    return q


@torch.no_grad
def low_rank_addition_fuse_decompression_dequantization(l, r, q, s, quantize_bit=8, is_head=False, num_heads=1, dtype=torch.bfloat16):
    ML, K = l.shape
    K, N = r.shape
    B, M, _ = q.shape

    if ML != M: # adjust
        l = pad_cut_L(l, M)

    if K < 16:
        l = torch.cat([l, torch.zeros((M, 16 - K), device=l.device, dtype=l.dtype)], dim=1).contiguous()
        r = torch.cat([r, torch.zeros((16 - K, N), device=r.device, dtype=r.dtype)], dim=0).contiguous()
        K = 16

    # dtype
    l, r, s = l.to(dtype), r.to(dtype), s.to(dtype)

    # 1D launch kernel where each block gets its own program.
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    x = torch.empty((B, M, N), device=l.device, dtype=torch.bfloat16)
    x_temp = torch.empty((B, M, N), device=l.device, dtype=torch.uint8)

    low_rank_addition_fuse_decompression_dequantization_kernel[grid](
        l, r, x, x_temp, q, s,
        B, M, N, K,
        l.stride(0), l.stride(1),
        r.stride(0), r.stride(1),
        x.stride(0), x.stride(1), x.stride(2),
        x_temp.stride(0), x_temp.stride(1), x_temp.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(0), s.stride(1),
        quantize_bit, elem_per_position,
        BLOCK_SIZE_K=K
    )
    del x_temp
    
    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    
    return x


@torch.no_grad
def outlier_subtraction_fuse_compression_quantization(x, s, channel, quantize_bit=8, dtype=torch.bfloat16):
    # Change shape if need
    is_head = len(x.shape) == 4
    if is_head:
        x = head_to_hidden_shape(x)

    # decide dtype
    x, s = x.to(dtype), s.to(dtype)
    B, M, N = x.shape

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    elem_per_position = 8 // quantize_bit
    x_temp = torch.empty((B, M, N), device=x.device, dtype=torch.bfloat16)
    q = torch.empty((B, M, N // elem_per_position), device=x.device, dtype=torch.uint8)
    
    # remove outlier channels
    x_outlier = x[:, :, channel]
    x[:, :, channel] = 0
    
    # quantize the rest part
    compression_quantization_kernel[grid](
        x, q, x_temp, s,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(0), s.stride(1),
        quantize_bit, elem_per_position
    )

    del x_temp, x
    return x_outlier, q


@torch.no_grad
def outlier_addition_fuse_decompression_dequantization(q, s, x_outlier, channel, quantize_bit=8, is_head=False, num_heads=1, dtype=torch.bfloat16):
    B, M, _ = q.shape
    N = s.shape[-1]

    # dtype
    s = s.to(dtype)

    # 1D launch kernel where each block gets its own program.
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    x = torch.empty((B, M, N), device=q.device, dtype=torch.bfloat16)
    x_temp = torch.empty((B, M, N), device=q.device, dtype=torch.uint8)

    decompression_dequantization_kernel[grid](
        x, x_temp, q, s,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        x_temp.stride(0), x_temp.stride(1), x_temp.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(0), s.stride(1),
        quantize_bit, elem_per_position,
    )
    del x_temp
    
    x[:, :, channel] = x_outlier
    
    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    
    return x


@torch.no_grad
def compression_quantization(x, s, quantize_bit=8, dtype=torch.bfloat16):
    # Change shape if need
    is_head = len(x.shape) == 4
    if is_head:
        x = head_to_hidden_shape(x)

    # decide dtype
    x, s = x.to(dtype), s.to(dtype)
    B, M, N = x.shape

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    elem_per_position = 8 // quantize_bit
    x_temp = torch.empty((B, M, N), device=x.device, dtype=torch.bfloat16)
    q = torch.empty((B, M, N // elem_per_position), device=x.device, dtype=torch.uint8)
    
    # quantize the rest part
    compression_quantization_kernel[grid](
        x, q, x_temp, s,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(0), s.stride(1),
        quantize_bit, elem_per_position
    )

    del x
    return q


@torch.no_grad
def decompression_dequantization(q, s, quantize_bit=8, is_head=False, num_heads=1, dtype=torch.bfloat16):
    B, M, _ = q.shape
    N = s.shape[-1]

    # dtype
    s = s.to(dtype)

    # 1D launch kernel where each block gets its own program.
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    x = torch.empty((B, M, N), device=q.device, dtype=torch.bfloat16)
    x_temp = torch.empty((B, M, N), device=q.device, dtype=torch.uint8)

    decompression_dequantization_kernel[grid](
        x, x_temp, q, s,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        x_temp.stride(0), x_temp.stride(1), x_temp.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(0), s.stride(1),
        quantize_bit, elem_per_position,
    )
    del x_temp
    
    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    
    return x


def compress_pack_channel_base(x, o_ratio, q_bit, q_method, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        o_channel_idx, scale = get_statistics_channel_base(x, o_ratio, q_bit, q_method)
    else:
        o_channel_idx, scale = static_value['outlier_channel_index'], static_value['scale']
    o, q = outlier_subtraction_fuse_compression_quantization(x, scale, o_channel_idx, q_bit)
    return o, q, o_channel_idx, scale


def compress_pack_rank_base(x, rank, q_bit, q_method, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        l, r, scale = get_statistics_rank_base(x, q_bit, q_method, rank)
    else:
        l, r, scale = static_value['L'], static_value['R'], static_value['scale']
    q = low_rank_subtraction_fuse_compression_quantization(l, r, x, scale, q_bit)
    return q, l, r, scale


def compress_pack_softmax_base(x, o_ratio, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        outlier = get_statistics_softmax(x, o_ratio)
    else:
        outlier = static_value['outlier']
    o = compress_softmax(x, outlier)
    return o, outlier


def compress_pack_quant_base(x, q_bit, q_method, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        scale = get_statistics_only_quant(x, q_bit, q_method)
    else:
        scale = static_value['scale']
    q = compression_quantization(x, scale, q_bit)
    return q, scale


def compress_pack_unstructed_pruning_base(x, o_ratio, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        outlier = get_statistics_outlier(x, o_ratio) # a funny reuse
    else:
        outlier = static_value['outlier']
    x_outlier = compress_unstructed_pruning(x, outlier)
    return x_outlier, outlier


def compress_pack_structed_pruning_base(x, channel_ratio, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        channel_idx = get_statistics_structed_pruning(x, channel_ratio)
    else:
        channel_idx = static_value['outlier_channel_index']
    x_outlier, channel_idx = compress_structed_pruning(x, channel_idx)
    return x_outlier, channel_idx


def compute_overhead(overhead_ratio, x, q_bit):
    b, s, d = x.shape
    # 16 * p / w; (s + d) * r * 16 / (b * s * d * w) 
    outlier_channel_ratio = overhead_ratio * q_bit / 16
    low_rank_size = overhead_ratio * b * s * d * q_bit / (16 * (s + d))
    return outlier_channel_ratio, low_rank_size


@torch.no_grad
def compute_compress_loss(x_original, q_bit, o_ratio, rank, is_head=False, num_heads=1):
    # x = x_original.clone()
    # batch = x.shape[0]
    # x_calib = x[0:batch // 2]
    # x_eval = x[batch // 2:]
    # x_eval_base = x_eval.clone()
    # # 1. channel
    # o_channel_idx, scale1 = get_statistics_channel_base(x_calib, o_ratio, q_bit, 'per-channel')
    
    # # 2. rank
    # l, r, scale2 = get_statistics_rank_base(x_calib, q_bit, 'per-channel', rank)
    
    # # compress & decompress
    # # baseline(raw quantization)
    # x_outlier_0, q0 = outlier_subtraction_fuse_compression_quantization(x_eval, scale1, [], q_bit)
    # x_outlier_decompressed0 = outlier_addition_fuse_decompression_dequantization(q0, scale1, x_outlier_0, [], q_bit, is_head=is_head, num_heads=num_heads)
    
    # x_outlier, q1 = outlier_subtraction_fuse_compression_quantization(x_eval, scale1, o_channel_idx, q_bit)
    # x_outlier_decompressed1 = outlier_addition_fuse_decompression_dequantization(q1, scale1, x_outlier, o_channel_idx, q_bit, is_head=is_head, num_heads=num_heads)
    
    # q2 = low_rank_subtraction_fuse_compression_quantization(l, r, x_eval, scale2, q_bit)
    # x_outlier_decompressed2 = low_rank_addition_fuse_decompression_dequantization(l, r, q2, scale2, q_bit, is_head=is_head, num_heads=num_heads)
    
    # # calculate the mse
    # mse0 = torch.nn.functional.mse_loss(x_eval_base, x_outlier_decompressed0)
    # mse1 = torch.nn.functional.mse_loss(x_eval_base, x_outlier_decompressed1)
    # mse2 = torch.nn.functional.mse_loss(x_eval_base, x_outlier_decompressed2)
    
    return 'channel'
    