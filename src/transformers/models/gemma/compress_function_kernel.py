import triton
import triton.language as tl
from packaging.version import Version

triton_version = triton.__version__
if Version(triton_version) < Version("3.0.0"):
    import triton.language.math as tl_math
else:
    import triton.language.extra.cuda.libdevice as tl_math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4, }, num_stages=2, 
                      num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def low_rank_subtraction_fuse_compression_quantization_kernel(
    # Pointers to matrices
    l_ptr, r_ptr, x_ptr, o_ptr, q_ptr, s_ptr,
    # Matrix dimensions
    B, M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_lm, stride_lk,
    stride_rk, stride_rn,
    stride_xb, stride_xm, stride_xn,
    stride_ob, stride_om, stride_on,
    stride_qb, stride_qm, stride_qn,
    stride_sb, stride_sn, # scaling factor has no stride in the m dimension
    # compress params
    quantize_bit: tl.constexpr, elem_per_position: tl.constexpr, 
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    o_ptr means outlier, q_ptr means quantized values, s_ptr means scaling factors
    """
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    l_ptrs = l_ptr + (offs_m[:, None] * stride_lm + offs_k[None, :] * stride_lk)
    r_ptrs = r_ptr + (offs_k[:, None] * stride_rk + offs_n[None, :] * stride_rn)

    x_ptrs = x_ptr + stride_xm * offs_m[:, None] + stride_xn * offs_n[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    o_ptrs = o_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :] + offs_b * stride_ob
    o_mask = (offs_b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    accumulator = tl.load(x_ptrs, mask=x_mask, other=0.0)
    accumulator = accumulator.to(tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(l_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(r_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator -= tl.dot(a, b, allow_tf32=True)
        # Advance the ptrs to the next K block.
        l_ptrs += BLOCK_SIZE_K * stride_lk
        r_ptrs += BLOCK_SIZE_K * stride_rk

    # divide the simple values and outlier values
    x = accumulator.to(tl.bfloat16)

    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q_ptrs = q_ptr + stride_qm * offs_m[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q_mask = (offs_b < B) & (offs_m[:, None] < M) & (offs_qn[None, :] < N // elem_per_position)
    q = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // elem_per_position), dtype=tl.uint8)

    s_ptrs = s_ptr + stride_sn * offs_n[None, :]
    s_mask = (offs_n[None, :] < N)
    s = tl.load(s_ptrs, mask=s_mask, other=1.0)

    x = tl_math.round(x / s)
    x = tl.where(x < -2 ** (quantize_bit - 1), -2 ** (quantize_bit - 1), x)
    x = tl.where(x > 2 ** (quantize_bit - 1) - 1, 2 ** (quantize_bit - 1) - 1, x)
    x = x + 2 ** (quantize_bit - 1)
    tl.store(o_ptrs, x, mask=o_mask)
    
    offs_x_slice_m = offs_m
    offs_x_slice_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    x_slice_ptrs = o_ptr + stride_xm * offs_x_slice_m[:, None] + stride_xn * offs_x_slice_n[None, :] + offs_b * stride_xb
    x_slice_mask = (offs_b < B) & (offs_x_slice_m[:, None] < M) & (offs_x_slice_n[None, :] < N)
    
    for i in range(elem_per_position):
        x_slice_ptrs_new = x_slice_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        element_fake_int = tl.load(x_slice_ptrs_new, mask=x_slice_mask, other=0.0).to(tl.float32) 
        element_int = tl_math.float2uint_rn(element_fake_int)
        q |= (element_int << (quantize_bit * i)).to(tl.uint8)

    tl.store(q_ptrs, q, mask=q_mask)
    
    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8, }, num_stages=1, 
                      num_warps=2), #! config: 1bit, 2; 2bit, 2; 4bit, 4; 8bit, 4
    ],
    key=['M', 'N'],
)
@triton.jit
def low_rank_addition_fuse_decompression_dequantization_kernel(
    # Pointers to matrices
    l_ptr, r_ptr, x_ptr, x_temp_ptr, q_ptr, s_ptr,
    # Matrix dimensions
    B, M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_lm, stride_lk,
    stride_rk, stride_rn,
    stride_xb, stride_xm, stride_xn,
    stride_x_tempb, stride_x_tempm, stride_x_tempn,
    stride_qb, stride_qm, stride_qn,
    stride_sb, stride_sn, # scaling factor has no stride in the m dimension
    # compress params
    quantize_bit: tl.constexpr, elem_per_position: tl.constexpr, 
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, 
):
    # basic pointers
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    l_ptrs = l_ptr + (offs_m[:, None] * stride_lm + offs_k[None, :] * stride_lk)
    r_ptrs = r_ptr + (offs_k[:, None] * stride_rk + offs_n[None, :] * stride_rn)

    x_ptrs = x_ptr + stride_xm * offs_m[:, None] + stride_xn * offs_n[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # quantization operators
    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q_ptrs = q_ptr + stride_qm * offs_m[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q_mask = (offs_b < B) & (offs_m[:, None] < M) & (offs_qn[None, :] < N // elem_per_position)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    offs_x_temp_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    x_temp_ptrs = x_temp_ptr + stride_x_tempm * offs_m[:, None] + stride_x_tempn * offs_x_temp_n[None, :] + offs_b * stride_x_tempb

    mask = (1 << quantize_bit) - 1

    # extract the quantized values
    for i in range(elem_per_position):
        x_temp_ptrs_new = x_temp_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        element_fake_int = tl_math.uint2float_rn((q & mask).to(tl.uint32))
        tl.store(x_temp_ptrs_new, element_fake_int)
        q = (q >> quantize_bit).to(tl.uint8)
        
    # dequantize
    offs_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    s_ptrs = s_ptr + stride_sn * offs_sn[None, :] 
    s_mask = (offs_sn[None, :] < N)
    s = tl.load(s_ptrs, mask=s_mask, other=1.0)
    
    offs_x_temp_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_temp_ptrs = x_temp_ptr + stride_xm * offs_m[:, None] + stride_xn * offs_x_temp_n[None, :] + offs_b * stride_xb
    
    x = tl.load(x_temp_ptrs, mask=x_mask, other=0.0)
    x = x - 2 ** (quantize_bit - 1)
    x = x.to(tl.bfloat16)
    x = x * s
    x = x.to(tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(l_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(r_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        x += tl.dot(a, b, allow_tf32=True, out_dtype=tl.float32)
        # Advance the ptrs to the next K block.
        l_ptrs += BLOCK_SIZE_K * stride_lk
        r_ptrs += BLOCK_SIZE_K * stride_rk
        
    x = x.to(tl.bfloat16)
    tl.store(x_ptrs, x, mask=x_mask)
    
    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=2, 
                      num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def compression_quantization_kernel(
    # Pointers to matrices
    x_ptr, q_ptr, x_temp_ptr, s_ptr,
    # Matrix dimensions
    B, M, N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_xb, stride_xm, stride_xn,
    stride_qb, stride_qm, stride_qn,
    stride_sb, stride_sn, # scaling factor has no stride in the m dimension
    # compress params
    quantize_bit: tl.constexpr, elem_per_position: tl.constexpr, 
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr, 
):
    """
    o_ptr means outlier, q_ptr means quantized values, s_ptr means scaling factors
    """
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = x_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)
    
    x_temp_ptrs = x_temp_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x_temp_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)
     
    offs_qm = offs_xm
    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q_ptrs = q_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q_mask = (offs_b < B) & (offs_qm[:, None] < M) & (offs_qn[None, :] < N // elem_per_position)
    q = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // elem_per_position), dtype=tl.uint8)
    
    offs_sn = offs_xn
    s_ptrs = s_ptr + stride_sn * offs_sn[None, :] 
    s_mask = (offs_sn[None, :] < N)
    
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)
    s = tl.load(s_ptrs, mask=s_mask, other=1.0)
    x = tl_math.round(x / s)
    x = tl.where(x < -2 ** (quantize_bit - 1), -2 ** (quantize_bit - 1), x)
    x = tl.where(x > 2 ** (quantize_bit - 1) - 1, 2 ** (quantize_bit - 1) - 1, x)
    x = x + 2 ** (quantize_bit - 1)
    tl.store(x_temp_ptrs, x, mask=x_temp_mask)
    
    offs_x_slice_m = offs_xm
    offs_x_slice_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    x_slice_ptrs = x_temp_ptr + stride_xm * offs_x_slice_m[:, None] + stride_xn * offs_x_slice_n[None, :] + offs_b * stride_xb
    x_slice_mask = (offs_b < B) & (offs_x_slice_m[:, None] < M) & (offs_x_slice_n[None, :] < N)
    
    for i in range(elem_per_position):
        x_slice_ptrs_new = x_slice_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        element_fake_int = tl.load(x_slice_ptrs_new, mask=x_slice_mask, other=0.0).to(tl.float32) 
        element_int = tl_math.float2uint_rn(element_fake_int)
        q |= (element_int << (quantize_bit * i)).to(tl.uint8)
    
    tl.store(q_ptrs, q, mask=q_mask)

    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8, }, num_stages=1, 
                      num_warps=2), #! config: 1bit, 2; 2bit, 2; 4bit, 4; 8bit, 4
    ],
    key=['M', 'N'],
)
@triton.jit
def decompression_dequantization_kernel(
    # Pointers to matrices
    x_ptr, x_temp_ptr, q_ptr, s_ptr,
    # Matrix dimensions
    B, M, N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_xb, stride_xm, stride_xn,
    stride_x_tempb, stride_x_tempm, stride_x_tempn,
    stride_qb, stride_qm, stride_qn,
    stride_sb, stride_sn, # scaling factor has no stride in the m dimension
    # compress params
    quantize_bit: tl.constexpr, elem_per_position: tl.constexpr, 
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr, 
):
    # basic pointers
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = x_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)

    # quantization operators
    offs_qm = offs_xm
    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q_ptrs = q_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q_mask = (offs_b < B) & (offs_qm[:, None] < M) & (offs_qn[None, :] < N // elem_per_position)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    offs_x_temp_m = offs_xm
    offs_x_temp_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    x_temp_ptrs = x_temp_ptr + stride_x_tempm * offs_x_temp_m[:, None] + stride_x_tempn * offs_x_temp_n[None, :] + offs_b * stride_x_tempb

    mask = (1 << quantize_bit) - 1

    # extract the quantized values
    for i in range(elem_per_position):
        x_temp_ptrs_new = x_temp_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        element_fake_int = tl_math.uint2float_rn((q & mask).to(tl.uint32))
        tl.store(x_temp_ptrs_new, element_fake_int)
        q = (q >> quantize_bit).to(tl.uint8)
        
    # dequantize
    offs_sn = offs_xn
    s_ptrs = s_ptr + stride_sn * offs_sn[None, :] 
    s_mask = (offs_sn[None, :] < N)
    s = tl.load(s_ptrs, mask=s_mask, other=1.0)
    
    offs_x_temp_m = offs_xm
    offs_x_temp_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_temp_ptrs = x_temp_ptr + stride_xm * offs_x_temp_m[:, None] + stride_xn * offs_x_temp_n[None, :] + offs_b * stride_xb
    
    x = tl.load(x_temp_ptrs, mask=x_mask, other=0.0)
    x = x - 2 ** (quantize_bit - 1)
    x = x.to(tl.bfloat16)
    x = x * s
    x = x.to(tl.bfloat16)
    tl.store(x_ptrs, x, mask=x_mask)
    
    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=2, 
                      num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def compression_quantization_with_zero_point_kernel(
    # Pointers to matrices
    x_ptr, q_ptr, x_temp_ptr, s_ptr, z_ptr,
    # Matrix dimensions
    B, M, N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_xb, stride_xm, stride_xn,
    stride_qb, stride_qm, stride_qn,
    stride_sb, stride_sn, # scaling factor has no stride in the m dimension
    stride_zb, stride_zn,
    # compress params
    quantize_bit: tl.constexpr, elem_per_position: tl.constexpr, 
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr, 
):
    """
    o_ptr means outlier, q_ptr means quantized values, s_ptr means scaling factors
    """
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = x_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)
    
    x_temp_ptrs = x_temp_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x_temp_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)
     
    offs_qm = offs_xm
    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q_ptrs = q_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q_mask = (offs_b < B) & (offs_qm[:, None] < M) & (offs_qn[None, :] < N // elem_per_position)
    q = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // elem_per_position), dtype=tl.uint8)
    
    offs_sn = offs_xn
    s_ptrs = s_ptr + stride_sn * offs_sn[None, :] 
    s_mask = (offs_sn[None, :] < N)
    
    offs_zn = offs_xn
    z_ptrs = z_ptr + stride_zn * offs_zn[None, :]
    z_mask = (offs_zn[None, :] < N)
    
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)
    s = tl.load(s_ptrs, mask=s_mask, other=1.0)
    z = tl.load(z_ptrs, mask=z_mask, other=0.0)
    x = tl_math.round(x / s + z)
    x = tl.where(x < -2 ** (quantize_bit - 1), -2 ** (quantize_bit - 1), x)
    x = tl.where(x > 2 ** (quantize_bit - 1) - 1, 2 ** (quantize_bit - 1) - 1, x)
    x = x + 2 ** (quantize_bit - 1)
    tl.store(x_temp_ptrs, x, mask=x_temp_mask)
    
    offs_x_slice_m = offs_xm
    offs_x_slice_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    x_slice_ptrs = x_temp_ptr + stride_xm * offs_x_slice_m[:, None] + stride_xn * offs_x_slice_n[None, :] + offs_b * stride_xb
    x_slice_mask = (offs_b < B) & (offs_x_slice_m[:, None] < M) & (offs_x_slice_n[None, :] < N)
    
    for i in range(elem_per_position):
        x_slice_ptrs_new = x_slice_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        element_fake_int = tl.load(x_slice_ptrs_new, mask=x_slice_mask, other=0.0).to(tl.float32) 
        element_int = tl_math.float2uint_rn(element_fake_int)
        q |= (element_int << (quantize_bit * i)).to(tl.uint8)
    
    tl.store(q_ptrs, q, mask=q_mask)
    
    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8, }, num_stages=1, 
                      num_warps=2), #! config: 1bit, 2; 2bit, 2; 4bit, 4; 8bit, 4
    ],
    key=['M', 'N'],
)
@triton.jit
def decompression_dequantization_with_zero_point_kernel(
    # Pointers to matrices
    x_ptr, x_temp_ptr, q_ptr, s_ptr, z_ptr,
    # Matrix dimensions
    B, M, N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_xb, stride_xm, stride_xn,
    stride_x_tempb, stride_x_tempm, stride_x_tempn,
    stride_qb, stride_qm, stride_qn,
    stride_sb, stride_sn, # scaling factor has no stride in the m dimension
    stride_zb, stride_zn,
    # compress params
    quantize_bit: tl.constexpr, elem_per_position: tl.constexpr, 
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr, 
):
    # basic pointers
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = x_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)

    # quantization operators
    offs_qm = offs_xm
    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q_ptrs = q_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q_mask = (offs_b < B) & (offs_qm[:, None] < M) & (offs_qn[None, :] < N // elem_per_position)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    offs_x_temp_m = offs_xm
    offs_x_temp_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    x_temp_ptrs = x_temp_ptr + stride_x_tempm * offs_x_temp_m[:, None] + stride_x_tempn * offs_x_temp_n[None, :] + offs_b * stride_x_tempb

    mask = (1 << quantize_bit) - 1

    # extract the quantized values
    for i in range(elem_per_position):
        x_temp_ptrs_new = x_temp_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        element_fake_int = tl_math.uint2float_rn((q & mask).to(tl.uint32))
        tl.store(x_temp_ptrs_new, element_fake_int)
        q = (q >> quantize_bit).to(tl.uint8)
        
    # dequantize
    offs_sn = offs_xn
    s_ptrs = s_ptr + stride_sn * offs_sn[None, :] 
    s_mask = (offs_sn[None, :] < N)
    s = tl.load(s_ptrs, mask=s_mask, other=1.0)
    
    offs_zn = offs_xn
    z_ptrs = z_ptr + stride_zn * offs_zn[None, :]
    z_mask = (offs_zn[None, :] < N)
    z = tl.load(z_ptrs, mask=z_mask, other=0.0)
    
    offs_x_temp_m = offs_xm
    offs_x_temp_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_temp_ptrs = x_temp_ptr + stride_xm * offs_x_temp_m[:, None] + stride_xn * offs_x_temp_n[None, :] + offs_b * stride_xb
    
    x = tl.load(x_temp_ptrs, mask=x_mask, other=0.0)
    x = x - 2 ** (quantize_bit - 1)
    x = x.to(tl.bfloat16)
    x = (x - z) * s
    x = x.to(tl.bfloat16)
    tl.store(x_ptrs, x, mask=x_mask)