#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "torch>=2.8.0",
#     "triton>=3.4.0",
#     "einops>=0.8.1",
#     "scipy>=1.16.2",
#     "tqdm>=4.67.1",
# ]
# ///

"""
Fast Linear Scale Invariant HiPPO in Triton.

Credit to https://github.com/state-spaces/s4/blob/main/notebooks/hippo_function_approximation.ipynb for the original implementation.
Note that this script requires a GPU.
If you don't have a GPU, you can run this on a GPU-powered Modal notebook within your browser:
https://modal.com/notebooks/andrewhinh/_/nb-xUjgmzT8a6Tcb86SQUTOv7

# Setup
chmod +x hippo.py

# Use
./hippo.py  # requires GPU
"""

import math
import time
from collections import defaultdict

import numpy as np
import scipy.linalg as la
import scipy.special as ss
import torch
import torch.nn as nn
import triton
import triton.language as tl
from tqdm import tqdm

seed = 0
torch.manual_seed(seed)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
device = "cuda"


# original


def shift_up(a, s=None, drop=True, dim=0):
    assert dim == 0
    if s is None:
        s = torch.zeros_like(a[0, ...])
    s = s.unsqueeze(dim)
    if drop:
        a = a[:-1, ...]
    return torch.cat((s, a), dim=dim)


def batch_mult(A, u, has_batch=None):
    """Matrix mult A @ u with special case to save memory if u has additional batch dim

    The batch dimension is assumed to be the second dimension
    A : (L, ..., N, N)
    u : (L, [B], ..., N)
    has_batch: True, False, or None. If None, determined automatically

    Output:
    x : (L, [B], ..., N)
      A @ u broadcasted appropriately
    """

    if has_batch is None:
        has_batch = len(u.shape) >= len(A.shape)

    if has_batch:
        u = u.permute([0] + list(range(2, len(u.shape))) + [1])
    else:
        u = u.unsqueeze(-1)
    v = A @ u
    if has_batch:
        v = v.permute([0] + [len(u.shape) - 1] + list(range(1, len(u.shape) - 1)))
    else:
        v = v[..., 0]
    return v


def interleave(a, b, uneven=False, dim=0):
    """Interleave two tensors of same shape"""
    # assert(a.shape == b.shape)
    assert dim == 0  # TODO temporary to make handling uneven case easier
    if dim < 0:
        dim = N + dim
    if uneven:
        a_ = a[-1:, ...]
        a = a[:-1, ...]
    c = torch.stack((a, b), dim + 1)
    out_shape = list(a.shape)
    out_shape[dim] *= 2
    c = c.view(out_shape)
    if uneven:
        c = torch.cat((c, a_), dim=dim)
    return c


def variable_unroll_general_sequential(A, u, s, op, variable=True):
    """Unroll with variable (in time/length) transitions A with general associative operation

    A : ([L], ..., N, N) dimension L should exist iff variable is True
    u : (L, [B], ..., N) updates
    s : ([B], ..., N) start state
    output : x (..., N)
    x[i, ...] = A[i]..A[0] s + A[i..1] u[0] + ... + A[i] u[i-1] + u[i]
    """

    if not variable:
        A = A.expand((u.shape[0],) + A.shape)

    outputs = []
    for A_, u_ in zip(torch.unbind(A, dim=0), torch.unbind(u, dim=0)):
        s = op(A_, s)
        s = s + u_
        outputs.append(s)

    output = torch.stack(outputs, dim=0)
    return output


def variable_unroll_general(
    A, u, s, op, compose_op=None, sequential_op=None, variable=True, recurse_limit=16
):
    """Bottom-up divide-and-conquer version of variable_unroll.

    compose is an optional function that defines how to compose A without multiplying by a leaf u
    """

    if u.shape[0] <= recurse_limit:
        if sequential_op is None:
            sequential_op = op
        return variable_unroll_general_sequential(A, u, s, sequential_op, variable)

    if compose_op is None:
        compose_op = op

    uneven = u.shape[0] % 2 == 1
    # has_batch = len(u.shape) >= len(A.shape)

    u_0 = u[0::2, ...]
    u_1 = u[1::2, ...]

    if variable:
        A_0 = A[0::2, ...]
        A_1 = A[1::2, ...]
    else:
        A_0 = A
        A_1 = A

    u_0_ = u_0
    A_0_ = A_0
    if uneven:
        u_0_ = u_0[:-1, ...]
        if variable:
            A_0_ = A_0[:-1, ...]

    u_10 = op(A_1, u_0_)  # batch_mult(A_1, u_0_, has_batch)
    u_10 = u_10 + u_1
    A_10 = compose_op(A_1, A_0_)

    # Recursive call
    x_1 = variable_unroll_general(
        A_10,
        u_10,
        s,
        op,
        compose_op,
        sequential_op,
        variable=variable,
        recurse_limit=recurse_limit,
    )

    x_0 = shift_up(x_1, s, drop=not uneven)
    x_0 = op(A_0, x_0)  # batch_mult(A_0, x_0, has_batch)
    x_0 = x_0 + u_0

    x = interleave(
        x_0, x_1, uneven, dim=0
    )  # For some reason this interleave is slower than in the (non-variable) unroll_recursive
    return x


def variable_unroll_matrix(A, u, s=None, variable=True, recurse_limit=16):
    if s is None:
        s = torch.zeros_like(u[0])
    has_batch = len(u.shape) >= len(A.shape)

    def op(x, y):
        return batch_mult(x, y, has_batch)

    def sequential_op(x, y):
        return batch_mult(x.unsqueeze(0), y.unsqueeze(0), has_batch)[0]

    return variable_unroll_general(
        A,
        u,
        s,
        op,
        compose_op=torch.matmul,
        sequential_op=sequential_op,
        variable=variable,
        recurse_limit=recurse_limit,
    )


def scaled_legendre_transition(N):
    q = np.arange(N, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1
    M = -(np.where(row >= col, r, 0) - np.diag(q))
    T = np.sqrt(np.diag(2 * q + 1))
    A = T @ M @ np.linalg.inv(T)
    B = np.diag(T)[:, None]
    B = B.copy()  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    return A, B


class HiPPOScale(nn.Module):
    def __init__(
        self,
        N: int,
        L: int,
        discretization: str,
    ):
        super().__init__()
        self.N = N
        A, B = scaled_legendre_transition(N)
        B = np.asarray(B, dtype=A.dtype).reshape(-1)  # (N,)
        A_stacked = np.empty((L, N, N), dtype=A.dtype)
        B_stacked = np.empty((L, N), dtype=A.dtype)
        eye_mat = np.eye(N, dtype=A.dtype)
        for t in range(1, L + 1):
            At = A / t
            Bt = B / t  # (N,)
            if discretization == "forward":
                A_stacked[t - 1] = eye_mat + At
                B_stacked[t - 1] = Bt
            elif discretization == "backward":
                M = eye_mat - At
                A_stacked[t - 1] = la.solve(M, eye_mat)
                B_stacked[t - 1] = la.solve(M, Bt)
            elif discretization == "bilinear":
                M = eye_mat - 0.5 * At
                Nn = eye_mat + 0.5 * At
                A_stacked[t - 1] = la.solve(M, Nn)
                B_stacked[t - 1] = la.solve(M, Bt)
            elif discretization == "zoh":
                delta = math.log(t + 1.0) - math.log(t)
                A_d = la.expm(A * delta)
                # exp([A B; 0 0] Δ) = [A_d B_d; 0 1]
                Z = np.zeros((N + 1, N + 1), dtype=A.dtype)
                Z[:N, :N] = A
                Z[:N, N] = B  # B is 1-D
                Ez = la.expm(Z * delta)
                A_stacked[t - 1] = A_d
                B_stacked[t - 1] = Ez[:N, N]  # (N,)
            else:
                raise ValueError(f"Unknown discretization: {discretization}")
        self.register_buffer("A_stacked", torch.tensor(A_stacked))
        self.register_buffer("B_stacked", torch.tensor(B_stacked))

        vals = np.linspace(0.0, 1.0, L)
        self.eval_matrix = torch.tensor(
            (B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T
        )

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        L = inputs.shape[0]

        inputs = inputs.unsqueeze(-1)
        u = torch.transpose(inputs, 0, -2)
        u = u * self.B_stacked[:L]
        u = torch.transpose(u, 0, -2)  # (length, ..., N)

        return variable_unroll_matrix(self.A_stacked[:L], u)

    def reconstruct(self, c):
        a = self.eval_matrix.to(c) @ c.unsqueeze(-1)
        return a


# Torch + Triton


@triton.jit
def _hippo_unroll_block_kernel(
    A_ptr,  # float* [L, N, N]
    U_ptr,  # float* [L, B, N]
    S_ptr,  # float* [B, N] (start state for block; updated to last state)
    C_ptr,  # float* [L, B, N] (output states)
    N: tl.constexpr,  # state dim <= BLOCK_N
    t0,  # block start (int)
    T,  # block length (int) <= BLOCK_T
    # strides in # elements
    stride_a_l,
    stride_a_i,
    stride_a_j,
    stride_u_l,
    stride_u_b,
    stride_u_n,
    stride_c_l,
    stride_c_b,
    stride_c_n,
    stride_s_b,
    BLOCK_N: tl.constexpr,  # tile size
    BLOCK_T: tl.constexpr,  # timesteps per kernel (>=L for 1 kernel launch)
):
    # 1 program per batch stream
    bid = tl.program_id(axis=0)
    n = tl.arange(0, BLOCK_N)

    # load starting state for this block
    s_ptr = S_ptr + bid * stride_s_b
    c = tl.load(s_ptr + n, mask=n < N, other=0.0).to(tl.float32)

    j = tl.arange(0, BLOCK_N)  # col idxs for loading A tiles

    for dt in range(0, BLOCK_T):
        active = dt < T

        t = t0 + dt

        # load A_t matrix tile [BLOCK_N, BLOCK_N]
        a_tile = tl.load(
            A_ptr
            + t * stride_a_l
            + (n[:, None] * stride_a_i)
            + (j[None, :] * stride_a_j),
            mask=(n[:, None] < N) & (j[None, :] < N) & active,
            other=0.0,
        ).to(tl.float32)

        # c_new = A_t [BLOCK_N, BLOCK_N] @ c [BLOCK_N]
        c_new = tl.sum(a_tile * c[None, :], axis=1)

        # load update vector u_t (already includes B @ input)
        u_row = tl.load(
            U_ptr + t * stride_u_l + bid * stride_u_b + n * stride_u_n,
            mask=(n < N) & active,
            other=0.0,
        ).to(tl.float32)

        # update state
        c_next = tl.where(active, c_new + u_row, c)

        # store current state for this timestep
        tl.store(
            C_ptr + t * stride_c_l + bid * stride_c_b + n * stride_c_n,
            c_next,
            mask=(n < N) & active,
        )

        c = c_next

    # write back last state of block to S_ptr to feed the next block
    tl.store(s_ptr + n, c, mask=n < N)


def variable_unroll_matrix_triton(
    A: torch.Tensor,
    u: torch.Tensor,
    s: torch.Tensor | None = None,
    block_t: int | None = None,
) -> torch.Tensor:
    """
    Compute x_t = A_t @ x_{t-1} + u_t for t=0..L-1.

    Args:
    A: (L, N, N)
    u: (L, ..., N) updates (already multiplied by B)
    s: optional (..., N) initial state; default zeros
    block_t: timesteps per kernel launch (set >= L to use one launch)

    Returns:
    x: (L, ..., N)
    """
    assert A.device.type == device and u.device.type == device
    assert A.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert u.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert A.ndim == 3 and A.shape[0] == u.shape[0]
    L, N, N2 = A.shape
    assert N == N2, "A must be (L, N, N)"

    # flatten batch dims in u to a single B
    if u.ndim == 2:
        # (L, N) -> (L, 1, N)
        u_flat = u.unsqueeze(1)
        batch_shape = ()
    else:
        # (L, ..., N) -> (L, B, N)
        batch_shape = u.shape[1:-1]
        B = int(math.prod(batch_shape)) if len(batch_shape) > 0 else 1
        u_flat = u.reshape(L, B, N)
    L, B, _ = u_flat.shape

    # prepare initial state s: (..., N) -> (B, N)
    if s is None:
        s_flat = torch.zeros((B, N), device=device, dtype=u.dtype)
    else:
        if s.ndim == 1:
            s_flat = s.expand(B, N)
        else:
            assert s.shape[-1] == N
            if s.shape[:-1] != batch_shape:
                s = s.expand(*batch_shape, N)
            s_flat = s.reshape(B, N).contiguous()

    A_c = A.contiguous()
    u_c = u_flat.contiguous()
    s_buf = s_flat.contiguous()

    out = torch.empty_like(u_c, dtype=torch.float32)  # fp32 for precision

    # strides in # elements
    stride_a_l, stride_a_i, stride_a_j = A_c.stride()
    stride_u_l, stride_u_b, stride_u_n = u_c.stride()
    stride_c_l, stride_c_b, stride_c_n = out.stride()
    stride_s_b = s_buf.stride(0)

    # gpu resources
    BLOCK_N = 64 if N <= 64 else 128 if N <= 128 else 256  # tile sizes (BLOCK_N >= N)
    assert BLOCK_N >= N, f"BLOCK_N must be >= N, got {BLOCK_N} < {N}"
    BLOCK_T = block_t if block_t else L

    # process sequence in blocks along time
    nblocks = (L + BLOCK_T - 1) // BLOCK_T
    for blk in range(nblocks):
        t0 = blk * BLOCK_T
        T = min(BLOCK_T, L - t0)
        grid = (B,)
        _hippo_unroll_block_kernel[grid](
            A_c,
            u_c,
            s_buf,
            out,
            N,
            t0,
            T,
            stride_a_l,
            stride_a_i,
            stride_a_j,
            stride_u_l,
            stride_u_b,
            stride_u_n,
            stride_c_l,
            stride_c_b,
            stride_c_n,
            stride_s_b,
            BLOCK_N,
            T,
        )

    out = out.to(u.dtype)
    # reshape back to (L, ..., N)
    if len(batch_shape) == 0:
        return out[:, 0, :]  # (L, N)
    else:
        return out.reshape((L, *batch_shape, N))


@torch.no_grad()
def legendre_P_matrix(
    x: torch.Tensor, N: int, compute_dtype=torch.float64
) -> torch.Tensor:
    # x: (T,)
    T = x.numel()
    x = x.to(compute_dtype)
    P = torch.empty(T, N, dtype=compute_dtype, device=x.device)
    P[:, 0] = 1.0
    if N > 1:
        P[:, 1] = x
        for n in range(1, N - 1):
            P[:, n + 1] = ((2 * n + 1) * x * P[:, n] - n * P[:, n - 1]) / (n + 1)
    return P


@torch.no_grad()
def build_eval_matrix_scale(
    B: torch.Tensor,
    N: int,
    L: int,
    compute_dtype=torch.float64,
    store_dtype=torch.float32,
) -> torch.Tensor:
    vals = torch.linspace(0.0, 1.0, L, dtype=compute_dtype)  # CPU ok; moves with module
    x = 2.0 * vals - 1.0  # (T,)
    P = legendre_P_matrix(x, N, compute_dtype)  # (T, N)
    Bv = B.to(P.device, P.dtype).view(-1)  # (N,)
    E = P * Bv[None, :]  # (T, N)
    return E.to(store_dtype).contiguous()


class HiPPOScaleFast(nn.Module):
    def __init__(
        self,
        N: int,
        L: int,
        discretization: str,
    ):
        super().__init__()
        self.N = N
        A, B = scaled_legendre_transition(N)
        B = np.asarray(B, dtype=A.dtype).reshape(-1)  # (N,)

        A_cts = (
            torch.as_tensor(A, dtype=torch.float64).to(device).contiguous()
        )  # [N, N]
        B_cts = torch.as_tensor(B, dtype=torch.float64).to(device).contiguous()  # [N]

        t = torch.arange(1, L + 1, device=device, dtype=torch.float64)  # [L]
        eye_mat = torch.eye(N, device=device, dtype=torch.float64).expand(L, N, N)

        A_scaled = A_cts[None, :, :] / t[:, None, None]  # [L, N, N]
        B_scaled = B_cts[None, :] / t[:, None]  # [L, N]

        if discretization == "forward":
            A_d = eye_mat + A_scaled
            B_d = B_scaled
        elif discretization == "backward":
            M = eye_mat - A_scaled
            A_d = torch.linalg.solve(M, eye_mat)
            B_d = torch.linalg.solve(M, B_scaled.unsqueeze(-1)).squeeze(-1)
        elif discretization == "bilinear":
            M = eye_mat - 0.5 * A_scaled
            Nn = eye_mat + 0.5 * A_scaled
            A_d = torch.linalg.solve(M, Nn)
            B_d = torch.linalg.solve(M, B_scaled.unsqueeze(-1)).squeeze(-1)
        elif discretization == "zoh":
            # Δ_t = log((t+1)/t)
            delta = torch.log((t + 1.0) / t)
            # exp([A B; 0 0] Δ) = [A_d B_d; 0 1]
            Z = torch.zeros(L, N + 1, N + 1, device=device, dtype=torch.float64)
            Z[:, :N, :N] = A_cts
            Z[:, :N, N] = B_cts
            E = torch.matrix_exp(Z * delta[:, None, None])
            A_d = E[:, :N, :N]
            B_d = E[:, :N, N]
        else:
            raise ValueError(f"Unknown discretization: {discretization}")

        A_stacked_t, B_stacked_t = A_d.contiguous(), B_d.contiguous()
        self.register_buffer("A_stacked", A_stacked_t.to(torch.float32))
        self.register_buffer("B_stacked", B_stacked_t.to(torch.float32))

        B_vec = torch.as_tensor(B, dtype=torch.float64)
        E = build_eval_matrix_scale(
            B_vec, N, L, compute_dtype=torch.float64, store_dtype=torch.float32
        )
        self.register_buffer("eval_matrix", E)

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        L = inputs.shape[0]

        inputs = inputs.unsqueeze(-1)
        u = torch.transpose(inputs, 0, -2)
        u = u * self.B_stacked[:L]
        u = torch.transpose(u, 0, -2)  # (length, ..., N)

        return variable_unroll_matrix_triton(self.A_stacked[:L], u)

    def reconstruct(self, c):
        a = self.eval_matrix.to(c) @ c.unsqueeze(-1)
        return a


# testing


def whitesignal(period, dt, freq, rms=0.5, batch_shape=()):
    """
    Produces output signal of length period / dt, band-limited to frequency freq
    Output shape (*batch_shape, period/dt)
    Adapted from the nengo library
    """

    if freq is not None and freq < 1.0 / period:
        raise ValueError(
            f"Make ``{freq=} >= 1. / {period=}`` to produce a non-zero signal",
        )

    nyquist_cutoff = 0.5 / dt
    if freq > nyquist_cutoff:
        raise ValueError(
            f"{freq} must not exceed the Nyquist frequency for the given dt ({nyquist_cutoff:0.3f})"
        )

    n_coefficients = int(np.ceil(period / dt / 2.0))
    shape = batch_shape + (n_coefficients + 1,)
    sigma = rms * np.sqrt(0.5)
    coefficients = 1j * np.random.normal(0.0, sigma, size=shape)
    coefficients[..., -1] = 0.0
    coefficients += np.random.normal(0.0, sigma, size=shape)
    coefficients[..., 0] = 0.0

    set_to_zero = np.fft.rfftfreq(2 * n_coefficients, d=dt) > freq
    coefficients *= 1 - set_to_zero
    power_correction = np.sqrt(1.0 - np.sum(set_to_zero, dtype=float) / n_coefficients)
    if power_correction > 0.0:
        coefficients /= power_correction
    coefficients *= np.sqrt(2 * n_coefficients)
    signal = np.fft.irfft(coefficients, axis=-1)
    signal = signal - signal[..., :1]  # Start from 0
    return signal


N = 64
T = 3
dt = 1e-3
freq = 3.0

vals = np.arange(0.0, T, dt)
u = whitesignal(T, dt, freq=freq)
u = torch.tensor(u, dtype=torch.float)
u = u.to(device)

results = defaultdict(list)
n_trials = 1

for d in ["forward", "backward", "bilinear", "zoh"]:
    print(d)
    for i in tqdm(range(n_trials)):
        start_time = time.monotonic_ns()
        hippo = HiPPOScale(N, L=int(T / dt), discretization=d).to(device)
        end_time = time.monotonic_ns()
        results[f"{d}-slow-init"].append(end_time - start_time)

        start_time = time.monotonic_ns()
        hippo = HiPPOScaleFast(N, L=int(T / dt), discretization=d).to(device)
        end_time = time.monotonic_ns()
        results[f"{d}-fast-init"].append(end_time - start_time)

        start_time = time.monotonic_ns()
        out = hippo(u)
        end_time = time.monotonic_ns()
        results[f"{d}-slow-fwd"].append(end_time - start_time)

        start_time = time.monotonic_ns()
        out = hippo(u)
        end_time = time.monotonic_ns()
        results[f"{d}-fast-fwd"].append(end_time - start_time)

        start_time = time.monotonic_ns()
        u_hippo = hippo.reconstruct(out)[-1].cpu()
        end_time = time.monotonic_ns()
        results[f"{d}-slow-reconstruct"].append(end_time - start_time)

        start_time = time.monotonic_ns()
        u_hippo = hippo.reconstruct(out)[-1].cpu()
        end_time = time.monotonic_ns()
        results[f"{d}-fast-reconstruct"].append(end_time - start_time)

for k, v in results.items():
    print(f"{k} p50: {v[n_trials // 2] / 1e6:.2f} ms")
