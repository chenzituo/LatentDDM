"""Low-order relative-coordinate charts for learned subdomain boundary inputs."""

from __future__ import annotations

import torch


Tensor = torch.Tensor


def decode_trace_code_2d(
    code: Tensor,
    roles: Tensor,
    *,
    channels: int = 3,
    modes: int = 2,
) -> Tensor:
    """Decode low-order tangential polynomial modes on four faces."""
    if roles.ndim != 4 or roles.shape[1] != 4:
        raise ValueError("2-D roles must have shape B,4,H,W")
    if modes <= 0:
        raise ValueError("2-D trace basis requires positive mode count")
    expected = 4 * channels * modes
    if code.shape != (roles.shape[0], expected):
        raise ValueError(f"2-D trace code must have shape B,{expected}")
    height, width = roles.shape[-2:]
    y = torch.linspace(-1.0, 1.0, height, device=code.device, dtype=code.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=code.device, dtype=code.dtype)
    coefficients = code.reshape(code.shape[0], 4, channels, modes)
    blocks = []
    for face in range(4):
        tangent = x[None, None, None, :] if face < 2 else y[None, None, :, None]
        basis = torch.cat(
            [tangent.pow(degree) for degree in range(modes)], dim=1,
        )
        values = torch.einsum(
            "bcm,bmhw->bchw", coefficients[:, face],
            basis.expand(code.shape[0], -1, height, width),
        )
        blocks.append(values * roles[:, face:face + 1])
    return torch.cat(blocks, dim=1)


def project_trace_code_2d(
    trace: Tensor,
    roles: Tensor,
    *,
    channels: int = 3,
    modes: int = 2,
    ridge: float = 1.0e-6,
) -> Tensor:
    """Least-squares projection onto the 2-D face trace chart."""
    if trace.ndim != 4 or trace.shape[1] != 4 * channels:
        raise ValueError("2-D trace must have shape B,4*channels,H,W")
    if roles.shape != (trace.shape[0], 4, *trace.shape[-2:]):
        raise ValueError("2-D trace roles do not align")
    if modes <= 0 or ridge <= 0:
        raise ValueError("invalid 2-D trace projection controls")
    output_dtype = trace.dtype
    work_dtype = torch.float64 if trace.dtype == torch.float64 else torch.float32
    # Least-squares solves are both unsupported and poorly conditioned in
    # BF16/FP16.  Disable an enclosing training autocast explicitly, solve in
    # FP32 (or preserve FP64), and cast the differentiable result back.
    with torch.autocast(device_type=trace.device.type, enabled=False):
        work_trace = trace.to(work_dtype)
        work_roles = roles.to(work_dtype)
        batch, _, height, width = work_trace.shape
        y = torch.linspace(
            -1.0, 1.0, height, device=trace.device, dtype=work_dtype,
        )
        x = torch.linspace(
            -1.0, 1.0, width, device=trace.device, dtype=work_dtype,
        )
        coefficients = []
        eye = torch.eye(modes, device=trace.device, dtype=work_dtype)[None]
        for face in range(4):
            tangent = (
                x[None, None, :].expand(batch, height, width)
                if face < 2 else
                y[None, :, None].expand(batch, height, width)
            )
            design = torch.stack(
                [tangent.pow(degree) for degree in range(modes)], dim=-1,
            ).reshape(batch, -1, modes)
            weight = work_roles[:, face].reshape(batch, -1)
            values = work_trace[
                :, face * channels:(face + 1) * channels,
            ].flatten(2).transpose(1, 2)
            gram = torch.einsum("bnm,bn,bnk->bmk", design, weight, design)
            rhs = torch.einsum("bnm,bn,bnc->bmc", design, weight, values)
            scale = torch.diagonal(
                gram, dim1=-2, dim2=-1,
            ).mean(-1).clamp_min(1.0)
            solution = torch.linalg.solve(
                gram + ridge * scale[:, None, None] * eye,
                rhs,
            ).transpose(1, 2)
            coefficients.append(solution)
        projected = torch.stack(coefficients, dim=1).reshape(
            batch, 4 * channels * modes,
        )
    return projected.to(output_dtype)


def decode_trace_code_3d(
    code: Tensor,
    roles: Tensor,
    *,
    channels: int = 4,
    order: int = 1,
) -> Tensor:
    """Decode tensor-polynomial relative-coordinate modes on cuboid faces."""
    if roles.ndim != 5 or roles.shape[1] != 6:
        raise ValueError("3-D roles must have shape B,6,D,H,W")
    if order not in (1, 2, 3, 4, 5):
        raise ValueError("3-D face chart order must be between 1 and 5")
    face_modes = (order + 1) ** 2
    expected = 6 * channels * face_modes
    if code.shape != (roles.shape[0], expected):
        raise ValueError(f"3-D trace code must have shape B,{expected}")
    depth, height, width = roles.shape[-3:]
    z = torch.linspace(-1.0, 1.0, depth, device=code.device, dtype=code.dtype)
    y = torch.linspace(-1.0, 1.0, height, device=code.device, dtype=code.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=code.device, dtype=code.dtype)
    coefficients = code.reshape(code.shape[0], 6, channels, face_modes)
    blocks = []
    for face in range(6):
        if face < 2:  # z-normal; tangential y,x
            first = y[None, None, None, :, None]
            second = x[None, None, None, None, :]
        elif face < 4:  # y-normal; tangential z,x
            first = z[None, None, :, None, None]
            second = x[None, None, None, None, :]
        else:  # x-normal; tangential z,y
            first = z[None, None, :, None, None]
            second = y[None, None, None, :, None]
        first = first.expand(code.shape[0], 1, depth, height, width)
        second = second.expand(code.shape[0], 1, depth, height, width)
        pairs = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for degree in range(2, order + 1):
            pairs.extend(
                (first_degree, second_degree)
                for first_degree in range(degree + 1)
                for second_degree in range(degree + 1)
                if max(first_degree, second_degree) == degree
            )
        basis = torch.cat(
            [first.pow(p) * second.pow(q) for p, q in pairs], dim=1,
        )
        values = torch.einsum(
            "bcm,bmzyx->bczyx", coefficients[:, face], basis,
        )
        blocks.append(values * roles[:, face:face + 1])
    return torch.cat(blocks, dim=1)


def project_trace_code_3d(
    trace: Tensor,
    roles: Tensor,
    *,
    channels: int = 4,
    order: int = 1,
    ridge: float = 1.0e-6,
) -> Tensor:
    """Least-squares projection onto cuboid-face tensor-polynomial charts."""
    if trace.ndim != 5 or trace.shape[1] != 6 * channels:
        raise ValueError("3-D trace must have shape B,6*channels,D,H,W")
    if roles.shape != (trace.shape[0], 6, *trace.shape[-3:]):
        raise ValueError("3-D trace roles do not align")
    if order not in (1, 2, 3, 4, 5) or ridge <= 0:
        raise ValueError("3-D trace projection ridge must be positive")
    output_dtype = trace.dtype
    # Degree-four/five monomial normal systems are noticeably more
    # ill-conditioned than the historical low-order charts.  They are tiny
    # (at most 36x36), so solve those projection systems in float64 and cast
    # the resulting coordinates back to the caller's dtype.
    work_dtype = (
        torch.float64
        if trace.dtype == torch.float64 or order >= 4 else torch.float32
    )
    with torch.autocast(device_type=trace.device.type, enabled=False):
        work_trace = trace.to(work_dtype)
        work_roles = roles.to(work_dtype)
        face_modes = (order + 1) ** 2
        batch, _, depth, height, width = work_trace.shape
        z = torch.linspace(
            -1.0, 1.0, depth, device=trace.device, dtype=work_dtype,
        )
        y = torch.linspace(
            -1.0, 1.0, height, device=trace.device, dtype=work_dtype,
        )
        x = torch.linspace(
            -1.0, 1.0, width, device=trace.device, dtype=work_dtype,
        )
        coefficients = []
        eye = torch.eye(
            face_modes, device=trace.device, dtype=work_dtype,
        )[None]
        for face in range(6):
            if face < 2:
                first = y[None, None, :, None].expand(
                    batch, depth, height, width,
                )
                second = x[None, None, None, :].expand(
                    batch, depth, height, width,
                )
            elif face < 4:
                first = z[None, :, None, None].expand(
                    batch, depth, height, width,
                )
                second = x[None, None, None, :].expand(
                    batch, depth, height, width,
                )
            else:
                first = z[None, :, None, None].expand(
                    batch, depth, height, width,
                )
                second = y[None, None, :, None].expand(
                    batch, depth, height, width,
                )
            pairs = [(0, 0), (1, 0), (0, 1), (1, 1)]
            for degree in range(2, order + 1):
                pairs.extend(
                    (first_degree, second_degree)
                    for first_degree in range(degree + 1)
                    for second_degree in range(degree + 1)
                    if max(first_degree, second_degree) == degree
                )
            design = torch.stack(
                [first.pow(p) * second.pow(q) for p, q in pairs], dim=-1,
            ).reshape(batch, -1, face_modes)
            weight = work_roles[:, face].reshape(batch, -1)
            values = work_trace[
                :, face * channels:(face + 1) * channels,
            ].flatten(2).transpose(1, 2)
            gram = torch.einsum("bnm,bn,bnk->bmk", design, weight, design)
            rhs = torch.einsum("bnm,bn,bnc->bmc", design, weight, values)
            scale = torch.diagonal(
                gram, dim1=-2, dim2=-1,
            ).mean(-1).clamp_min(1.0)
            solution = torch.linalg.solve(
                gram + ridge * scale[:, None, None] * eye, rhs,
            ).transpose(1, 2)
            coefficients.append(solution)
        projected = torch.stack(coefficients, dim=1).reshape(
            batch, 6 * channels * face_modes,
        )
    return projected.to(output_dtype)


def trace_relative_square(
    prediction: Tensor,
    target: Tensor,
    roles: Tensor,
    channels: int,
) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("trace prediction and target must align")
    active = roles.repeat_interleave(channels, dim=1)
    numerator = ((prediction - target).square() * active).sum()
    denominator = (target.square() * active).sum().clamp_min(1.0e-8)
    return numerator / denominator
