import torch

from .boundary_trace_basis import (
    decode_trace_code_2d,
    decode_trace_code_3d,
    project_trace_code_2d,
    project_trace_code_3d,
)


def test_zero_trace_codes_decode_exactly_to_zero() -> None:
    roles2 = torch.ones(2, 4, 7, 9)
    roles3 = torch.ones(2, 6, 5, 7, 9)
    assert torch.count_nonzero(decode_trace_code_2d(torch.zeros(2, 24), roles2)) == 0
    assert torch.count_nonzero(decode_trace_code_3d(torch.zeros(2, 96), roles3)) == 0


def test_constant_face_coefficients_respect_roles() -> None:
    roles = torch.zeros(1, 4, 5, 7)
    roles[:, 0, :2] = 1
    code = torch.zeros(1, 24)
    code.reshape(1, 4, 3, 2)[:, 0, :, 0] = 2.0
    trace = decode_trace_code_2d(code, roles)
    torch.testing.assert_close(
        trace[:, :3], 2.0 * roles[:, :1].expand(-1, 3, -1, -1),
    )
    assert torch.count_nonzero(trace[:, 3:]) == 0


def test_cubic_face_chart_projection_reconstructs_its_range() -> None:
    roles = torch.zeros(2, 4, 9, 11)
    roles[:, 0, :2] = 1
    roles[:, 1, -2:] = 1
    roles[:, 2, :, :2] = 1
    roles[:, 3, :, -2:] = 1
    generator = torch.Generator().manual_seed(17)
    code = torch.randn(2, 4 * 2 * 4, generator=generator)
    trace = decode_trace_code_2d(code, roles, channels=2, modes=4)
    projected = project_trace_code_2d(
        trace, roles, channels=2, modes=4, ridge=1.0e-8,
    )
    reconstructed = decode_trace_code_2d(
        projected, roles, channels=2, modes=4,
    )
    torch.testing.assert_close(reconstructed, trace, rtol=2.0e-5, atol=2.0e-5)


def test_bilinear_cube_face_projection_reconstructs_its_range() -> None:
    roles = torch.zeros(2, 6, 7, 9, 11)
    roles[:, 0, :2] = 1
    roles[:, 1, -2:] = 1
    roles[:, 2, :, :2] = 1
    roles[:, 3, :, -2:] = 1
    roles[:, 4, :, :, :2] = 1
    roles[:, 5, :, :, -2:] = 1
    generator = torch.Generator().manual_seed(19)
    code = torch.randn(2, 6 * 3 * 4, generator=generator)
    trace = decode_trace_code_3d(code, roles, channels=3)
    projected = project_trace_code_3d(
        trace, roles, channels=3, ridge=1.0e-8,
    )
    reconstructed = decode_trace_code_3d(
        projected, roles, channels=3,
    )
    torch.testing.assert_close(reconstructed, trace, rtol=3.0e-5, atol=3.0e-5)


def test_quadratic_cube_face_projection_reconstructs_its_range() -> None:
    roles = torch.zeros(1, 6, 7, 9, 11)
    roles[:, 0, :2] = 1
    roles[:, 1, -2:] = 1
    roles[:, 2, :, :2] = 1
    roles[:, 3, :, -2:] = 1
    roles[:, 4, :, :, :2] = 1
    roles[:, 5, :, :, -2:] = 1
    generator = torch.Generator().manual_seed(23)
    code = torch.randn(1, 6 * 2 * 9, generator=generator)
    trace = decode_trace_code_3d(code, roles, channels=2, order=2)
    projected = project_trace_code_3d(
        trace, roles, channels=2, order=2, ridge=1.0e-8,
    )
    reconstructed = decode_trace_code_3d(
        projected, roles, channels=2, order=2,
    )
    torch.testing.assert_close(reconstructed, trace, rtol=5.0e-5, atol=5.0e-5)


def test_quintic_cube_face_projection_reconstructs_its_range() -> None:
    roles = torch.zeros(1, 6, 10, 10, 10)
    roles[:, 0, :2] = 1
    roles[:, 1, -2:] = 1
    roles[:, 2, :, :2] = 1
    roles[:, 3, :, -2:] = 1
    roles[:, 4, :, :, :2] = 1
    roles[:, 5, :, :, -2:] = 1
    generator = torch.Generator().manual_seed(29)
    code = torch.randn(1, 6 * 2 * 36, generator=generator)
    trace = decode_trace_code_3d(
        code, roles, channels=2, order=5,
    )
    projected = project_trace_code_3d(
        trace, roles, channels=2, order=5, ridge=1.0e-8,
    )
    reconstructed = decode_trace_code_3d(
        projected, roles, channels=2, order=5,
    )
    torch.testing.assert_close(
        reconstructed, trace, rtol=2.0e-4, atol=2.0e-4,
    )


def test_trace_projection_is_safe_inside_bfloat16_autocast() -> None:
    roles2 = torch.ones(1, 4, 5, 7, dtype=torch.bfloat16)
    roles3 = torch.ones(1, 6, 5, 5, 7, dtype=torch.bfloat16)
    trace2 = torch.randn(1, 8, 5, 7).bfloat16()
    trace3 = torch.randn(1, 12, 5, 5, 7).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        code2 = project_trace_code_2d(
            trace2, roles2, channels=2, modes=4,
        )
        code3 = project_trace_code_3d(
            trace3, roles3, channels=2, order=3,
        )
    assert code2.dtype == torch.bfloat16
    assert code3.dtype == torch.bfloat16
    assert torch.isfinite(code2).all()
    assert torch.isfinite(code3).all()
