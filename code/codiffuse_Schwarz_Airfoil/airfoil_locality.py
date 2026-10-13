"""Strictly local physical conditions for Airfoil subdomain operators.

The historical transition condition contains absolute coordinates, tile
centres, and a crop of a globally evaluated SDF.  Those are useful for an
ordinary patch predictor but violate the transport-DDM locality contract.  The
helpers here expose only fields measurable inside one core-plus-halo patch and
PDE constants shared by every patch.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


Tensor = torch.Tensor

STRICT_LOCALITY_CONTRACT = "airfoil_patch_local_physics_v3_native_mesh"
STRICT_LOCAL_SCALAR_NAMES = (
    "log_reynolds_over_10000",
    "dt_over_0p01",
    "dt_over_dx",
    "diffusive_cfl_times_100",
)


def local_signed_distance_from_fluid_2d(
    fluid: Tensor,
    radius_cells: float = 4.0,
) -> tuple[Tensor, Tensor]:
    """Return a truncated signed distance using only per-patch occupancy.

    ``fluid`` has shape ``[N,1,H,W]``.  A positive value denotes fluid.  The
    distance is a deterministic Chebyshev-distance approximation computed by
    local morphological dilation.  Replicate padding ensures that an
    artificial patch face is not mistaken for body geometry.  Geometry-free
    fluid patches are exactly ``+1`` and reveal no distance to an airfoil in a
    different patch.

    The second return value is one scalar mask per patch indicating that both
    fluid and solid occur locally.
    """
    if fluid.ndim != 4 or fluid.shape[1] != 1:
        raise ValueError("fluid occupancy must have shape N,1,H,W")
    if not math.isfinite(radius_cells) or radius_cells <= 0.0:
        raise ValueError("local SDF radius must be positive and finite")
    label = fluid > 0.5
    padded = F.pad(label.to(torch.float32), (1, 1, 1, 1), mode="replicate")
    local_max = F.max_pool2d(padded, 3, stride=1)
    local_min = 1.0 - F.max_pool2d(1.0 - padded, 3, stride=1)
    boundary = local_max > local_min
    has_interface = (
        label.flatten(1).any(1) & (~label).flatten(1).any(1)
    )[:, None, None, None]

    # Distance is needed only inside a fixed narrow band.  Repeated 3x3
    # dilation is inexpensive for the 40x40 local Airfoil patches and avoids
    # consulting a global Euclidean distance transform.
    distance = torch.full_like(fluid, float(radius_cells))
    reached = boundary
    distance = torch.where(reached, torch.zeros_like(distance), distance)
    steps = int(math.ceil(radius_cells))
    for step in range(1, steps + 1):
        reached_next = F.max_pool2d(
            reached.to(torch.float32), 3, stride=1, padding=1,
        ) > 0.0
        newly_reached = reached_next & ~reached
        distance = torch.where(
            newly_reached,
            torch.full_like(distance, min(float(step), radius_cells)),
            distance,
        )
        reached = reached_next
    normalized = (distance / radius_cells).clamp(0.0, 1.0)
    signed = torch.where(label, normalized, -normalized).to(fluid.dtype)
    signed = torch.where(
        has_interface,
        signed,
        torch.where(label, torch.ones_like(signed), -torch.ones_like(signed)),
    )
    return signed, has_interface.to(dtype=fluid.dtype)


def strict_local_conditions(
    transition,
    current_fraction: Tensor,
    phase_fraction: Tensor,
    *,
    radius_cells: float = 4.0,
) -> tuple[Tensor, Tensor]:
    """Build coordinate-free, geometry-local conditions for one transition.

    The physical channel order is

    ``q_n(3), sdf_n, sdf_np1, [mesh_n, mesh_np1], wall_n(2), wall_np1(2),``
    ``[-dt*gradp_n(2)], [q_n-q_nm1(3)], valid``.

    Bracketed blocks are present when the dataset stores the corresponding
    discrete solver state or requests two-step history.  Wall velocity is a
    prescribed *local boundary value*: it is zero in every patch that does not
    contain an airfoil interface.
    """
    if current_fraction.shape != (transition.batch_size,) or (
        phase_fraction.shape != (transition.batch_size,)
    ):
        raise ValueError("fractions must contain one value per transition")
    current, endpoint, sdf_current, sdf_endpoint = transition.geometry(
        current_fraction,
    )
    cover = transition.cover
    # The stored Airfoil window is an observation crop through a larger CFD
    # domain, not a physical wall.  With no guard solution available, use the
    # local one-sided (zero-normal-gradient) continuation rather than filling
    # the halo with the normalized value zero.  The separate validity channel
    # still tells the operator which cells are ghosts.  This construction uses
    # only the boundary value already inside the crop.
    padding_mode = "replicate"

    previous = cover.tile(transition.q_normalized, padding_mode=padding_mode)
    sdf_current_crop = cover.tile(
        sdf_current[:, None], padding_mode="replicate",
    )
    sdf_endpoint_crop = cover.tile(
        sdf_endpoint[:, None], padding_mode="replicate",
    )
    local_current, current_has_interface = local_signed_distance_from_fluid_2d(
        (sdf_current_crop > 0.0).to(sdf_current_crop), radius_cells,
    )
    local_endpoint, endpoint_has_interface = local_signed_distance_from_fluid_2d(
        (sdf_endpoint_crop > 0.0).to(sdf_endpoint_crop), radius_cells,
    )

    # Compute the prescribed wall velocity analytically, then expose it only
    # in the narrow band around geometry that occurs inside this patch.  The
    # operator never sees xx/yy themselves.
    omega_current = current[:, 1, None, None]
    omega_endpoint = endpoint[:, 1, None, None]
    wall_global = torch.stack((
        -omega_current * transition.yy,
        omega_current * transition.xx,
        -omega_endpoint * transition.yy,
        omega_endpoint * transition.xx,
    ), dim=1)
    wall = cover.tile(wall_global, padding_mode="replicate")
    current_band = torch.exp(
        -local_current.abs() * radius_cells / 4.0,
    ) * current_has_interface
    endpoint_band = torch.exp(
        -local_endpoint.abs() * radius_cells / 4.0,
    ) * endpoint_has_interface
    wall = torch.cat((
        wall[:, :2] * current_band,
        wall[:, 2:] * endpoint_band,
    ), dim=1)

    valid = cover.valid_tiles(
        transition.batch_size,
        transition.device,
        transition.q_n.dtype,
    )
    fields = [
        previous,
        local_current,
        local_endpoint,
    ]
    native_level_n = getattr(transition, "native_level_n", None)
    native_level_np1 = getattr(transition, "native_level_np1", None)
    if native_level_n is not None:
        if native_level_np1 is None:
            raise RuntimeError("endpoint native AMR level is missing")
        # One means that the raster cell was solved natively at this level;
        # 1/2, 1/4, ... identify values prolonged from successively coarser
        # local discrete operators.  This is mesh metadata, not a global
        # coordinate prompt.
        level = float(transition.dataset.level)
        fields.extend((
            cover.tile(
                torch.exp2(native_level_n - level), padding_mode="replicate",
            ),
            cover.tile(
                torch.exp2(native_level_np1 - level), padding_mode="replicate",
            ),
        ))
    fields.append(wall)
    gradp_increment = getattr(transition, "gradp_increment_normalized", None)
    if gradp_increment is not None:
        fields.append(cover.tile(
            gradp_increment, padding_mode=padding_mode,
        ))
    if getattr(transition.dataset, "history_steps", 1) == 2:
        fields.append(cover.tile(
            transition.history_normalized, padding_mode=padding_mode,
        ))
    fields.append(valid)
    # Preserve locally extrapolated ghost values as boundary data.  Only the
    # returned solution is masked by ``valid``; erasing every input ghost here
    # would silently impose an arbitrary normalized Dirichlet condition.
    physical = torch.cat(fields, dim=1)

    params = transition.dataset.physical_parameters
    dt = transition.time_np1 - transition.time_n
    viscosity = 1.0 / float(params["reynolds"])
    common = torch.stack((
        torch.full_like(dt, math.log(float(params["reynolds"]) / 10_000.0)),
        dt / 0.01,
        dt / transition.dataset.dx,
        100.0 * viscosity * dt / (transition.dataset.dx**2),
    ), dim=1)
    scalar = common[:, None].expand(
        -1, cover.tile_count, -1,
    ).reshape(transition.batch_size * cover.tile_count, -1)
    return physical, scalar


def strict_local_endpoint_conditions(
    transition,
    *,
    radius_cells: float = 4.0,
) -> tuple[Tensor, Tensor]:
    """Return source/destination conditions for a complete local time step."""
    source = torch.zeros_like(transition.time_n)
    endpoint = torch.ones_like(transition.time_n)
    return strict_local_conditions(
        transition, source, endpoint, radius_cells=radius_cells,
    )
