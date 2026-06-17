from typing import Tuple

import torch


def coil_sensitivity(
    grid_size: Tuple[int, int, int],
    coil_orientation: Tuple[float, float, float],
    coil_position: Tuple[float, float, float],
    fov_scaling: Tuple[float, float, float],
) -> torch.Tensor:
    """
    Compute the coil sensitivity map.

    Coordinate positions:
    (-0.5, -0.5) lower left corner
    (+0.5, +0.5) upper right corner

    Parameters
    ----------
    grid_size : Tuple[int, int, int]
        Grid size in each dimension (nx, ny, nz).
    coil_orientation : Tuple[float, float, float]
        Coil orientation (mx, my, mz).
    coil_position : Tuple[float, float, float]
        Coil position (px, py, pz).
    fov_scaling : Tuple[float, float, float]
        Field of view scaling factors.

    Returns
    -------
    torch.Tensor
        Complex-valued coil sensitivity map of shape (nx, ny, nz).
    """
    sqrt3_over_2 = torch.sqrt(torch.tensor(3.0)) / 2

    # Create meshgrid for spatial positions
    rx, ry, rz = torch.meshgrid(
        torch.linspace(
            -sqrt3_over_2 * fov_scaling[0], sqrt3_over_2 * fov_scaling[0], grid_size[0]
        )
        + coil_position[0],
        torch.linspace(
            -sqrt3_over_2 * fov_scaling[1], sqrt3_over_2 * fov_scaling[1], grid_size[1]
        )
        + coil_position[1],
        torch.linspace(
            -sqrt3_over_2 * fov_scaling[2], sqrt3_over_2 * fov_scaling[2], grid_size[2]
        )
        + coil_position[2],
        indexing="ij",
    )

    # Handle degenerate dimensions where grid size is 1
    rx = torch.where(torch.tensor(grid_size[0]) == 1, torch.zeros_like(rx), rx)
    ry = torch.where(torch.tensor(grid_size[1]) == 1, torch.zeros_like(ry), ry)
    rz = torch.where(torch.tensor(grid_size[2]) == 1, torch.zeros_like(rz), rz)

    # Compute radial distance
    r = torch.sqrt(rx**2 + ry**2 + rz**2)

    # Compute magnetic field components
    dot_product = (
        coil_orientation[0] * rx + coil_orientation[1] * ry + coil_orientation[2] * rz
    )
    Bx = 3 * rx * dot_product / r**5 - coil_orientation[0] / r**3
    By = 3 * ry * dot_product / r**5 - coil_orientation[1] / r**3

    # Compute complex-valued sensitivity map
    sensitivity_map = -Bx + 1j * By

    return sensitivity_map


def cyclic_head_coil(
    coil_config: Tuple[int, int, int, int],
    fov_scaling: Tuple[float, float, float] = (1, 1, 1),
    coil_radius: float = 2,
    coil_position_z: float = 0,
    coil_orientation_z: float = 0,
    phase_offset: float = 0,
) -> torch.Tensor:
    """
    Generate the coil sensitivity map for a cyclic head coil configuration.

    Parameters
    ----------
    coil_config : Tuple[int, int, int, int]
        Number of coils and grid size (num_coils, nx, ny, nz).
    fov_scaling : Tuple[float, float, float]
        Field of view scaling factors.
    coil_radius : float
        Radius of cyclic head coil.
    coil_position_z : float
        Coil position along the z-axis.
    coil_orientation_z : float
        Coil orientation along the z-axis.
    phase_offset : float
        Phase offset for coil positioning.

    Returns
    -------
    torch.Tensor
        Complex-valued coil sensitivity map of shape (num_coils, nx, ny, nz).
    """
    num_coils, nx, ny, nz = coil_config

    # Initialize the coil sensitivity map as a complex tensor
    sensitivity_map = torch.zeros(coil_config, dtype=torch.complex128)

    # Generate phase angles for coil positioning
    phase_angles = torch.linspace(0, 2 * torch.pi, num_coils + 1)[:-1] + phase_offset

    # Compute coil sensitivity for each coil
    for coil_idx, phi in enumerate(phase_angles):
        # Define coil orientation
        coil_orientation = (torch.cos(phi), torch.sin(phi), coil_orientation_z)
        # Define coil position
        coil_position = (
            torch.cos(phi) * coil_radius,
            torch.sin(phi) * coil_radius,
            coil_position_z,
        )

        # Compute coil sensitivity map for this coil
        sensitivity_map[coil_idx, :, :, :] = coil_sensitivity(
            (nx, ny, nz),
            coil_orientation,
            coil_position,
            fov_scaling,
        )

    return sensitivity_map
