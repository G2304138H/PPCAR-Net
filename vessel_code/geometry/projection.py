from __future__ import annotations
import math
import numpy as np


def view_features_from_angles(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    phi = np.deg2rad(np.asarray(phi_deg, dtype=np.float64))
    return np.stack([np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)], axis=1).astype(np.float32)


def _stage2_camera_parameters(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    *,
    sid_m: float,
    source_to_iso_m: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    phi = np.deg2rad(np.asarray(phi_deg, dtype=np.float64))
    num_views = int(theta.size)
    detector_to_iso = np.full((num_views,), float(sid_m - source_to_iso_m), dtype=np.float64)
    coordinate_change = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    coordinate_change_inverse = np.linalg.inv(coordinate_change)
    sensor = np.zeros((num_views, 3), dtype=np.float64)
    source = np.zeros((num_views, 3), dtype=np.float64)
    local_x = np.zeros((num_views, 3), dtype=np.float64)
    local_y = np.zeros((num_views, 3), dtype=np.float64)
    for index in range(num_views):
        rotation_1 = coordinate_change @ np.asarray(
            [
                [math.cos(theta[index]), -math.sin(theta[index]), 0.0],
                [math.sin(theta[index]), math.cos(theta[index]), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ) @ coordinate_change_inverse
        rotation_2 = coordinate_change @ np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(phi[index]), math.sin(phi[index])],
                [0.0, -math.sin(phi[index]), math.cos(phi[index])],
            ]
        ) @ coordinate_change_inverse
        rotation = rotation_1 @ rotation_2
        sensor[index] = rotation @ np.asarray([0.0, 0.0, detector_to_iso[index]])
        source[index] = -sensor[index] / detector_to_iso[index] * float(source_to_iso_m)
        local_x[index] = rotation @ np.asarray([0.0, 1.0, 0.0])
        local_y[index] = rotation @ np.asarray([-1.0, 0.0, 0.0])
    return sensor, source, local_x, local_y


def project_points_to_image_rc(
    points_m: np.ndarray,
    *,
    view_index: int,
    sensor: np.ndarray,
    source: np.ndarray,
    local_x: np.ndarray,
    local_y: np.ndarray,
    image_dim: int,
    pixel_spacing_mm: float,
) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64)
    normal = np.cross(local_x[view_index], local_y[view_index])
    rays = source[view_index][None, :] - points
    denominator = rays @ normal
    numerator = -((points - sensor[view_index][None, :]) @ normal)
    scale = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=np.abs(denominator) > 1.0e-12,
    )
    plane = points + scale[:, None] * rays
    sensor_width_m = float(pixel_spacing_mm) * float(image_dim) / 1000.0
    x_axis = local_x[view_index]
    y_axis = local_y[view_index]
    lower_fraction = (1.0 - float(image_dim) / 2.0) / float(image_dim)
    upper_fraction = (float(image_dim) - float(image_dim) / 2.0) / float(image_dim)
    local_origin = sensor[view_index] + sensor_width_m * lower_fraction * (x_axis + y_axis)
    local_x_max = sensor[view_index] + sensor_width_m * (upper_fraction * x_axis + lower_fraction * y_axis)
    local_y_max = sensor[view_index] + sensor_width_m * (lower_fraction * x_axis + upper_fraction * y_axis)
    delta = plane - local_origin[None, :]
    projected_x = (delta @ x_axis)[:, None] * x_axis[None, :]
    projected_y = (delta @ y_axis)[:, None] * y_axis[None, :]
    sign_x = np.sign(np.sum((local_x_max - local_origin)[None, :] * projected_x, axis=1))
    sign_y = np.sign(np.sum((local_y_max - local_origin)[None, :] * projected_y, axis=1))
    pixel_x = sign_x * float(image_dim) * np.linalg.norm(projected_x, axis=1) / np.linalg.norm(local_x_max - local_origin)
    pixel_y = sign_y * float(image_dim) * np.linalg.norm(projected_y, axis=1) / np.linalg.norm(local_y_max - local_origin)
    # convert3D_to_pixels returns y=image_dim-pixel_y; the Stage-2 filled-ring
    # rasterizer then uses row=image_dim-y, which simplifies to pixel_y.
    return np.stack([pixel_y, pixel_x], axis=1)


def _shift_mask(mask: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    output = np.zeros_like(mask, dtype=np.bool_)
    source_row_start = max(0, -row_shift)
    source_row_end = min(mask.shape[0], mask.shape[0] - row_shift)
    source_col_start = max(0, -col_shift)
    source_col_end = min(mask.shape[1], mask.shape[1] - col_shift)
    if source_row_end <= source_row_start or source_col_end <= source_col_start:
        return output
    destination_row_start = source_row_start + row_shift
    destination_row_end = source_row_end + row_shift
    destination_col_start = source_col_start + col_shift
    destination_col_end = source_col_end + col_shift
    output[destination_row_start:destination_row_end, destination_col_start:destination_col_end] = mask[
        source_row_start:source_row_end, source_col_start:source_col_end
    ]
    return output


def _binary_closing_disk(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    offsets = [
        (row, col)
        for row in range(-radius, radius + 1)
        for col in range(-radius, radius + 1)
        if row * row + col * col <= radius * radius
    ]
    dilated = np.zeros_like(mask, dtype=np.bool_)
    for row, col in offsets:
        dilated |= _shift_mask(mask, row, col)
    eroded = np.ones_like(mask, dtype=np.bool_)
    for row, col in offsets:
        eroded &= _shift_mask(dilated, row, col)
    return eroded


def _gaussian_blur(mask: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma))
    kernel /= kernel.sum()
    values = mask.astype(np.float64)
    padded_rows = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    rows = sum(
        float(weight) * padded_rows[index : index + values.shape[0], :]
        for index, weight in enumerate(kernel)
    )
    padded_cols = np.pad(rows, ((0, 0), (radius, radius)), mode="edge")
    return sum(
        float(weight) * padded_cols[:, index : index + values.shape[1]]
        for index, weight in enumerate(kernel)
    )


def render_mesh_views(
    triangles_m_centered: np.ndarray,
    *,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    image_dim: int,
    sid_m: float,
    pixel_spacing_mm: float,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError("Pillow is required to rasterize the CT annotation surface.") from error

    sensor, source, local_x, local_y = _stage2_camera_parameters(
        theta_deg, phi_deg, sid_m=sid_m
    )
    flat_vertices = np.asarray(triangles_m_centered, dtype=np.float64).reshape(-1, 3)
    images: list[np.ndarray] = []
    for view_index in range(len(theta_deg)):
        projected = project_points_to_image_rc(
            flat_vertices,
            view_index=view_index,
            sensor=sensor,
            source=source,
            local_x=local_x,
            local_y=local_y,
            image_dim=image_dim,
            pixel_spacing_mm=pixel_spacing_mm,
        ).reshape(-1, 3, 2)
        canvas = Image.new("L", (int(image_dim), int(image_dim)), color=0)
        draw = ImageDraw.Draw(canvas)
        for triangle in projected:
            if not np.isfinite(triangle).all():
                continue
            draw.polygon(
                [(float(point[1]), float(point[0])) for point in triangle],
                fill=255,
            )
        raw_mask = np.asarray(canvas, dtype=np.uint8) > 0
        closed = _binary_closing_disk(raw_mask, radius=2)
        images.append((_gaussian_blur(closed, sigma=0.5) > 0.25).astype(np.float32))
    return np.stack(images, axis=0), (sensor, source, local_x, local_y)
