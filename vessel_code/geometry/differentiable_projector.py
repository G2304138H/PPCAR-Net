# Transferred from methods/model/differentiable_projector.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
import math
import time
from typing import NamedTuple, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionCameras(NamedTuple):
    """Precomputed camera geometry for one or more projection views."""

    v_sensor: torch.Tensor
    v_source: torch.Tensor
    local_x: torch.Tensor
    local_y: torch.Tensor

class PreparedSurfaceGeometry(NamedTuple):
    """Radius-independent moving frames for one vessel tree."""

    centerlines: torch.Tensor
    normals: torch.Tensor
    convecs: torch.Tensor

class DifferentiableVesselProjector(nn.Module):
    """
    Differentiable projection from vessel code [M, N, 4] to a soft 2D mask.

    Notes:
    - Uses smooth kernel splatting instead of hard pixel indexing.
    - Geometry is approximate but fully differentiable w.r.t vessel code.
    - Intended for training losses and qualitative checks, not exact replication
      of the non-differentiable skimage/NumPy renderer.
    """

    def __init__(
        self,
        image_size: int = 256,
        sid: float = 0.9,
        source_to_iso: float = 0.75,
        imager_pixel_spacing: float = 0.35,
        num_circle_points: int = 48,
        radial_subsamples: int = 1,
        axial_subsamples: int = 4,
        center_main_branch: bool = True,
        crop_mode: str = "strict",
        splat_sigma_px: float = 1.25,
        splat_radius: int = 2,
        blur_sigma_px: float = 1.0,
        blur_kernel_size: int = 7,
        intensity_scale: float = 0.1,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.sid = float(sid)
        self.source_to_iso = float(source_to_iso)
        self.imager_pixel_spacing = float(imager_pixel_spacing)
        self.num_circle_points = int(num_circle_points)
        self.radial_subsamples = int(max(1, radial_subsamples))
        self.axial_subsamples = int(axial_subsamples)
        self.center_main_branch = bool(center_main_branch)
        self.crop_mode = str(crop_mode)
        if self.crop_mode not in {"strict", "relaxed", "none"}:
            raise ValueError("crop_mode must be one of: strict, relaxed, none")
        self.splat_sigma_px = float(splat_sigma_px)
        self.splat_radius = int(splat_radius)
        self.intensity_scale = float(intensity_scale)

        # Match vessel_code.geometry.tube_functions.get_vessel_surface, which uses np.linspace(0, 2*pi, num_circle_points)
        # and therefore includes the endpoint duplicate.
        angles = torch.linspace(0.0, 2.0 * math.pi, steps=self.num_circle_points)
        self.register_buffer("circle_cos", torch.cos(angles))
        self.register_buffer("circle_sin", torch.sin(angles))

        offsets = torch.cartesian_prod(
            torch.arange(-self.splat_radius, self.splat_radius + 1),
            torch.arange(-self.splat_radius, self.splat_radius + 1),
        )
        # cartesian_prod returns (dy, dx); store the raster convention explicitly.
        self.register_buffer(
            "splat_offsets_xy", offsets[:, [1, 0]], persistent=False
        )

        blur_kernel = self._build_gaussian_kernel(blur_kernel_size, blur_sigma_px)
        self.register_buffer("blur_kernel", blur_kernel)

    @staticmethod
    def _build_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        half = (kernel_size - 1) / 2.0
        x = torch.arange(kernel_size, dtype=torch.float32) - half
        xx, yy = torch.meshgrid(x, x, indexing="ij")
        k = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        k = k / torch.clamp(k.sum(), min=1e-8)
        return k.view(1, 1, kernel_size, kernel_size)

    @staticmethod
    def _estimate_derivatives(centerline: torch.Tensor) -> torch.Tensor:
        d = torch.zeros_like(centerline)
        n = centerline.shape[0]
        if n < 2:
            return d
        d[0] = centerline[1] - centerline[0]
        d[-1] = centerline[-1] - centerline[-2]
        if n > 2:
            d[1:-1] = 0.5 * (centerline[2:] - centerline[:-2])
        return d

    @staticmethod
    def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)

    def _tubeplot_frame(self, centerline: torch.Tensor, derivatives: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Match the moving frame convention used by vessel_code.geometry.tube_functions.get_vessel_surface."""
        n = int(centerline.shape[0])
        device = centerline.device
        dtype = centerline.dtype
        if n == 0:
            empty = centerline.new_zeros((0, 3))
            return empty, empty

        ref_index = 0
        if n > 1:
            ref_index = int(torch.argmin(torch.abs(centerline[1])).detach().item())
        normal = torch.zeros((3,), device=device, dtype=dtype)
        normal[ref_index] = 1.0

        normals = []
        convecs = []
        for ki in range(n):
            d = derivatives[ki]
            convec = self._safe_normalize(torch.cross(normal, d, dim=0))
            normal = self._safe_normalize(torch.cross(d, convec, dim=0))
            normals.append(normal)
            convecs.append(convec)
        return torch.stack(normals, dim=0), torch.stack(convecs, dim=0)

    def _non_diff_centering_surface_points(
        self,
        centerline: torch.Tensor,
        radius: torch.Tensor,
        normals: torch.Tensor,
        convecs: torch.Tensor,
    ) -> torch.Tensor:
        """Detached main-branch samples used to reproduce legacy centering.

        The centering offset is a preprocessing transform, not part of the
        learned projection geometry. Build it outside autograd so subtracting
        the resulting offset preserves gradients through the rendered surface
        without differentiating through the offset calculation itself.
        """
        n = int(centerline.shape[0])
        if n == 0:
            return centerline.new_zeros((0, 3))

        with torch.no_grad():
            centerline = centerline.detach()
            radius = radius.detach()
            normals = normals.detach()
            convecs = convecs.detach()
            rings = []
            for ki in range(n):
                if ki == 0:
                    rho = torch.linspace(
                        1.0 / 50.0,
                        1.0,
                        steps=49,
                        device=centerline.device,
                        dtype=centerline.dtype,
                    )
                elif ki == n - 1:
                    rho = torch.flip(
                        torch.linspace(
                            1.0 / 50.0,
                            1.0,
                            steps=49,
                            device=centerline.device,
                            dtype=centerline.dtype,
                        ),
                        dims=[0],
                    )
                else:
                    rho = torch.ones(
                        (1,), device=centerline.device, dtype=centerline.dtype
                    )

                ring = (
                    centerline[ki][None, None, :]
                    + rho[:, None, None]
                    * radius[ki]
                    * (
                        self.circle_cos[None, :, None]
                        * normals[ki][None, None, :]
                        + self.circle_sin[None, :, None]
                        * convecs[ki][None, None, :]
                    )
                )
                rings.append(ring.reshape(-1, 3))

            return torch.cat(rings, dim=0).detach()

    def prepare_surface_geometry(
        self, vessel_code: torch.Tensor
    ) -> PreparedSurfaceGeometry:
        """Build the radius-independent surface frame once.

        This is particularly useful for an iterative radius refiner: its XYZ
        geometry is fixed while only the radius channel changes between stages.
        """

        if vessel_code.ndim != 3 or vessel_code.shape[-1] != 4:
            raise ValueError("vessel_code must have shape [M,N,4]")
        centerlines = vessel_code[..., :3]
        if int(centerlines.shape[0]) < 1:
            raise ValueError("vessel_code must contain at least one branch")
        normals = []
        convecs = []
        for branch_index in range(int(centerlines.shape[0])):
            centerline = centerlines[branch_index]
            derivatives = self._estimate_derivatives(centerline)
            branch_normals, branch_convecs = self._tubeplot_frame(
                centerline, derivatives
            )
            normals.append(branch_normals)
            convecs.append(branch_convecs)
        return PreparedSurfaceGeometry(
            centerlines=centerlines,
            normals=torch.stack(normals, dim=0),
            convecs=torch.stack(convecs, dim=0),
        )

    def _surface_points_from_prepared_geometry(
        self,
        geometry: PreparedSurfaceGeometry,
        radius: torch.Tensor,
    ) -> torch.Tensor:
        """Generate surface samples from cached moving frames and new radii."""

        centerlines = geometry.centerlines
        if radius.shape != centerlines.shape[:2]:
            raise ValueError(
                "radius must match prepared geometry [M,N], got "
                f"{tuple(radius.shape)} and {tuple(centerlines.shape[:2])}"
            )
        device = centerlines.device
        dtype = centerlines.dtype
        m = int(centerlines.shape[0])

        points_per_branch = []
        main_branch_points = None

        for bi in range(m):
            c = centerlines[bi]
            r = torch.clamp(radius[bi], min=1e-5)
            normals = geometry.normals[bi]
            convecs = geometry.convecs[bi]

            ring = (
                c[:, None, :]
                + r[:, None, None]
                * (
                    self.circle_cos[None, :, None] * normals[:, None, :]
                    + self.circle_sin[None, :, None] * convecs[:, None, :]
                )
            )
            base_ring_points = ring.reshape(-1, 3)

            branch_points_list = []
            if self.radial_subsamples > 1:
                rho = torch.linspace(
                    1.0 / float(self.radial_subsamples),
                    1.0,
                    steps=self.radial_subsamples,
                    device=device,
                    dtype=dtype,
                )
                disk = (
                    c[:, None, None, :]
                    + rho[None, :, None, None]
                    * r[:, None, None, None]
                    * (
                        self.circle_cos[None, None, :, None] * normals[:, None, None, :]
                        + self.circle_sin[None, None, :, None] * convecs[:, None, None, :]
                    )
                )
                branch_points_list.append(disk.reshape(-1, 3))
            else:
                branch_points_list.append(base_ring_points)
            # Densify along centerline direction to avoid ring-slice artifacts.
            if ring.shape[0] > 1 and self.axial_subsamples > 0:
                alpha = torch.linspace(0.0, 1.0, steps=self.axial_subsamples + 2, device=device, dtype=dtype)[1:-1]
                interp = (
                    (1.0 - alpha[None, :, None, None]) * ring[:-1, None, :, :]
                    + alpha[None, :, None, None] * ring[1:, None, :, :]
                )
                branch_points_list.append(interp.reshape(-1, 3))

            branch_points = torch.cat(branch_points_list, dim=0)
            points_per_branch.append(branch_points)

            if bi == 0 and self.center_main_branch:
                main_branch_points = self._non_diff_centering_surface_points(c, r, normals, convecs)

        pts = torch.cat(points_per_branch, dim=0)
        if self.center_main_branch and main_branch_points is not None:
            pts = pts - torch.mean(main_branch_points, dim=0, keepdim=True)
        return pts

    def _surface_points_from_vessel(self, vessel_code: torch.Tensor) -> torch.Tensor:
        # vessel_code: [M, N, 4]
        geometry = self.prepare_surface_geometry(vessel_code)
        return self._surface_points_from_prepared_geometry(
            geometry, vessel_code[..., 3]
        )

    def _camera_basis_batched(
        self, theta_deg: torch.Tensor, phi_deg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized camera bases matching the legacy scalar construction."""

        if theta_deg.shape != phi_deg.shape:
            raise ValueError("theta_deg and phi_deg must have matching shapes")
        theta = torch.deg2rad(theta_deg)
        phi = torch.deg2rad(phi_deg)
        zero = torch.zeros_like(theta)
        one = torch.ones_like(theta)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        r_ap1_native = torch.stack(
            (
                torch.stack((cos_theta, -sin_theta, zero), dim=-1),
                torch.stack((sin_theta, cos_theta, zero), dim=-1),
                torch.stack((zero, zero, one), dim=-1),
            ),
            dim=-2,
        )
        r_ap2_native = torch.stack(
            (
                torch.stack((one, zero, zero), dim=-1),
                torch.stack((zero, cos_phi, sin_phi), dim=-1),
                torch.stack((zero, -sin_phi, cos_phi), dim=-1),
            ),
            dim=-2,
        )
        coord = theta.new_tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        )
        # coord is orthogonal, so transpose is its exact inverse and avoids a
        # small matrix inversion on every projector call.
        inv_coord = coord.transpose(-1, -2)
        r_total = coord @ r_ap1_native @ r_ap2_native @ inv_coord

        detector_to_iso = self.sid - self.source_to_iso
        axis_z = theta.new_tensor([0.0, 0.0, 1.0])
        axis_y = theta.new_tensor([0.0, 1.0, 0.0])
        axis_neg_x = theta.new_tensor([-1.0, 0.0, 0.0])

        v_sensor = r_total @ (axis_z * detector_to_iso)
        v_source = (
            -v_sensor
            / max(float(detector_to_iso), 1e-8)
            * self.source_to_iso
        )

        local_x = self._safe_normalize(r_total @ axis_y)
        local_y = self._safe_normalize(r_total @ axis_neg_x)
        return v_sensor, v_source, local_x, local_y

    def _camera_basis(self, theta_deg: torch.Tensor, phi_deg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Match vessel_code.geometry.fwd_projection_functions.get_local_params(coord_system_change=True).
        return self._camera_basis_batched(theta_deg, phi_deg)

    def prepare_cameras(
        self,
        theta_deg: torch.Tensor,
        phi_deg: torch.Tensor,
    ) -> ProjectionCameras:
        """Precompute invariant camera geometry for a vector of views."""

        theta = theta_deg.reshape(-1)
        phi = phi_deg.reshape(-1)
        if theta.numel() != phi.numel():
            raise ValueError("theta_deg and phi_deg must contain the same views")
        if theta.numel() == 0:
            raise ValueError("At least one projection camera is required")
        return ProjectionCameras(*self._camera_basis_batched(theta, phi))

    def _project_points_batched(
        self,
        points: torch.Tensor,
        cameras: ProjectionCameras,
        *,
        apply_crop: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project shared or per-view points without data-dependent filtering.

        Args:
            points: ``[P,3]`` shared by every view or ``[V,P,3]``.
            cameras: precomputed geometry for ``V`` views.
        Returns:
            detector coordinates ``[V,P,2]`` and a valid/crop mask ``[V,P]``.
        """

        view_count = int(cameras.v_sensor.shape[0])
        if points.ndim == 2:
            points_by_view = points.unsqueeze(0).expand(view_count, -1, -1)
        elif points.ndim == 3 and int(points.shape[0]) == view_count:
            points_by_view = points
        else:
            raise ValueError(
                "points must have shape [P,3] or [V,P,3] matching the cameras"
            )

        v_sensor = cameras.v_sensor[:, None, :]
        v_source = cameras.v_source[:, None, :]
        local_x = cameras.local_x
        local_y = cameras.local_y
        normal = torch.cross(local_x, local_y, dim=-1)

        ray = v_source - points_by_view
        denominator = torch.sum(ray * normal[:, None, :], dim=-1)
        valid = torch.abs(denominator) > 1e-8
        denominator_safe = torch.where(
            valid, denominator, torch.ones_like(denominator)
        )
        relative_to_sensor = points_by_view - v_sensor
        scale = -torch.sum(
            relative_to_sensor * normal[:, None, :], dim=-1
        ) / denominator_safe
        projected = points_by_view + scale.unsqueeze(-1) * ray

        image_dimension = float(self.image_size)
        sensor_width = self.imager_pixel_spacing * image_dimension / 1000.0
        origin_fraction = (1.0 - image_dimension / 2.0) / image_dimension
        maximum_fraction = (
            image_dimension - image_dimension / 2.0
        ) / image_dimension
        x_axis = local_x[:, None, :]
        y_axis = local_y[:, None, :]
        local_origin = v_sensor + sensor_width * origin_fraction * (x_axis + y_axis)

        local_vector = projected - local_origin
        x_scale = torch.sum(local_vector * x_axis, dim=-1) / torch.clamp(
            torch.sum(x_axis * x_axis, dim=-1), min=1e-8
        )
        y_scale = torch.sum(local_vector * y_axis, dim=-1) / torch.clamp(
            torch.sum(y_axis * y_axis, dim=-1), min=1e-8
        )
        detector_axis_span = sensor_width * (
            maximum_fraction - origin_fraction
        )
        if abs(detector_axis_span) <= 1e-8:
            x_px = torch.zeros_like(x_scale)
            y_px = torch.zeros_like(y_scale)
        else:
            x_px = image_dimension * x_scale / detector_axis_span
            y_px = image_dimension * y_scale / detector_axis_span
        coordinates = torch.stack((x_px, y_px), dim=-1)
        valid = valid & torch.isfinite(coordinates).all(dim=-1)
        if apply_crop and self.crop_mode == "strict":
            valid = (
                valid
                & (x_px > 0.0)
                & (x_px < image_dimension)
                & (y_px > 0.0)
                & (y_px < image_dimension)
            )
        elif apply_crop and self.crop_mode == "relaxed":
            valid = (
                valid
                & (x_px >= -self.splat_radius)
                & (x_px <= image_dimension - 1.0 + self.splat_radius)
                & (y_px >= -self.splat_radius)
                & (y_px <= image_dimension - 1.0 + self.splat_radius)
            )
        return coordinates, valid

    def _project_points(self, points: torch.Tensor, theta_deg: torch.Tensor, phi_deg: torch.Tensor) -> torch.Tensor:
        # points: [P,3] -> continuous detector coordinates [x_px, y_px].
        # Detector y increases along local_y; rasterization converts it to an
        # image row exactly once with row = image_size - y_px.
        cameras = self.prepare_cameras(theta_deg.reshape(1), phi_deg.reshape(1))
        coordinates, valid = self._project_points_batched(points, cameras)
        return coordinates[0, valid[0]]

    def _splat_points_batched(
        self,
        points_px: torch.Tensor,
        valid_points: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rasterize ``[B,P,2]`` points using one vectorized scatter."""

        if points_px.ndim != 3 or points_px.shape[-1] != 2:
            raise ValueError("points_px must have shape [B,P,2]")
        batch_size, point_count, _ = points_px.shape
        h = self.image_size
        w = self.image_size
        device = points_px.device
        if valid_points is None:
            valid_points = torch.ones(
                (batch_size, point_count), dtype=torch.bool, device=device
            )
        if valid_points.shape != points_px.shape[:2]:
            raise ValueError("valid_points must have shape [B,P]")

        img_flat = torch.zeros(
            (batch_size * h * w,), device=device, dtype=points_px.dtype
        )
        if point_count == 0:
            return img_flat.view(batch_size, 1, h, w)

        safe_points = torch.where(
            valid_points.unsqueeze(-1), points_px, torch.zeros_like(points_px)
        )
        x = safe_points[..., 0]
        y = safe_points[..., 1]
        x0 = torch.round(x)
        y0 = torch.round(y)
        offsets = self.splat_offsets_xy.to(device=device)
        px = x0.unsqueeze(-1) + offsets[:, 0]
        py = y0.unsqueeze(-1) + offsets[:, 1]
        inside = (
            valid_points.unsqueeze(-1)
            & (px >= 0)
            & (px < w)
            & (py >= 0)
            & (py < h)
        )
        px_safe = px.clamp(0, w - 1)
        py_safe = py.clamp(0, h - 1)
        distance_squared = (x.unsqueeze(-1) - px_safe) ** 2 + (
            y.unsqueeze(-1) - py_safe
        ) ** 2
        weights = torch.exp(
            -distance_squared
            / (2.0 * self.splat_sigma_px * self.splat_sigma_px)
        ) * inside.to(dtype=points_px.dtype)
        linear_indices = py_safe.long() * w + px_safe.long()
        batch_offsets = (
            torch.arange(batch_size, device=device) * (h * w)
        ).view(batch_size, 1, 1)
        linear_indices = linear_indices + batch_offsets
        img_flat.index_add_(
            0, linear_indices.reshape(-1), weights.reshape(-1)
        )
        img = img_flat.view(batch_size, 1, h, w)
        img = 1.0 - torch.exp(-self.intensity_scale * img)
        pad = self.blur_kernel.shape[-1] // 2
        img = F.conv2d(
            img,
            self.blur_kernel.to(device=device, dtype=img.dtype),
            padding=pad,
        )
        return torch.clamp(img, 0.0, 1.0)

    def _splat_points(self, points_px: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        # points_px uses the stored Stage-2 image's (column, row) convention.
        # The legacy path flips detector y inside convert3D_to_pixels and flips
        # it back during raster indexing, so its stored row is the unflipped
        # coordinate returned by _project_points. Do not flip y again here.
        if batch_size != 1:
            raise ValueError(
                "Use _splat_points_batched for more than one point batch"
            )
        return self._splat_points_batched(points_px.unsqueeze(0))

    def _render_shared_surface(
        self,
        vessel_code: torch.Tensor,
        cameras: ProjectionCameras,
    ) -> torch.Tensor:
        surface_points = self._surface_points_from_vessel(vessel_code)
        projected_points, valid = self._project_points_batched(
            surface_points, cameras
        )
        return self._splat_points_batched(projected_points, valid)

    def render_shared_surface(
        self,
        vessel_code: torch.Tensor,
        cameras: ProjectionCameras,
    ) -> torch.Tensor:
        """Render one vessel surface through every precomputed camera."""

        if vessel_code.ndim != 3:
            raise ValueError("vessel_code must have shape [M,N,4]")
        return self._render_shared_surface(vessel_code, cameras)

    def render_multiview_batch(
        self,
        vessel_code: torch.Tensor,
        theta_deg: torch.Tensor,
        phi_deg: torch.Tensor,
        *,
        cameras: ProjectionCameras | None = None,
        prepared_geometries: tuple[PreparedSurfaceGeometry, ...] | None = None,
        branch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render every vessel in a batch through every corresponding view.

        Unlike repeated calls to :meth:`forward`, this generates each 3D
        surface once, then broadcasts those points across its cameras.

        Args:
            vessel_code: ``[B,M,N,4]``.
            theta_deg, phi_deg: ``[B,V]``.
            cameras: optional precomputed flattened ``B*V`` camera geometry.
            prepared_geometries: optional radius-independent geometry, one per
                batch item, for iterative radius-only rendering.
            branch_mask: optional boolean ``[B,M]`` mask. Surface samples from
                inactive branch slots are excluded from rasterization.
        Returns:
            soft masks with shape ``[B,V,1,H,W]``.
        """

        if vessel_code.ndim != 4 or vessel_code.shape[-1] != 4:
            raise ValueError("vessel_code must have shape [B,M,N,4]")
        batch_size = int(vessel_code.shape[0])
        num_branches = int(vessel_code.shape[1])
        if branch_mask is not None:
            branch_mask = branch_mask.to(
                device=vessel_code.device,
                dtype=torch.bool,
            )
            if branch_mask.shape != (batch_size, num_branches):
                raise ValueError(
                    "branch_mask must have shape [B,M] matching vessel_code, "
                    f"got {tuple(branch_mask.shape)} and "
                    f"{(batch_size, num_branches)}."
                )
        theta = theta_deg.to(
            device=vessel_code.device, dtype=vessel_code.dtype
        )
        phi = phi_deg.to(device=vessel_code.device, dtype=vessel_code.dtype)
        if theta.ndim == 1 and batch_size == 1:
            theta = theta.unsqueeze(0)
        if phi.ndim == 1 and batch_size == 1:
            phi = phi.unsqueeze(0)
        if theta.ndim != 2 or phi.shape != theta.shape:
            raise ValueError("theta_deg and phi_deg must have shape [B,V]")
        if int(theta.shape[0]) != batch_size:
            raise ValueError("camera batch dimension must match vessel_code")
        num_views = int(theta.shape[1])
        if num_views < 1:
            raise ValueError("At least one projection view is required")

        if cameras is None:
            cameras = self.prepare_cameras(theta.reshape(-1), phi.reshape(-1))
        elif int(cameras.v_sensor.shape[0]) != batch_size * num_views:
            raise ValueError("cameras must contain B*V flattened views")

        if prepared_geometries is None:
            prepared_geometries = tuple(
                self.prepare_surface_geometry(vessel_code[index])
                for index in range(batch_size)
            )
        if len(prepared_geometries) != batch_size:
            raise ValueError("prepared_geometries must contain one item per batch")
        surfaces = torch.stack(
            [
                self._surface_points_from_prepared_geometry(
                    prepared_geometries[index], vessel_code[index, ..., 3]
                )
                for index in range(batch_size)
            ],
            dim=0,
        )
        point_count = int(surfaces.shape[1])
        surfaces_by_view = (
            surfaces[:, None]
            .expand(batch_size, num_views, point_count, 3)
            .reshape(batch_size * num_views, point_count, 3)
        )
        projected_points, valid = self._project_points_batched(
            surfaces_by_view, cameras
        )
        if branch_mask is not None:
            if point_count % num_branches != 0:
                raise RuntimeError(
                    "Rendered surface points cannot be mapped evenly back to "
                    f"{num_branches} branches."
                )
            points_per_branch = point_count // num_branches
            active_surface_points = branch_mask[:, :, None].expand(
                batch_size,
                num_branches,
                points_per_branch,
            ).reshape(batch_size, point_count)
            active_surface_points = active_surface_points[:, None].expand(
                batch_size,
                num_views,
                point_count,
            ).reshape(batch_size * num_views, point_count)
            valid = valid & active_surface_points
        rendered = self._splat_points_batched(projected_points, valid)
        return rendered.reshape(
            batch_size,
            num_views,
            1,
            self.image_size,
            self.image_size,
        )

    def forward(
        self,
        vessel_code: torch.Tensor,
        theta_deg: torch.Tensor,
        phi_deg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vessel_code: [B,M,N,4] or [M,N,4]
            theta_deg: [B] or scalar
            phi_deg: [B] or scalar
        Returns:
            soft_mask: [B,1,H,W] in [0,1]
        """
        shared_geometry = vessel_code.ndim == 3
        if vessel_code.ndim == 4:
            shared_geometry = (
                int(vessel_code.shape[0]) == 1 or vessel_code.stride(0) == 0
            )
        if vessel_code.ndim not in {3, 4}:
            raise ValueError("vessel_code must have shape [M,N,4] or [B,M,N,4]")

        device = vessel_code.device
        dtype = vessel_code.dtype
        theta_deg = theta_deg.reshape(-1).to(device=device, dtype=dtype)
        phi_deg = phi_deg.reshape(-1).to(device=device, dtype=dtype)
        if shared_geometry:
            cameras = self.prepare_cameras(theta_deg, phi_deg)
            shared_vessel = vessel_code if vessel_code.ndim == 3 else vessel_code[0]
            return self._render_shared_surface(shared_vessel, cameras)

        batch_size = int(vessel_code.shape[0])
        if theta_deg.numel() == 1:
            theta_deg = theta_deg.repeat(batch_size)
        if phi_deg.numel() == 1:
            phi_deg = phi_deg.repeat(batch_size)
        if theta_deg.numel() != batch_size or phi_deg.numel() != batch_size:
            raise ValueError("Each vessel batch item requires one projection view")
        cameras = self.prepare_cameras(theta_deg, phi_deg)
        surfaces = torch.stack(
            [
                self._surface_points_from_vessel(vessel_code[index])
                for index in range(batch_size)
            ],
            dim=0,
        )
        projected_points, valid = self._project_points_batched(surfaces, cameras)
        return self._splat_points_batched(projected_points, valid)

    @staticmethod
    def _synchronize_for_timing(device: torch.device) -> None:
        """Wait for queued device work before reading a wall-clock timer."""

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def forward_with_component_timing(
        self,
        vessel_code: torch.Tensor,
        theta_deg: torch.Tensor,
        phi_deg: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Render masks while timing mutually exclusive renderer components.

        This deliberately synchronizes around each component so CUDA/MPS wall
        times are meaningful. It is intended for evaluation diagnostics, not
        for the normal training forward path.
        """

        shared_geometry = vessel_code.ndim == 3
        if vessel_code.ndim == 4:
            shared_geometry = (
                int(vessel_code.shape[0]) == 1 or vessel_code.stride(0) == 0
            )
        if vessel_code.ndim not in {3, 4}:
            raise ValueError("vessel_code must have shape [M,N,4] or [B,M,N,4]")
        device = vessel_code.device
        dtype = vessel_code.dtype
        theta_deg = theta_deg.reshape(-1).to(device=device, dtype=dtype)
        phi_deg = phi_deg.reshape(-1).to(device=device, dtype=dtype)
        if not shared_geometry:
            batch_size = int(vessel_code.shape[0])
            if theta_deg.numel() == 1:
                theta_deg = theta_deg.repeat(batch_size)
            if phi_deg.numel() == 1:
                phi_deg = phi_deg.repeat(batch_size)
            if theta_deg.numel() != batch_size or phi_deg.numel() != batch_size:
                raise ValueError("Each vessel batch item requires one projection view")
        cameras = self.prepare_cameras(theta_deg, phi_deg)

        timings_ms = {
            "surface_generation_ms": 0.0,
            "surface_projection_ms": 0.0,
            "rasterization_ms": 0.0,
        }
        self._synchronize_for_timing(device)
        started = time.perf_counter()
        if shared_geometry:
            shared_vessel = vessel_code if vessel_code.ndim == 3 else vessel_code[0]
            surface_points: torch.Tensor = self._surface_points_from_vessel(
                shared_vessel
            )
        else:
            surface_points = torch.stack(
                [
                    self._surface_points_from_vessel(vessel_code[index])
                    for index in range(int(vessel_code.shape[0]))
                ],
                dim=0,
            )
        self._synchronize_for_timing(device)
        timings_ms["surface_generation_ms"] = (
            time.perf_counter() - started
        ) * 1000.0

        started = time.perf_counter()
        projected_points, valid = self._project_points_batched(
            surface_points, cameras
        )
        self._synchronize_for_timing(device)
        timings_ms["surface_projection_ms"] = (
            time.perf_counter() - started
        ) * 1000.0

        started = time.perf_counter()
        rendered = self._splat_points_batched(projected_points, valid)
        self._synchronize_for_timing(device)
        timings_ms["rasterization_ms"] = (
            time.perf_counter() - started
        ) * 1000.0
        return rendered, timings_ms

    def render_shared_surface_with_component_timing(
        self,
        vessel_code: torch.Tensor,
        cameras: ProjectionCameras,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Timed shared-surface rendering with already prepared cameras."""

        if vessel_code.ndim != 3:
            raise ValueError("vessel_code must have shape [M,N,4]")
        device = vessel_code.device
        timings_ms = {
            "surface_generation_ms": 0.0,
            "surface_projection_ms": 0.0,
            "rasterization_ms": 0.0,
        }
        self._synchronize_for_timing(device)
        started = time.perf_counter()
        surface_points = self._surface_points_from_vessel(vessel_code)
        self._synchronize_for_timing(device)
        timings_ms["surface_generation_ms"] = (
            time.perf_counter() - started
        ) * 1000.0

        started = time.perf_counter()
        projected_points, valid = self._project_points_batched(
            surface_points, cameras
        )
        self._synchronize_for_timing(device)
        timings_ms["surface_projection_ms"] = (
            time.perf_counter() - started
        ) * 1000.0

        started = time.perf_counter()
        rendered = self._splat_points_batched(projected_points, valid)
        self._synchronize_for_timing(device)
        timings_ms["rasterization_ms"] = (
            time.perf_counter() - started
        ) * 1000.0
        return rendered, timings_ms
