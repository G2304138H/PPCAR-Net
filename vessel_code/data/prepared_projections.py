"""Read the bundled seven-view projection examples for visualization."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from vessel_code.geometry.projection import view_features_from_angles


@dataclass
class PreparedCase:
    path: Path
    vessel_code_mm: np.ndarray | None


class PreparedProjectionDataset:
    def __init__(self, paths, artery_type):
        if artery_type not in ('rca', 'lca'):
            raise ValueError('artery_type must be rca or lca')
        self.paths = list(paths)
        self.artery_type = artery_type

    def __iter__(self):
        for path in self.paths:
            path = Path(path)
            with np.load(path, allow_pickle=False) as data:
                if str(data['vessel_type'].item()).lower() != self.artery_type:
                    raise ValueError('Prepared example and requested artery_type disagree')
                def array(key, shape):
                    value = np.asarray(data[key], dtype=np.float32)
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError(f'{path}: {key} must be finite with shape {shape}')
                    return value
                images = array('images', (7, 256, 256))
                if np.any(images < 0) or np.any(images > 1):
                    raise ValueError('Prepared masks must be in [0, 1]')
                theta = array('theta_deg', (7,))
                phi = array('phi_deg', (7,))
                features = array('view_features', (7, 4))
                if not np.allclose(features, view_features_from_angles(theta, phi), atol=1e-5):
                    raise ValueError('Prepared view features disagree with camera angles')
                scale = float(data['input_scale_to_mm'].item())
                sid = float(data['sid'].item())
                spacing = float(data['imager_pixel_spacing'].item())
                if not all(np.isfinite(v) and v > 0 for v in (scale, sid, spacing)):
                    raise ValueError('Prepared units and camera parameters must be positive')
                center = array('projection_center_offset', (3,)) * scale
                vessel = None
                if 'raw_vessel_code_mm' in data:
                    branches = 7 if self.artery_type == 'rca' else 13
                    vessel = array('raw_vessel_code_mm', (branches, 200, 4))
                    if np.any(vessel[..., 3] < 0):
                        raise ValueError('Reference radii must be nonnegative')
                projections = dict(images=images, theta_deg=theta, phi_deg=phi,
                    view_features=features, projection_center_offset_mm=center,
                    image_size=np.asarray(256), sid_m=np.asarray(sid),
                    pixel_spacing_mm=np.asarray(spacing))
            yield PreparedCase(path, vessel), projections
