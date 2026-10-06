"""JSON configuration loading and the public B-spline-only contract."""
from __future__ import annotations
import copy
import json
from pathlib import Path

def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result

def load_config(path, _seen=()):
    path = Path(path).expanduser().resolve()
    if path in _seen:
        raise ValueError('Circular configuration inheritance')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Configuration must be a JSON object')
    parent = value.pop('extends', None)
    return merge(load_config(path.parent / parent, (*_seen, path)), value) if parent else value

def validate_model_config(config):
    effective = merge(config, config.get('model', {}))
    if effective.get('centerline_prediction_mode', 'bspline_control_points') != 'bspline_control_points':
        raise ValueError('This release supports B-spline centrelines only')
    if effective.get('radius_prediction_mode', 'raw') != 'raw':
        raise ValueError('This release supports dense (raw) radii only')
    if effective.get('feature_backbone', 'vggt') != 'vggt':
        raise ValueError('This release supports the original frozen VGGT backbone only')
    if effective.get('decoder_architecture', 'absolute_parallel') != 'absolute_parallel':
        raise ValueError('This release supports the absolute_parallel branch decoder only')
    if effective.get('vggt_finetune', False):
        raise ValueError('The VGGT backbone must remain frozen')
    if effective.get('only_centerline', False):
        raise ValueError('This release includes dense radius prediction')
