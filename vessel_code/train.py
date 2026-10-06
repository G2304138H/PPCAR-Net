"""Train the B-spline coarse model, geometry refiner, or radius refiner."""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
from pathlib import Path
from vessel_code.config import load_config, validate_model_config


def extract_centerline_checkpoint(source, destination):
    """Reuse the frozen evidence network embedded in a supplied final model."""
    import torch
    ckpt=torch.load(source,map_location='cpu',weights_only=False)
    cfg=ckpt['config']['model']; state={}
    for key,value in ckpt['model_state_dict'].items():
        for prefix,target in [('bspline_control_refiner.centerline_image_feature_encoder.','image_feature_encoder.'),
                              ('bspline_control_refiner.input_centerline_probability_head.','input_centerline_probability_head.')]:
            if key.startswith(prefix):state[target+key[len(prefix):]]=value
    if not state:raise ValueError('Checkpoint has no embedded centreline evidence network')
    torch.save({'schema_version':2,'task':'single_view_centerline_probability','model_state_dict':state,
        'config':{'model':{'centerline_map_size':cfg['bspline_refiner_centerline_map_size']}},
        'target_representation':{'mode':cfg['bspline_refiner_centerline_probability_target_mode'],
            'affinity_gamma':cfg['bspline_refiner_centerline_probability_target_gamma'],
            'mask_threshold':cfg['bspline_refiner_centerline_probability_mask_threshold']}},destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--resume',nargs='?',const='latest')
    args=p.parse_args();config=load_config(args.config);validate_model_config(config)
    stage=config.pop('stage',None)
    if stage not in ['coarse','geometry','radius']:raise ValueError('stage must be coarse, geometry, or radius')
    if stage=='radius':
        from vessel_code.parametric.train_radius_refiner import main as engine
    else:
        from vessel_code.parametric.train import main as engine
        config['train_bspline_refiner_only']=stage=='geometry'
        config['train_radius_refiner_only']=False
    argv=sys.argv
    with tempfile.TemporaryDirectory(prefix='vessel-code-training-') as temp:
        source=config.pop('centerline_initialization_checkpoint',None)
        if source and stage=='geometry':
            dest=Path(temp)/'centerline.pt';extract_centerline_checkpoint(source,dest)
            config['model']['bspline_refiner_pretrained_centerline_checkpoint']=str(dest)
        resolved=Path(temp)/'config.json';resolved.write_text(json.dumps(config))
        sys.argv=['train','--config',str(resolved)]+(['--resume',args.resume] if args.resume else [])
        try:engine()
        finally:sys.argv=argv
if __name__=='__main__':main()
