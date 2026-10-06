"""B-spline reconstruction: visualization or paper metrics, from annotated CT or prepared projections."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from vessel_code.config import load_config, validate_model_config
from vessel_code.data.ct_volume import CTProjectionDataset
from vessel_code.data.prepared_projections import PreparedProjectionDataset


def validate_evaluation_config(config):
    if config.get('mode') not in ('visualization', 'paper_metric'):
        raise ValueError('mode must be visualization or paper_metric')
    if config.get('artery_type') not in ('rca', 'lca'):
        raise ValueError('artery_type must be rca or lca')
    if 'view_indices' in config:
        raise ValueError('view_indices is not supported; set num_views to select the first N views')
    input_format = config.get('input_format', 'annotated_ct')
    if input_format not in ('annotated_ct', 'prepared_projections'):
        raise ValueError('input_format must be annotated_ct or prepared_projections')
    if input_format == 'prepared_projections' and config['mode'] != 'visualization':
        raise ValueError('Prepared projections support visualization only; paper metrics require an annotated CT volume')
    num_views = config.get('num_views', 2)
    if type(num_views) is not int or not 1 <= num_views <= 7:
        raise ValueError('num_views must be an integer from 1 to 7')
    return list(range(num_views))


def prediction_roles(output, artery_type, threshold=0.5):
    required = 1 if artery_type == 'rca' else 2
    roles = {}
    for role, key in [('coarse', 'coarse_decoded_vessel_mm'), ('refined', 'decoded_vessel_mm')]:
        if key not in output:
            continue
        probs = output.get('coarse_branch_exist_probs', output['branch_exist_probs']) if role == 'coarse' else output['branch_exist_probs']
        active = probs[0].detach().cpu().numpy() >= threshold
        active[:required] = True
        roles[role] = (output[key][0].detach().cpu().numpy(), active)
    if 'coarse' not in roles:
        roles = {'coarse': roles['refined']}
    return roles


def paper_metrics(case, vessel_centered_mm, active, center_mm, *, require_centerline=True):
    from vessel_code.parametric.paper_metrics import (
        isotropic_grid_from_source, resample_binary_volume_nearest_affine,
        rasterize_vessel_to_mask, dice_similarity_3d, cldice_metrics_3d,
        connected_component_count_3d, symmetric_chamfer_distance,
    )
    if require_centerline and case.vessel_code_mm is None:
        raise ValueError(f'{case.path}: paper_metric requires vessel_code_mm for the paper Chamfer distance; set require_centerline=false for explicitly incomplete volumetric-only metrics')
    shape, affine = isotropic_grid_from_source(source_shape=case.mask.shape,
        source_index_to_world_affine=case.affine_mm, spacing_mm=0.5)
    target = resample_binary_volume_nearest_affine(case.mask,
        source_index_to_world_affine=case.affine_mm, output_shape=shape,
        output_index_to_world_affine=affine)
    absolute = np.asarray(vessel_centered_mm).copy()
    absolute[...,:3] += center_mm
    pred, _ = rasterize_vessel_to_mask(absolute, active,
        source_volume_shape=case.mask.shape, source_index_to_world_affine=case.affine_mm,
        output_shape=shape, output_index_to_world_affine=affine)
    result = {'dice_percent': 100*float(dice_similarity_3d(pred,target)),
              'cldice_percent': 100*float(cldice_metrics_3d(pred,target)['cldice_3d']),
              'connected_components': int(connected_component_count_3d(pred)),
              'chamfer_mm': None}
    if case.vessel_code_mm is not None:
        valid = case.vessel_code_mm[...,3] > 0
        result['chamfer_mm'] = symmetric_chamfer_distance(absolute[active,:,:3].reshape(-1,3), case.vessel_code_mm[...,:3][valid])
    return result


def summarize(records):
    result = {}
    for role in sorted({r['role'] for r in records}):
        rows = [r for r in records if r['role']==role]
        result[role] = {}
        for metric in ['dice_percent','cldice_percent','chamfer_mm','connected_components']:
            values = np.asarray([r[metric] for r in rows if r.get(metric) is not None], dtype=float)
            result[role][metric] = {'n':len(values), 'mean':float(values.mean()) if len(values) else None,
                'standard_error':float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None}
    return result


def visualize(path, case, projections, roles, view_indices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from vessel_code.geometry.projection import _stage2_camera_parameters, project_points_to_image_rc
    from vessel_code.shared.visualization import _surface_coords_for_valid_branches, _plot_surface_collection, _set_equal_3d_axes
    center = projections['projection_center_offset_mm']
    fig, axes = plt.subplots(1,7,figsize=(21,3),layout='constrained')
    cams = _stage2_camera_parameters(projections['theta_deg'],projections['phi_deg'],sid_m=float(projections.get('sid_m', 0.9)))
    colors={'coarse':'#24b4c4','refined':'#ffb34f'}
    for i,ax in enumerate(axes):
        ax.imshow(projections['images'][i],cmap='gray',vmin=0,vmax=1)
        for role,(vessel,active) in roles.items():
            for branch in vessel[active]:
                rc=project_points_to_image_rc(branch[:,:3]*0.001,view_index=i,
                    sensor=cams[0],source=cams[1],local_x=cams[2],local_y=cams[3],
                    image_dim=256,pixel_spacing_mm=float(projections['pixel_spacing_mm']))
                ax.plot(rc[:,1],rc[:,0],color=colors[role],lw=0.8)
        ax.set(xlim=(0,255),ylim=(255,0),title=f'View {i}'+(' • input' if i in view_indices else ''))
        ax.axis('off')
    fig.suptitle('Input projection masks with predicted centrelines: coarse (cyan), refined (orange)')
    fig.savefig(path/'projection_overlay.png',dpi=150);plt.close(fig)
    all_points = [v[active,:,:3].reshape(-1,3) for v,active in roles.values()]
    if case.vessel_code_mm is not None:
        valid = case.vessel_code_mm[...,3] > 0
        all_points.append(case.vessel_code_mm[...,:3][valid] - center)
    common_points = np.concatenate(all_points)
    padding = max(float(v[active,:,3].max()) for v,active in roles.values())
    common_points = np.stack([common_points.min(axis=0)-padding, common_points.max(axis=0)+padding])
    fig=plt.figure(figsize=(6*len(roles),6),layout='constrained')
    for j,(role,(vessel,active)) in enumerate(roles.items(),1):
        ax=fig.add_subplot(1,len(roles),j,projection='3d')
        surf=_surface_coords_for_valid_branches(vessel,np.flatnonzero(active).tolist())
        _plot_surface_collection(ax,surf,color=colors[role],alpha=0.85,zorder=2)
        if case.vessel_code_mm is not None:
            gt=case.vessel_code_mm.copy();gt[...,:3]-=center
            for b in gt[np.any(gt[...,3]>0,axis=1)]: ax.plot(*b[:,:3].T,color='#333333',lw=1,label=None)
        ax.set(title=role.capitalize(),xlabel='Left (mm)',ylabel='Anterior (mm)',zlabel='Superior (mm)')
        _set_equal_3d_axes(ax, common_points)
    if case.vessel_code_mm is not None: fig.suptitle('Predicted artery surfaces; annotated centreline in black')
    fig.savefig(path/'reconstruction.png',dpi=150);plt.close(fig)


def run(config):
    import torch
    from vessel_code.parametric.model import build_model_for_checkpoint
    from vessel_code.parametric.online_vggt import OnlineVGGTFeatureProvider
    indices = validate_evaluation_config(config)
    out = Path(config['output_dir']); out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists():
        raise FileExistsError(f'{out} already contains a run; choose a new output directory')
    checkpoint_path=Path(config['checkpoint'])
    # Research checkpoints contain config and numpy RNG state, so load trusted local files only.
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model_config=checkpoint['config'];validate_model_config(model_config)
    if model_config['artery_type'].lower()!=config['artery_type']:
        raise ValueError('Checkpoint and input artery_type disagree')
    if model_config.get('target_coordinate_frame','projection_centered')!='projection_centered':
        raise ValueError('Evaluation requires a projection-centered checkpoint')
    device=torch.device(config.get('device','auto') if config.get('device','auto')!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model=build_model_for_checkpoint(model_config,checkpoint['view_feat_dim'],checkpoint.get('inferred_feature_dim'),checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.to(device).eval()
    root=Path(config['input_dir']); files=sorted(root.glob('*.npz'))
    if config.get('case_ids') is not None:
        requested=set(map(str,config['case_ids']));available={p.stem for p in files}
        if requested-available:raise ValueError(f'Missing requested cases: {sorted(requested-available)}')
        files=[p for p in files if p.stem in requested]
    if not files:raise ValueError(f'No selected NPZ cases in {root}')
    provider_config=dict(model_config,online_vggt_cache_read=False,online_vggt_cache_write=False,online_vggt_cache_dir=None)
    provider=OnlineVGGTFeatureProvider(provider_config,device=device,raw_image_dataset_dir=root)
    records=[]
    run_record={'config':config,'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'cases':[p.name for p in files],'view_indices':indices,'metric_grid_spacing_mm':0.5,
        'feature_context':model_config.get('expected_vggt_context_mode',model_config.get('vggt_context_mode')),
        'status':'running'}
    (out/'run.json').write_text(json.dumps(run_record,indent=2)+'\n')
    dataset_type = PreparedProjectionDataset if config.get('input_format') == 'prepared_projections' else CTProjectionDataset
    for case,projections in dataset_type(files,config['artery_type']):
        if config['mode']=='paper_metric' and config.get('require_centerline',True) and case.vessel_code_mm is None:
            raise ValueError(f'{case.path}: vessel_code_mm is required for complete paper metrics')
        dest=out/case.path.stem;dest.mkdir(exist_ok=False)
        np.savez_compressed(dest/'projections.npz',**projections)
        batch={'images':torch.from_numpy(projections['images'][indices][None]).to(device),
               'view_mask':torch.ones(1,len(indices),dtype=torch.bool,device=device),
               'selected_view_indices':torch.tensor([indices],device=device), 'path':[str(case.path)]}
        with torch.inference_mode():
            features=provider.features_for_batch(batch)
            output=model(views=torch.from_numpy(projections['view_features'][indices][None]).to(device),
                         view_mask=batch['view_mask'],image_features=features,images=batch['images'])
        roles=prediction_roles(output,config['artery_type'])
        center=projections['projection_center_offset_mm']
        for role,(vessel,active) in roles.items():
            absolute=vessel.copy();absolute[...,:3]+=center
            np.savez_compressed(dest/f'{role}.npz',vessel_code_mm=absolute,
                vessel_code_centered_mm=vessel,branch_exists=active,projection_center_offset_mm=center)
            if config['mode']=='paper_metric':
                records.append(dict(case_id=case.path.stem,role=role,**paper_metrics(case,vessel,active,center,require_centerline=config.get('require_centerline',True))))
        if config['mode']=='visualization': visualize(dest,case,projections,roles,indices)
        print(f'{case.path.stem}: done',flush=True)
    if records:
        (out/'per_case.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        with (out/'per_case.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        (out/'summary.json').write_text(json.dumps(summarize(records),indent=2,allow_nan=False)+'\n')
    run_record['status']='complete';(out/'run.json').write_text(json.dumps(run_record,indent=2)+'\n')
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args();run(load_config(args.config))

if __name__=='__main__':main()
