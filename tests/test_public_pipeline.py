from pathlib import Path
import numpy as np
import pytest
from vessel_code.data.ct_volume import load_annotated_ct, render_ct_case
from vessel_code.evaluate import paper_metrics, summarize, validate_evaluation_config
from vessel_code.config import validate_model_config


def sample(tmp_path, **overrides):
    vol=np.zeros((20,20,30),np.uint8);vol[8:12,8:12,4:26]=1
    args=dict(vol=vol,spacing=np.ones(3)*0.5,coordinate_frame=np.array('LAS'),projection_center_offset_mm=np.array([5,5,7.5]))
    args.update(overrides);p=tmp_path/'sample.npz';np.savez(p,**args);return p


def test_canonical_ct_renders_seven_views(tmp_path):
    case=load_annotated_ct(sample(tmp_path),'rca');views=render_ct_case(case)
    assert views['images'].shape==(7,256,256)
    assert views['images'].sum(axis=(1,2)).min()>0
    assert set(np.unique(views['images'])) <= {0,1}
    assert views['view_features'].shape==(7,4)
    np.testing.assert_allclose(views['projection_center_offset'],[.005,.005,.0075])


def test_translation_of_world_frame_preserves_masks(tmp_path):
    p=sample(tmp_path);a=load_annotated_ct(p,'lca');one=render_ct_case(a)
    shift=np.array([70,20,-30]);affine=np.diag([.5,.5,.5,1.]);affine[:3,3]=shift
    p=sample(tmp_path,affine=affine,projection_center_offset_mm=np.array([5,5,7.5])+shift)
    two=render_ct_case(load_annotated_ct(p,'lca'))
    np.testing.assert_array_equal(one['images'],two['images'])


@pytest.mark.parametrize('override',[
    {'coordinate_frame':np.array('RAS')}, {'spacing':np.array([1.,0.,1.])},
    {'vol':np.ones((4,4,4))*100}, {'theta_deg':np.zeros(7)},
    {'affine':np.eye(4)}, {'projection_center_offset_mm':np.array([np.nan,0,0])}])
def test_bad_inputs_rejected(tmp_path,override):
    with pytest.raises(ValueError):load_annotated_ct(sample(tmp_path,**override),'rca')


def test_frame_and_center_must_be_explicit(tmp_path):
    p=tmp_path/'missing.npz';np.savez(p,vol=np.ones((5,5,5)),spacing=np.ones(3))
    with pytest.raises(ValueError,match='coordinate_frame'):load_annotated_ct(p,'rca')
    np.savez(p,vol=np.ones((5,5,5)),spacing=np.ones(3),coordinate_frame='LAS')
    with pytest.raises(ValueError,match='projection_center'):load_annotated_ct(p,'rca')


def test_only_two_evaluation_modes():
    for mode in ['visualization','paper_metric']:validate_evaluation_config({'mode':mode,'artery_type':'rca'})
    with pytest.raises(ValueError):validate_evaluation_config({'mode':'optimization','artery_type':'rca'})
    with pytest.raises(ValueError):validate_evaluation_config({'mode':'paper_metric','artery_type':'rca','view_indices':[1,1]})
    with pytest.raises(ValueError):validate_model_config({'centerline_prediction_mode':'adaptive_landmarks'})


def test_metric_centrelines_not_invented(tmp_path):
    case=load_annotated_ct(sample(tmp_path),'rca')
    vessel=np.zeros((1,10,4));vessel[0,:,2]=np.linspace(-4,4,10);vessel[...,3]=1
    with pytest.raises(ValueError,match='vessel_code_mm'):paper_metrics(case,vessel,np.array([True]),np.array([5,5,7.5]))
    result=paper_metrics(case,vessel,np.array([True]),np.array([5,5,7.5]),require_centerline=False)
    assert result['chamfer_mm'] is None
    assert 0<=result['dice_percent']<=100
    assert result['connected_components']==1


def test_summary_reports_per_metric_sample_counts():
    rows=[dict(role='refined',dice_percent=x,cldice_percent=x,chamfer_mm=None,connected_components=1) for x in [2.,4.]]
    s=summarize(rows)['refined'];assert s['dice_percent']=={'n':2,'mean':3.,'standard_error':1.}
    assert s['chamfer_mm']=={'n':0,'mean':None,'standard_error':None}
