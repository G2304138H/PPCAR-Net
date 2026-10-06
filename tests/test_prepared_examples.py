from pathlib import Path
import json
import numpy as np
import pytest
from vessel_code.data.prepared_projections import PreparedProjectionDataset
from vessel_code.evaluate import validate_evaluation_config

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('anatomy', ['rca', 'lca'])
def test_bundled_example(anatomy):
    cfg=json.loads((ROOT/f'configs/{anatomy}/evaluate_example.json').read_text())
    assert validate_evaluation_config(cfg)==[0,1]
    file=ROOT/cfg['input_dir']/f'{cfg["case_ids"][0]}.npz'
    case, projections=next(iter(PreparedProjectionDataset([file],anatomy)))
    with np.load(file,allow_pickle=False) as source:
        np.testing.assert_array_equal(projections['images'],source['images'])
        np.testing.assert_allclose(projections['projection_center_offset_mm'],source['projection_center_offset']*source['input_scale_to_mm'])
        np.testing.assert_array_equal(case.vessel_code_mm,source['raw_vessel_code_mm'])
    assert projections['images'].shape==(7,256,256)
    with pytest.raises(ValueError,match='disagree'):
        next(iter(PreparedProjectionDataset([file],'lca' if anatomy=='rca' else 'rca')))
    with pytest.raises(ValueError,match='annotated CT'):
        validate_evaluation_config(dict(cfg,mode='paper_metric'))


def test_mismatched_camera_metadata_is_rejected(tmp_path):
    with np.load(ROOT/'examples/rca/rca_0457.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    data['theta_deg']=data['theta_deg'][::-1]
    path=tmp_path/'bad.npz';np.savez_compressed(path,**data)
    with pytest.raises(ValueError,match='camera angles'):
        next(iter(PreparedProjectionDataset([path],'rca')))
