import torch
import numpy as np
from vessel_code.preprocessing.vessel_code_transform_utils import bspline_basis_matrix
from vessel_code.parametric.representation import decode_bspline_centerlines_local

def test_clamped_bspline_endpoints_and_gradient():
    basis=torch.tensor(bspline_basis_matrix(np.linspace(0,1,200),20,3)[0],dtype=torch.float32)
    controls=torch.randn(2,7,20,3,requires_grad=True)
    dense=decode_bspline_centerlines_local(controls,basis)
    assert dense.shape==(2,7,200,3)
    torch.testing.assert_close(dense[:,:,0],controls[:,:,0])
    torch.testing.assert_close(dense[:,:,-1],controls[:,:,-1])
    dense.square().mean().backward()
    assert controls.grad is not None and torch.isfinite(controls.grad).all()
