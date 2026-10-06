"""Create a small canonical RCA NPZ to check the loader; not a clinical sample."""
import argparse
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='data/evaluation/rca/synthetic.npz');args=p.parse_args()
    shape=(48,48,64);spacing=np.array([0.5,0.5,0.5],dtype=np.float32)
    x,y,z=np.indices(shape)*0.5
    cx=12+2*np.sin(z/10)
    mask=((x-cx)**2+(y-12)**2<1.5**2)&(z>3)&(z<28)
    t=np.linspace(3.5,27.5,200);vessel=np.zeros((7,200,4),np.float32)
    vessel[0]=np.stack([12+2*np.sin(t/10),np.full_like(t,12),t,np.full_like(t,1.5)],axis=-1)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,vol=mask.astype(np.uint8),spacing=spacing,coordinate_frame=np.asarray('LAS'),
        artery_type=np.asarray('rca'),projection_center_offset_mm=vessel[0,:,:3].mean(axis=0),vessel_code_mm=vessel)
    print(path)
if __name__=='__main__':main()
