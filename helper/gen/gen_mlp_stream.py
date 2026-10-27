#!/usr/bin/env python3
"""gen_mlp_stream.py — feature + golden for the pipelined-MLP integration test.
Reads capture_array.hex, runs Javad's A12 feature assembly + full forward + post,
emits flat per-pixel files for the streaming tb. Optional crop for speed."""

import argparse, struct, os
import numpy as np, torch, torch.nn as nn

def f32_hex(v): return struct.pack('>f', float(v)).hex()
def hex_f32(h): return struct.unpack('>f', bytes.fromhex(h))[0]

class ParametricModel(nn.Module):
    def __init__(self, i, o, h):
        super().__init__()
        self.layer1=nn.Linear(i,h); self.layer2=nn.Linear(h,h)
        self.layer3=nn.Linear(h,o); self.residual=nn.Linear(i,o)
        self.activation=nn.ReLU()
    def forward(self,x):
        t=self.activation(self.layer1(x)); t=self.activation(self.layer2(t))
        return self.layer3(t)+self.residual(x)

def reflect(c,N):
    if c<0: return -c
    if c>=N: return 2*(N-1)-c
    return c

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights', default='#86-NN2_best.pth')
    ap.add_argument('--capture', default='data_files/capture_array.hex')
    ap.add_argument('--crop', type=int, default=16,
                    help='NxN top-left crop to test (0 = full 256)')
    args=ap.parse_args()
    os.makedirs('data_files', exist_ok=True)

    FULL_W=FULL_H=256
    IN_DIM,HIDDEN,OUT_DIM=29,48,3

    model=ParametricModel(IN_DIM,OUT_DIM,HIDDEN)
    sd=torch.load(args.weights,map_location='cpu')['model_state_dict']
    for name in ['layer1','layer2','layer3','residual']:
        getattr(model,name).weight.data=sd[f'models.1.model.{name}.weight']
        getattr(model,name).bias.data  =sd[f'models.1.model.{name}.bias']
    model.eval()

    # load full capture image (feature gather needs full neighborhood)
    img=np.zeros((FULL_H,FULL_W,3),dtype=np.float32)
    with open(args.capture) as f:
        lines=[l.strip() for l in f if l.strip() and not l.strip().startswith('//')]
    for idx,line in enumerate(lines):
        line=line.zfill(24)
        B=hex_f32(line[0:8]); G=hex_f32(line[8:16]); R=hex_f32(line[16:24])
        img[idx//FULL_W, idx%FULL_W]=[R,G,B]
    img=np.clip(img,0,1)

    DOMAIN,LUT_N=8.0,4096; step=(2*DOMAIN)/(LUT_N-1)
    def lut_sig(z):
        if z<=-DOMAIN: return np.float32(0.0)
        if z>= DOMAIN: return np.float32(1.0)
        i=max(0,min(LUT_N-1,int(round((z+DOMAIN)/step))))
        return np.float32(1.0/(1.0+np.exp(-(-DOMAIN+i*step))))

    xs=np.linspace(0,1,FULL_W,dtype=np.float32)
    ys=np.linspace(0,1,FULL_H,dtype=np.float32)
    def px(x,y): return img[reflect(y,FULL_H),reflect(x,FULL_W)]

    N=args.crop if args.crop>0 else 256
    feats=[]; golden=[]
    for y in range(N):
        for x in range(N):
            f=[xs[x],ys[y]]; f.extend(img[y,x])
            for dy in(-1,0,1):
                for dx in(-1,0,1):
                    if dy==0 and dx==0: continue
                    f.extend(px(x+dx,y+dy))
            assert len(f)==29
            feats.append(f)
            ft=torch.tensor(f,dtype=torch.float32).view(1,IN_DIM)
            with torch.no_grad(): z=model(ft).view(3).numpy()
            d=np.array([lut_sig(zi) for zi in z],dtype=np.float32)
            fin=np.clip(img[y,x]+(4.0*d-2.0),0,1).astype(np.float32)
            golden.append(fin)

    # features: 29 FP32 per pixel, one value per line (flat) -> readmemh
    with open('data_files/mlp_features.hex','w') as fo:
        for f in feats:
            for v in f: fo.write(f32_hex(v)+"\n")
    # golden: 3 FP32 per pixel, one value per line
    with open('data_files/mlp_golden.hex','w') as fo:
        for g in golden:
            for v in g: fo.write(f32_hex(v)+"\n")
    print(f"wrote {len(feats)} pixels ({N}x{N}): mlp_features.hex (29/px), mlp_golden.hex (3/px)")

if __name__=='__main__': main()