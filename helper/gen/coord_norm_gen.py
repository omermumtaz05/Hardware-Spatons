import numpy as np, struct
xs = np.linspace(0,1,256,dtype=np.float32)
with open('data_files/coord_norm.hex','w') as fo:
    for v in xs: fo.write(struct.pack('>f',float(v)).hex()+'\n')