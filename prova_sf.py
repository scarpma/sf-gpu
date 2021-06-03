import stats
import numpy as np
import torch

db = np.load('/m100_work/INF21_fldturb_0/velocities.npy',
    allow_pickle=True)

sf = stats.sf_wrap(db, dtype=np.float32, mode='3d', chunk_size=10000 )

np.savetxt('prova_sf.dat', sf.T)

