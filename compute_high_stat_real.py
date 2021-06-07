import os
import os.path as osp
from datetime import datetime

import subprocess as sp
from params import *

from db_utils import *
import stats

#from datetime import datetime
#now = datetime.now()
#dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
#print("ESECUZIONE: ", dt_string, "\n")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        '-npy_path',
        type=str,
        required=True,
        help='path of the model to load'
)
parser.add_argument(
        '-out_dir',
        type=str,
        required=True,
        help='output directory'
)
parser.add_argument(
        '--smooth',
        action='store_true',
        default=False,
        help='smooth the computed trajectories'
)

args = parser.parse_args()

smooth = args.smooth
npy_path = args.npy_path
out_dir = args.out_dir

traj_per_iter = 50000
bs = 25000
n_batch = 2

save_path = out_dir
if not osp.exists(save_path):
    os.makedirs(save_path)

#Define Output PDF files
save_pdf = save_path+f'/pdf'

#Define Output SF files
save_sf  = save_path+f'/sf'
save_dl  = save_path+f'/sfdl'

db = np.load(npy_path)
idx = np.random.permutation(db.shape[0])
db = db[idx]
n_splits = 6
cs = db.shape[0] // n_splits



#Main Loop

for i in range(n_splits):

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"Iteration n: {i}, {current_time}", flush=True)

    velo = db[cs*i:] if i == n_splits - 1 else db[cs*i:cs*(i+1)]

    # sample 50k Trajectories
    # noise = np.random.normal(0, 1, size=(bs, NOISE_DIM)) #VAR

    if smooth:
        velo = ff.gaussian_filter1d(velo, sigma=2,
        mode='nearest', truncate=5, axis=1)

    #Compute first der
    acce = np.gradient(velo,axis=1)

    print("done.")

    #Compute histo
    print('Computing Histo ...', flush=True)

    #Define Output PDF bins
    nbins=600
    hist_vex, bin_ve = np.histogram(velo[:,:,0].flatten(), nbins, (-12,12), density=False)
    hist_acx, bin_ac = np.histogram(acce[:,:,0].flatten(), nbins, (-6,6),   density=False)
    hist_vey, bin_ve = np.histogram(velo[:,:,1].flatten(), nbins, (-12,12), density=False)
    hist_acy, bin_ac = np.histogram(acce[:,:,1].flatten(), nbins, (-6,6),   density=False)
    hist_vez, bin_ve = np.histogram(velo[:,:,2].flatten(), nbins, (-12,12), density=False)
    hist_acz, bin_ac = np.histogram(acce[:,:,2].flatten(), nbins, (-6,6),   density=False)

    #Write Histo
    array_to_save = np.stack((bin_ve[:-1],hist_vex,hist_vey,hist_vez,
                          bin_ac[:-1],hist_acx,hist_acy,hist_acz)).T
    np.savetxt(save_pdf+f'_{i}.dat', array_to_save)

    #Compute SF
    print('Computing SF ...', flush=True)
    sf = stats.sf_wrap(velo, dtype=np.float32, chunk_size=5000, mode='3d').T
    np.savetxt(save_sf+f'_{i}.dat', sf)

    dl = np.zeros(shape=sf.shape)
    dl[:,0]  = sf[:,0]
    dl[:,1:] =  np.gradient( np.log(sf[:,1:]), np.log(sf[:,0]), axis=0 )
    np.savetxt(save_dl+f'_{i}.dat', dl)

