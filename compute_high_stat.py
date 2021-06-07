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

def getTRAIN_num(path):
    idx = path.find('TRAIN')
    return int(path[idx+5])


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        '-model_path',
        type=str,
        required=True,
        help='path of the model to load'
)
parser.add_argument(
        '-iters',
        type=int,
        required=True,
        help='iterations, i.e. number of computations of sf and pdf'
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
model_path = args.model_path
train_n = getTRAIN_num(model_path)
print('train n ', train_n)
out_dir = args.out_dir

#Defines number of iterations each generating 50k trajs
print('smooth: ', smooth)
iters  = args.iters
traj_per_iter = 50000
bs = 25000
n_batch = 2

semidisp = (DB_MAX - DB_MIN)/2.
media = (DB_MAX + DB_MIN)/2.

save_path = out_dir
if not osp.exists(save_path):
    os.makedirs(save_path)

#Define Output PDF files
save_pdf = save_path+f'/pdf_train{train_n}'

#Define Output SF files
save_sf  = save_path+f'/sf_train{train_n}'
save_dl  = save_path+f'/sfdl_train{train_n}'


#Load Model
path = model_path
print('Loading Model ...')
gen = load_model(path)


#Main Loop

for i in range(iters):

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"Iteration n: {i}, {current_time}", flush=True)

    #Generate 50k Trajectories
    print('Generating Trajectories ...')
    # noise = np.random.normal(0, 1, size=(bs, NOISE_DIM)) #VAR
    velo = np.zeros(shape=(traj_per_iter,SIG_LEN,CHANNELS))
    for j in range(n_batch):
        noise = np.random.standard_t(4, size=(bs, NOISE_DIM)) #VAR
        velo[bs*j:bs*(j+1),:,COMPONENTS] = gen.predict(noise, verbose=1, batch_size=bs)
    #Renormalize in the original range
    velo = velo * semidisp + media
    if smooth:
        velo = ff.gaussian_filter1d(velo, sigma=2,
        mode='nearest', truncate=5, axis=1)

    #Remove Borders
    velo = velo[:,100:1900,:] # WE WITHOUT EXTREMES

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
    dl = np.zeros(shape=sf.shape)
    dl[:,0]  = sf[:,0]
    dl[:,1:] =  np.gradient( np.log(sf[:,1:]), np.log(sf[:,0]), axis=0 )

    #Write SF
    np.savetxt(save_sf+f'_{i}.dat', sf)
    np.savetxt(save_dl+f'_{i}.dat', dl)

