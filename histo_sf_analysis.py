#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from params import *
import glob
import os.path as osp


dict3d = {
    'x2': 1,'y2': 2,'z2': 3,'xy': 4,'yz': 5,
    'xz': 6,'x4': 7,'y4': 8,'z4': 9,'x2y2': 10,
    'y2z2': 11,'x2z2': 12,'x6': 13,'x6': 14,
    'x6': 15,'x8': 16,'x8': 17,'x8': 18,
    'x10': 19,'x10': 20,'x10': 21}
dict1d = {
    'x2': 1,'x4': 2,'x6': 3,'x8': 4,
    'x10': 5,'x12': 6}



def compute_sf_mean(paths):

    assert len(paths) >= 1

    for ii, fl in enumerate(paths):
        print('read {}'.format(paths[ii]), flush=True)
        if ii == 0:
            sfg = np.loadtxt(fl)
        else:
            temp = np.loadtxt(fl)
            assert all(temp[:,0] == sfg[:,0])
            sfg[:,1:] = sfg[:,1:] + temp[:,1:]

    sfg[:,1:] = sfg[:,1:] / len(paths)

    return sfg


    ## # COMPUTE FLATNESS
    ## 
    ## ftg       = np.zeros(shape=(sfg.shape[0], sfg.shape[1]-1))
    ## ftg[:,0]  = sfg[:,0]
    ## for ii in range(1, sfg.shape[1]-1):
    ##     ftg[:,ii] = sfg[:,ii+1] / (sfg[:,1])**(ii+1)
    ## 
    ## ftr       = np.zeros(shape=(sfr.shape[0], sfr.shape[1]-1))
    ## ftr[:,0]  = sfr[:,0]
    ## for ii in range(1, sfg.shape[1]-1):
    ##     ftr[:,ii] = sfr[:,ii+1] / (sfr[:,1])**(ii+1)
    ## 
    ## 
    ## 
    ## # COMPUTE LOG DERIVATIVES ESS
    ## 
    ## dlg_ess       = np.zeros(shape=(dlg.shape[0], dlg.shape[1]-1))
    ## dlg_ess[:,0]  = dlg[:,0]
    ## for ii in range(1, sfg.shape[1]-1):
    ##     dlg_ess[:,ii] = dlg[:,ii+1] / dlg[:,1]
    ## 
    ## dlr_ess       = np.zeros(shape=(dlr.shape[0], dlr.shape[1]-1))
    ## dlr_ess[:,0]  = dlr[:,0]
    ## for ii in range(1, sfg.shape[1]-1):
    ##     dlr_ess[:,ii] = dlr[:,ii+1] / dlr[:,1]



def compute_pdf(paths):

    assert len(paths) >= 1

    for ii, fl in enumerate(paths):
        print('read {}'.format(paths[ii]), flush=True)
        temp = np.loadtxt(fl)

        if ii == 0:
            vgx = temp[:,np.array([0,1])]
            vgy = temp[:,np.array([0,2])]
            vgz = temp[:,np.array([0,3])]
            agx = temp[:,np.array([4,5])]
            agy = temp[:,np.array([4,6])]
            agz = temp[:,np.array([4,7])]
        else:
            assert all(temp[:,0] == vgx[:,0]), 'different bin for velo'
            assert all(temp[:,4] == agx[:,0]), 'different bin for acce'
            assert all(temp[:,0] == vgy[:,0]), 'different bin for velo'
            assert all(temp[:,4] == agy[:,0]), 'different bin for acce'
            assert all(temp[:,0] == vgz[:,0]), 'different bin for velo'
            assert all(temp[:,4] == agz[:,0]), 'different bin for acce'
            vgx[:,1] = vgx[:,1] + temp[:,1]
            agx[:,1] = agx[:,1] + temp[:,5]
            vgy[:,1] = vgy[:,1] + temp[:,2]
            agy[:,1] = agy[:,1] + temp[:,6]
            vgz[:,1] = vgz[:,1] + temp[:,3]
            agz[:,1] = agz[:,1] + temp[:,7]


    # SUM HISTOS AND COMPUTE STANDARDIZED PDF

    db_vex = vgx[1,0] - vgx[0,0]
    db_acx = agx[1,0] - agx[0,0]
    v_normx = np.sum(vgx[:,1]) * db_vex
    a_normx = np.sum(agx[:,1]) * db_acx

    db_vey = vgy[1,0] - vgy[0,0]
    db_acy = agy[1,0] - agy[0,0]
    v_normy = np.sum(vgy[:,1]) * db_vey
    a_normy = np.sum(agy[:,1]) * db_acy

    db_vez = vgz[1,0] - vgz[0,0]
    db_acz = agz[1,0] - agz[0,0]
    v_normz = np.sum(vgz[:,1]) * db_vez
    a_normz = np.sum(agz[:,1]) * db_acz

    # normalize from histogram to density
    vgx[:,1] = vgx[:,1] / v_normx
    agx[:,1] = agx[:,1] / a_normx
    vgy[:,1] = vgy[:,1] / v_normy
    agy[:,1] = agy[:,1] / a_normy
    vgz[:,1] = vgz[:,1] / v_normz
    agz[:,1] = agz[:,1] / a_normz


    # standardize
    mean_vx = 0.
    std_vx = 0.
    mean_ax = 0.
    std_ax = 0.

    mean_vy = 0.
    std_vy = 0.
    mean_ay = 0.
    std_ay = 0.

    mean_vz = 0.
    std_vz = 0.
    mean_az = 0.
    std_az = 0.

    for jj in range(vgx.shape[0]):
        mean_vx += db_vex * vgx[jj,1]*(vgx[jj,0] + db_vex/2)
        mean_ax += db_acx * agx[jj,1]*(agx[jj,0] + db_acx/2)
        mean_vy += db_vey * vgy[jj,1]*(vgy[jj,0] + db_vey/2)
        mean_ay += db_acy * agy[jj,1]*(agy[jj,0] + db_acy/2)
        mean_vz += db_vez * vgz[jj,1]*(vgz[jj,0] + db_vez/2)
        mean_az += db_acz * agz[jj,1]*(agz[jj,0] + db_acz/2)
    for jj in range(vgx.shape[0]):
        std_vx += db_vex * vgx[jj,1]*((vgx[jj,0] + db_vex/2) - mean_vx)**2.
        std_ax += db_acx * agx[jj,1]*((agx[jj,0] + db_acx/2) - mean_ax)**2.
        std_vy += db_vey * vgy[jj,1]*((vgy[jj,0] + db_vey/2) - mean_vy)**2.
        std_ay += db_acy * agy[jj,1]*((agy[jj,0] + db_acy/2) - mean_ay)**2.
        std_vz += db_vez * vgz[jj,1]*((vgz[jj,0] + db_vez/2) - mean_vz)**2.
        std_az += db_acz * agz[jj,1]*((agz[jj,0] + db_acz/2) - mean_az)**2.

    std_vx = np.sqrt(std_vx)
    std_ax = np.sqrt(std_ax)
    std_vy = np.sqrt(std_vy)
    std_ay = np.sqrt(std_ay)
    std_vz = np.sqrt(std_vz)
    std_az = np.sqrt(std_az)

    vg_stdx = np.copy(vgx)
    ag_stdx = np.copy(agx)
    vg_stdy = np.copy(vgy)
    ag_stdy = np.copy(agy)
    vg_stdz = np.copy(vgz)
    ag_stdz = np.copy(agz)

    vg_stdx[:,0] = (vg_stdx[:,0] - mean_vx) / std_vx
    vg_stdx[:,1] =  vg_stdx[:,1] * std_vx
    ag_stdx[:,0] = (ag_stdx[:,0] - mean_ax) / std_ax
    ag_stdx[:,1] =  ag_stdx[:,1] * std_ax

    vg_stdy[:,0] = (vg_stdy[:,0] - mean_vy) / std_vy
    vg_stdy[:,1] =  vg_stdy[:,1] * std_vy
    ag_stdy[:,0] = (ag_stdy[:,0] - mean_ay) / std_ay
    ag_stdy[:,1] =  ag_stdy[:,1] * std_ay

    vg_stdz[:,0] = (vg_stdz[:,0] - mean_vz) / std_vz
    vg_stdz[:,1] =  vg_stdz[:,1] * std_vz
    ag_stdz[:,0] = (ag_stdz[:,0] - mean_az) / std_az
    ag_stdz[:,1] =  ag_stdz[:,1] * std_az



    pdfsDict = {
        'binvx':vgx[:,0],
        'pdfvx':vgx[:,1],
        'binax':agx[:,0],
        'pdfax':agx[:,1],
        'binvx_std':vg_stdx[:,0],
        'pdfvx_std':vg_stdx[:,1],
        'binax_std':ag_stdx[:,0],
        'pdfax_std':ag_stdx[:,1],
        'pdfay_std':ag_stdy[:,1],
        'binvy':vgy[:,0],
        'pdfvy':vgy[:,1],
        'binay':agy[:,0],
        'pdfay':agy[:,1],
        'binvy_std':vg_stdy[:,0],
        'pdfvy_std':vg_stdy[:,1],
        'binay_std':ag_stdy[:,0],
        'pdfay_std':ag_stdy[:,1],
        'binvz':vgz[:,0],
        'pdfvz':vgz[:,1],
        'binaz':agz[:,0],
        'pdfaz':agz[:,1],
        'binvz_std':vg_stdz[:,0],
        'pdfvz_std':vg_stdz[:,1],
        'binaz_std':ag_stdz[:,0],
        'pdfaz_std':ag_stdz[:,1],
    }

    return pdfsDict


def save_pdfDict(pdfDict, path, train=None):
    train_str = '' if train is None else '_train{}'.format(train)
    a = pdfDict

    pdfvx = np.stack((a['binvx'], a['pdfvx'])).T
    filename = 'pdfvx{}.txt'.format(train_str)
    np.savetxt(osp.join(path, filename), pdfvx)

    pdfax = np.stack((a['binax'], a['pdfax'])).T
    filename = 'pdfax{}.txt'.format(train_str)
    np.savetxt(osp.join(path, filename), pdfax)

    pdfvx_std = np.stack((a['binvx_std'], a['pdfvx_std'])).T
    filename = 'pdfvx_std{}.txt'.format(train_str)
    np.savetxt(osp.join(path, filename), pdfvx_std)

    pdfax_std = np.stack((a['binax_std'], a['pdfax_std'])).T
    filename = 'pdfax_std{}.txt'.format(train_str)
    np.savetxt(osp.join(path, filename), pdfax_std)

    pdfvy = np.stack((a['binvy'], a['pdfvy'])).T
    filename = 'pdfvy{}.txt'.format(train_str)
    np.savetxt(osp.join(path, filename), pdfvy)

    pdfay = np.stack((a['binay'], a['pdfay'])).T
    filename = 'pdfay{}.txt'.format(train_str)
    np.savetxt(osp.join(path, filename), pdfay)

    pdfvy_std = np.stack((a['binvy_std'], a['pdfvy_std'])).T
    filename = 'pdfvy_std{}.txt'.format(train_str)
    np.savetxt(osp.join(path, filename), pdfvy_std)

    pdfay_std = np.stack((a['binay_std'], a['pdfay_std'])).T
    filename = 'pdfay_std{}.txt'.format(train_str)
    np.savetxt(osp.join(path, filename), pdfay_std)

    pdfvz = np.stack((a['binvz'], a['pdfvz'])).T
    filename = 'pdfvz{}.txt'.format(train_str)
    np.savetxt(osp.join(path, filename), pdfvz)

    pdfaz = np.stack((a['binaz'], a['pdfaz'])).T
    filename = 'pdfaz{}.txt'.format(train_str)
    np.savetxt(osp.join(path, filename), pdfaz)

    pdfvz_std = np.stack((a['binvz_std'], a['pdfvz_std'])).T
    filename = 'pdfvz_std{}.txt'.format(train_str)
    np.savetxt(osp.join(path, filename), pdfvz_std)

    pdfaz_std = np.stack((a['binaz_std'], a['pdfaz_std'])).T
    filename = 'pdfaz_std{}.txt'.format(train_str)
    np.savetxt(osp.join(path, filename), pdfaz_std)

    return



import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        '-path',
        type=str,
        required=True,
        help='path of the directory with files to analyze'
)
parser.add_argument(
        '-different_trainings',
        action='store_true',
        default=False,
        help='analyze each single train and also all together'
)
args = parser.parse_args()
path = args.path
different_trainings = args.different_trainings


# FIND pdf.DAT FILES AND COMPUTE PDF FROM HISTO
print('----- compute pdf -----')
if different_trainings:

    read_file_pdf_train1 = glob.glob(osp.join(path, f'pdf_train1_*.dat'))
    read_file_pdf_train2 = glob.glob(osp.join(path, f'pdf_train2_*.dat'))
    read_file_pdf_train3 = glob.glob(osp.join(path, f'pdf_train3_*.dat'))
    read_file_pdf_train4 = glob.glob(osp.join(path, f'pdf_train4_*.dat'))
    read_file_pdf_all = glob.glob(osp.join(path, f'pdf_train*_*.dat'))

    print('each training')

    pdfDict = compute_pdf(read_file_pdf_train1)
    save_pdfDict(pdfDict, path, train=1)
    pdfDict = compute_pdf(read_file_pdf_train2)
    save_pdfDict(pdfDict, path, train=2)
    pdfDict = compute_pdf(read_file_pdf_train3)
    save_pdfDict(pdfDict, path, train=3)
    pdfDict = compute_pdf(read_file_pdf_train4)
    save_pdfDict(pdfDict, path, train=4)

    print('all together')

    pdfDict = compute_pdf(read_file_pdf_all)
    save_pdfDict(pdfDict, path, train=None)

else:

    read_file_pdf = glob.glob(osp.join(path, f'pdf_*.dat'))

    pdfDict = compute_pdf(read_file_pdf)
    save_pdfDict(pdfDict, path, train=None)




# FIND sf.DAT FILES, FOR EACH COMPUTE LOG DERIVATIVES
# AND THEN COMPUTE MEAN FOR EACH TRAINING AND AMONG ALL
# TRAININGS

print('----- compute sf -----')

if different_trainings:

    read_file_sf_train1 = glob.glob(osp.join(path, f'sf_train1_*.dat'))
    read_file_sf_train2 = glob.glob(osp.join(path, f'sf_train2_*.dat'))
    read_file_sf_train3 = glob.glob(osp.join(path, f'sf_train3_*.dat'))
    read_file_sf_train4 = glob.glob(osp.join(path, f'sf_train4_*.dat'))
    read_file_sf_all = glob.glob(osp.join(path, f'sf_train*_*.dat'))
    read_file_dl_train1 = glob.glob(osp.join(path, f'sfdl_train1_*.dat'))
    read_file_dl_train2 = glob.glob(osp.join(path, f'sfdl_train2_*.dat'))
    read_file_dl_train3 = glob.glob(osp.join(path, f'sfdl_train3_*.dat'))
    read_file_dl_train4 = glob.glob(osp.join(path, f'sfdl_train4_*.dat'))
    read_file_dl_all = glob.glob(osp.join(path, f'sfdl_train*_*.dat'))

    print('each training')

    sf = compute_sf_mean(read_file_sf_train1)
    np.savetxt(osp.join(path, 'sf_train1.txt'), sf)
    dl = compute_sf_mean(read_file_dl_train1)
    np.savetxt(osp.join(path, 'sfdl_train1.txt'), dl)

    sf = compute_sf_mean(read_file_sf_train1)
    np.savetxt(osp.join(path, 'sf_train2.txt'), sf)
    dl = compute_sf_mean(read_file_dl_train2)
    np.savetxt(osp.join(path, 'sfdl_train2.txt'), dl)

    sf = compute_sf_mean(read_file_sf_train3)
    np.savetxt(osp.join(path, 'sf_train3.txt'), sf)
    dl = compute_sf_mean(read_file_dl_train3)
    np.savetxt(osp.join(path, 'sfdl_train3.txt'), dl)

    sf = compute_sf_mean(read_file_sf_train4)
    np.savetxt(osp.join(path, 'sf_train4.txt'), sf)
    dl = compute_sf_mean(read_file_dl_train4)
    np.savetxt(osp.join(path, 'sfdl_train4.txt'), dl)

    print('all together')

    sf = compute_sf_mean(read_file_sf_all)
    np.savetxt(osp.join(path, 'sf.txt'), sf)
    dl = compute_sf_mean(read_file_dl_all)
    np.savetxt(osp.join(path, 'sfdl.txt'), dl)

else:

    read_file_sf = glob.glob(osp.join(path, f'sf_*.dat'))
    read_file_dl = glob.glob(osp.join(path, f'sfdl_*.dat'))

    sf = compute_sf_mean(read_file_sf)
    np.savetxt(osp.join(path, 'sf.txt'), sf)
    dl = compute_sf_mean(read_file_dl)
    np.savetxt(osp.join(path, 'sfdl.txt'), dl)



