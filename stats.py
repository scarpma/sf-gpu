#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np


dict3d = {
    'x2': 1,'y2': 2,'z2': 3,'xy': 4,'yz': 5,
    'xz': 6,'x4': 7,'y4': 8,'z4': 9,'x2y2': 10,
    'y2z2': 11,'x2z2': 12,'x6': 13,'x6': 14,
    'x6': 15,'x8': 16,'x8': 17,'x8': 18,
    'x10': 19,'x10': 20,'x10': 21}
dict1d = {
    'x2': 1,'x4': 2,'x6': 3,'x8': 4,
    'x10': 5,'x12': 6}


def compute_sf_mixed(vel, taus, device, dtype):
    vel = vel.astype(dtype)
    sf = np.zeros((21,len(taus)))
    taus = taus.astype(np.int64)

    time_len = vel.shape[1]
    vel_ = torch.from_numpy(vel).to(device)

    for tau_idx, tau in enumerate(taus):
        t_1 = torch.tensor(np.arange(tau, time_len))
        t_2 = torch.tensor(np.arange(0, time_len - tau))

        diffx = vel_[:, t_1, 0] - vel_[:, t_2, 0]
        diffy = vel_[:, t_1, 1] - vel_[:, t_2, 1]
        diffz = vel_[:, t_1, 2] - vel_[:, t_2, 2]

        # <(du_i)(du_j)>
        sf[3,tau_idx] += torch.sum(diffx*diffy).cpu()
        sf[4,tau_idx] += torch.sum(diffy*diffz).cpu()
        sf[5,tau_idx] += torch.sum(diffx*diffz).cpu()
        # <(du_i)**2>
        diffx = torch.pow(diffx,2)
        diffy = torch.pow(diffy,2)
        diffz = torch.pow(diffz,2)
        sf[0,tau_idx] += torch.sum(diffx).cpu()
        sf[1,tau_idx] += torch.sum(diffy).cpu()
        sf[2,tau_idx] += torch.sum(diffz).cpu()
        # <(du_i)**4>        
        sf[6,tau_idx] += torch.sum(torch.pow(diffx,2)).cpu()
        sf[7,tau_idx] += torch.sum(torch.pow(diffy,2)).cpu()
        sf[8,tau_idx] += torch.sum(torch.pow(diffz,2)).cpu()
        # <(du_i)**2(du_j)**2>
        sf[9,tau_idx] += torch.sum(diffx*diffy).cpu()
        sf[10,tau_idx] += torch.sum(diffy*diffz).cpu()
        sf[11,tau_idx] += torch.sum(diffx*diffz).cpu()
        # <(du_i)**6>
        sf[12,tau_idx] += torch.sum(torch.pow(diffx,3)).cpu()
        sf[13,tau_idx] += torch.sum(torch.pow(diffy,3)).cpu()
        sf[14,tau_idx] += torch.sum(torch.pow(diffz,3)).cpu()
        # <(du_i)**8>
        sf[15,tau_idx] += torch.sum(torch.pow(diffx,4)).cpu()
        sf[16,tau_idx] += torch.sum(torch.pow(diffy,4)).cpu()
        sf[17,tau_idx] += torch.sum(torch.pow(diffz,4)).cpu()
        # <(du_i)**10>
        sf[18,tau_idx] += torch.sum(torch.pow(diffx,5)).cpu()
        sf[19,tau_idx] += torch.sum(torch.pow(diffy,5)).cpu()
        sf[20,tau_idx] += torch.sum(torch.pow(diffz,5)).cpu()

    return sf

def compute_sf(vel, taus, device, dtype):
    vel = vel.astype(dtype)
    sf = np.zeros((6,len(taus)))
    taus = taus.astype(np.int64)

    time_len = vel.shape[1]
    vel_ = torch.from_numpy(vel[:,:,0]).to(device)

    for tau_idx, tau in enumerate(taus):
        t_1 = torch.tensor(np.arange(tau, time_len))
        t_2 = torch.tensor(np.arange(0, time_len - tau))

        diff = vel_[:, t_1] - vel_[:, t_2]
        diff = torch.pow(diff)
        # <(du_x)**2>
        sf[0,tau_idx] += torch.sum(diff).cpu()
        # <(du_x)**4>
        sf[1,tau_idx] += torch.sum(torch.pow(diff,2)).cpu()
        # <(du_x)**6>
        sf[2,tau_idx] += torch.sum(torch.pow(diff, 3)).cpu()
        # <(du_x)**8>
        sf[3,tau_idx] += torch.sum(torch.pow(diff, 4)).cpu()
        # <(du_x)**10>
        sf[4,tau_idx] += torch.sum(torch.pow(diff, 5)).cpu()
        # <(du_x)**12>
        sf[5,tau_idx] += torch.sum(torch.pow(diff, 6)).cpu()

    return sf


def sf_wrap(
    vel,
    device=torch.device('cuda'),
    dtype=np.float64,
    chunk_size=False,
    mode='1d',
):
    compute = compute_sf if mode=='1d' else compute_sf_mixed

    # DEFINE TAUS
    taus = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15,
        18, 22, 26, 31, 38, 46, 55, 66, 79,
        95, 114, 137, 164, 197, 237, 284,
        341, 410, 492, 590, 708, 850, 1020],
        dtype=np.int64)

    # DO COMPUTATION ON GPU OR CPU (IN PARALLEL)    
    if chunk_size:
        n_chunks = int(np.ceil(vel.shape[0] / chunk_size))
        idxs = np.arange(0, min([chunk_size,vel.shape[0]]))
        vel_chunk = vel[idxs]
        sf_chunk = compute(vel_chunk, taus, device, dtype=dtype)
        sf = sf_chunk
        for ii in range(1,n_chunks):
            idxs = np.arange(ii*chunk_size, min([(ii+1)*chunk_size,vel.shape[0]]))
            vel_chunk = vel[idxs]
            sf_chunk = compute(vel_chunk, taus, device, dtype=dtype)
            sf = sf + sf_chunk
    else:
        sf = compute(vel, taus, device, dtype=dtype)

    # NORMALIZE
    time_len = vel.shape[1]
    tau_idxs = np.arange(sf.shape[1])
    sf[:,tau_idxs] = sf[:,tau_idxs] / (( time_len - taus ) * vel.shape[0])
    sf = np.r_[taus.reshape(1,-1), sf]

    return sf
