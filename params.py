#!/usr/bin/env python
# coding: utf-8

REAL_DB_PATH = ('/m100_work/INF21_fldturb_0/velocities.npy')
COMPONENTS = slice(0,3)
DB_NAME = "tracers"
WGAN_TYPE = "wgangp3"

SIG_LEN = 2000
CHANNELS = 3
NOISE_DIM = 100

DB_MAX = 10.314160931635518
DB_MIN = -11.244815117091042

# Activate to smoothen training dataset
SMOOTH_REAL_DB = False
if SMOOTH_REAL_DB:
    sigma_smooth_real=2
    trunc_smooth_real=5
