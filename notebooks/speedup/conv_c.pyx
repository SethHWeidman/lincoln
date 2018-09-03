
import cython
import torch
#cython: language_level=3, boundscheck=False

import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def _pad_1d_obs_conv(np.ndarray obs,
                     np.int pad):
    cdef np.ndarray a = np.zeros(pad, dtype=DTYPE)
    z = np.concatenate([a, obs, a])
    return z
