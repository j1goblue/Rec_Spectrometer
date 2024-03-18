import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn.functional as F
from params import *

def lorentzian(x, x0, gamma=1):
    """Lorentzian function centered at x0 with width gamma."""
    return gamma / (np.pi * ((x - x0)**2 + gamma**2))

# linear decay function for "width-lambda" effect
def decay_ratio(lam, beta=0.25):
    q = (beta-1) * lam / num_wavelengths + 1
    return q

"""Create spectrum data"""
def create_lorentzian_spectrum(min_dist=40, p=None, width_decay=False, add_base=False):
    # 1. generate 1-,2-,3-peak samples w.r.t. probability p:
    if p:
        num_peaks = np.random.choice(np.arange(max_num_peaks)+1, p)
    else:
        num_peaks = np.random.randint(1, max_num_peaks+1)
    
    widths = (width_range[1] - width_range[0]) * np.random.rand(num_peaks) + width_range[0]
    magnitudes = (magnitude_range[1] - magnitude_range[0]) * np.random.rand(num_peaks, 1) + magnitude_range[0]
    
    # 2. keep a minimum distance between each peak:
    if min_dist:
        wavelength_values = np.arange(peak_range[0], peak_range[1], 1)
        mu_list = []
        while len(mu_list) < num_peaks:
            m_temp = np.random.choice(wavelength_values, 1, replace=False)[0]
            if all(abs(m_temp - m) >= min_dist for m in mu_list):
                mu_list.append(m_temp) 
        mu = np.asarray(mu_list)
    else:
        mu = np.random.choice(np.arange(peak_range[0], peak_range[1]), num_peaks, replace=False)
    
    # 3. apply "width-lambda" effect:
    if width_decay:
        widths = decay_ratio(mu) * widths

    x_stack = np.zeros((num_peaks, num_wavelengths))
    x_values = np.arange(0, num_wavelengths, step=1)
    for j in range(num_peaks):
        x_stack[j, :] = lorentzian(x_values, mu[j], widths[j])
        x_stack[j, :] /= np.max(x_stack[j, :])
    x_stack = magnitudes * x_stack

    # 4. add base
    if add_base:
        sp_mask = np.random.rand((num_wavelengths)) > 1-sp_prob
        sp_vals = np.zeros_like(sp_mask).astype(float)
        sp_vals[sp_mask] = np.random.rand(np.sum(sp_mask))
        sp_base = gaussian_filter1d(sp_vals, sigma=sp_sigma)
        sp_base /= sp_base.max()
        sp_base *= sp_thres

        x += sp_base
    
    x = np.sum(x_stack, axis=0)
    x -= np.min(x)
    x /= np.max(x)
    # generate corresponding peak ground truth
    x_pk = np.zeros_like(x)
    x_pk[mu] = x[mu]

    return torch.from_numpy(x), torch.from_numpy(x_pk)

"""Noiseless batch"""
def create_batch_noiseless(batch_size, p=None, add_base=False):
    X = torch.zeros((batch_size, num_wavelengths))
    X_pk = torch.zeros((batch_size, num_wavelengths))
    for i in range(batch_size):
        X[i, :], X_pk[i, :] = create_lorentzian_spectrum(p=p, add_base=add_base)

    y = torch.matmul(X, R.T)
    
    return X, X_pk, y

"""Noisy batch"""
def create_batch_noisy(batch_size, min_dist=40, p=None, width_decay=False, add_base=False):
    X = torch.zeros((batch_size, num_wavelengths))
    X_pk = torch.zeros((batch_size, num_wavelengths))
    for i in range(batch_size):
        X[i, :], X_pk[i, :] = create_lorentzian_spectrum(min_dist=min_dist, p=p, width_decay=width_decay, add_base=add_base)

    M, N = 2, 4
    ## step 1: add Gaussian noise on R
    y_temp = torch.zeros(batch_size*M, num_sensors)
    
    err_percent_R = 0.05
    for m in range(M):
        R_noise = err_percent_R * R * torch.randn_like(R)
        R_temp = R + R_noise
        y_temp[batch_size*m:batch_size*(m+1),:] = torch.matmul(X, R_temp.T)

    ## step 2: add noise on y_
    y = torch.repeat_interleave(y_temp, N, dim=0)

    noise_scale_y = 1e-5
    # y += noise_scale_y * torch.randn_like(y)
    # y += noise_scale_y * torch.abs(torch.randn_like(y))
    y += noise_scale_y * F.relu(torch.randn_like(y))
    
    ## step 3: repeat X accordingly
    X_temp = X.repeat(M, 1)
    X_rep = torch.repeat_interleave(X_temp, N, dim=0)
    X_pk_temp = X_pk.repeat(M, 1)
    X_pk_rep = torch.repeat_interleave(X_pk_temp, N, dim=0)

    # step 4: shuffle the augmented batch
    perm_indices = torch.randperm(batch_size*M*N)
    X_batch = X_rep[perm_indices]
    X_pk_batch = X_pk_rep[perm_indices]
    y_batch = y[perm_indices]
    
    return X_batch, X_pk_batch, y_batch