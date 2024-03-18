import numpy as np
import torch

# use GPU (or CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# simulated data parameters
R: torch.FloatTensor = torch.from_numpy(np.genfromtxt('response_matrix.txt', dtype=np.float32))
trunc_num = 0
if trunc_num:
    R = R[:,:-trunc_num]
# R_inv = torch.linalg.pinv(R).to(device)
start_wavelength = 400

num_sensors =  R.size(0)
num_wavelengths = R.size(1)

max_num_peaks = 3
width_range = (10, 25)
magnitude_range = (0.6, 1.4)

peak_range = (0, 205)

# training parameters
learning_rate = 0.0005
batch_size = 256
num_iters = 3000

# commercial gt & nnls prediction
lam_gts_comm_1pk = [400.66, 
                    426.08, 
                    453.36, 
                    467.03, 
                    475.3, 
                    520.74, 
                    561, 
                    569.25, 
                    592.36, 
                    631.49]
# lam_pred_nnls_1pk = [411, 425, 450, 468, 480, 524, 566, 570, 591, 634]

lam_gts_comm_2pk = [[399.92, 504.55], 
                    [465.09, 593.45], 
                    [402.09, 576.79], 
                    [467.22, 570.99], 
                    [467.57, 569.28]]