import torch
import numpy as np

from params import num_sensors, num_wavelengths, start_wavelength
# from params import lam_gts_comm_1pk

# load & preprocess real dataset
def load_real_dataset(data_file_list, Ip_dir, Spec_dir):
    num_samples = len(data_file_list)

    real_y_batch = torch.empty((num_samples, num_sensors), dtype=torch.float)
    real_X_batch = torch.empty((num_samples, num_wavelengths), dtype=torch.float)

    for idx, fname in enumerate(data_file_list):
        real_y = torch.from_numpy(np.genfromtxt((Ip_dir+"Ip_{fname}").format(fname=fname), dtype=np.float32))
        real_y_batch[idx, :] = real_y

        raw_X = np.genfromtxt((Spec_dir+"{fname}").format(fname=fname), dtype=np.float32)
        eff_wavelengths = np.arange(start_wavelength, start_wavelength+num_wavelengths)
        full_x = np.interp(eff_wavelengths, raw_X[:, 0], raw_X[:, 1]).reshape(-1)
        full_x = full_x - full_x.min()
        X_gt = full_x / full_x.max()
        real_X_batch[idx, :] = torch.from_numpy(X_gt)

    return real_y_batch, real_X_batch


# normalization functions
def max_normalize(y):
    return y / torch.amax(y, dim=1)[:,None]

def log_transform(y):
    return torch.log(y)

def min_max_scaling(y):
    min_val = torch.min(y, dim=1, keepdim=True).values
    max_val = torch.max(y, dim=1, keepdim=True).values
    return (y - min_val) / (max_val - min_val)

def log_min_max_mapping(y):
    return min_max_scaling(log_transform(y))

# numerically safe version for normalization functions
def log_transform_safe(y):
    eps = 1e-15
    safe_y = torch.clamp(y, min=eps)
    return torch.log(safe_y)

def min_max_scaling_safe(y):
    eps = 1e-15
    min_val = torch.min(y, dim=1, keepdim=True).values
    max_val = torch.max(y, dim=1, keepdim=True).values
    return (y - min_val) / (max_val - min_val + eps)

def log_min_max_mapping_safe(y):
    return min_max_scaling_safe(log_transform_safe(y))

# unified version for all normalization functions above
def normalize_y(y, func=None):
    if func:
        return func(y)
    return y


# weighted MSELoss
def rescale_loss(pred, tar, alpha=10):
    loss = (pred - tar) ** 2
    loss[tar > 0] = loss[tar > 0] * alpha
    loss = torch.mean(loss)
    return loss

def rescale_loss_mix(pred, tar_spec, tar_pk, alpha=10):
    loss = (pred - tar_spec) ** 2
    loss[tar_pk > 0] = loss[tar_pk > 0] * alpha
    loss = torch.mean(loss)
    return loss

def rescale_loss_reduce_base(pred, tar_spec, alpha=10.0, thres=0.25):
    loss = (pred - tar_spec) ** 2
    loss[tar_spec > thres] = loss[tar_spec > thres] * alpha
    loss = torch.mean(loss)
    return loss

# # robust scaling: IQR method
# def robust_scale(data):
#     # Calculate the median for each feature
#     medians = torch.median(data, dim=0).values

#     # Calculate the interquartile range for each feature
#     q1 = torch.quantile(data, 0.25, dim=0)
#     q3 = torch.quantile(data, 0.75, dim=0)
#     iqr = q3 - q1

#     # Avoid dividing by zero (or a very small number) by replacing zeros with ones
#     iqr[iqr == 0] = 1.

#     # Subtract the median and scale by the interquartile range
#     scaled_data = (data - medians) / iqr

#     return scaled_data

# evaluate RMSE on real dataset
def eval_RMSE(model, y_data, gt_data, normalization_func=None):
    model.eval()
    model.cpu()
    with torch.no_grad():
        input_y_data = normalize_y(y_data, func=normalization_func)
        pred_data = model(input_y_data)
        pred_data = pred_data / torch.max(pred_data, dim=1, keepdim=True)[0]

        rmse_val = torch.mean(torch.sqrt(torch.mean((pred_data - gt_data)**2, dim=1)))
   
    return rmse_val.item()

# # evaluate MAE on real data (only for single-peak samples)
# def eval_MAE_single_peak(model, y_data, gt_data, normalization_func=None, lam_gt_1pk=lam_gt_comm_1pk, verbose=False):
#     model.eval()
#     model.cpu()
#     with torch.no_grad():
#         input_y_data = normalize_y(y_data, func=normalization_func)
#         pred_data = model(input_y_data)
#         pred_data = pred_data / torch.max(pred_data, dim=1, keepdim=True)[0]

#         # compute relative MAE
#         lam_pred_dl = (start_wavelength + torch.argmax(pred_data, dim=1)).tolist()
#         lam_pred_dl_1pk = lam_pred_dl[-10:]

#         lam_dl_errs_1pk = []
#         for idx in range(len(lam_gt_1pk)):
#             lam_err_dl = (lam_pred_dl_1pk[idx] - lam_gt_1pk[idx]) / lam_gt_1pk[idx]
#             lam_dl_errs_1pk.append(100.0*lam_err_dl)

#     mae_val = np.round(np.mean(np.abs(lam_dl_errs_1pk)), 2)

#     if verbose:
#         print("The single-peak predictions: {}.".format(lam_pred_dl_1pk))
#         print("The single-peak ground truth: {}.".format(lam_gt_1pk))
#         # print("The major peak predictions for double-peak: {}.".format(lam_pred_dl[:4]))
#         # print("The major peak prediction for 3LED: {}.".format(lam_pred_dl[4]))

#     return mae_val

# MAE benchmark functions
def eval_MAE_single_peak(preds, lam_gts, verbose=False):

    lam_preds = (start_wavelength + np.argmax(preds, axis=1)).tolist()
    lam_errs = []
    for idx in range(len(lam_gts)):
        lam_err = (lam_preds[idx] - lam_gts[idx]) / lam_gts[idx]
        lam_errs.append(100.0*lam_err)
    
    mae_val = np.round(np.mean(np.abs(lam_errs)), 3)

    if verbose:
        print("-"*37)
        print("| idx | gt_lam | pred_lam | rel_err |")
        print("-"*37)
        for idx, (pred_val, gt_val, rel_err) in enumerate(zip(lam_preds, lam_gts, lam_errs)):
            print(f"|  {idx}  | {gt_val:.2f} |    {pred_val}   | {rel_err: 7.3f} |")
        print("-"*37)
    return mae_val

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
def eval_MAE_double_peak(preds, lam_gts, flt_sig=4.0, verbose=False):

    lam_errs = []
    lam_preds = []
    for idx in range(preds.shape[0]):
        pred_smoothed = gaussian_filter1d(preds[idx,:], sigma=flt_sig)
        peaks, _ =  find_peaks(pred_smoothed)
        int_values = pred_smoothed[peaks]

        top_2_indices = np.argsort(int_values)[-2:]
        lam_pred = peaks[np.sort(top_2_indices)]
        lam_pred += start_wavelength
        lam_preds.append(lam_pred)

        for i in range(2):
            lam_err = (lam_pred[i] - lam_gts[idx][i]) / lam_gts[idx][i]
            lam_errs.append(100*lam_err)

    mae_val = np.round(np.mean(np.abs(lam_errs)), 3)    

    if verbose:
        multi_peak_names = ['2_S1','2_S3','2_S4','2_S5','3LED']
        print("-"*38)
        print("| name | gt_lam | pred_lam | rel_err |")
        print("-"*38)
        for idx, name in enumerate(multi_peak_names):
            for j in range(2):
                print(f"| {name} | {lam_gts[idx][j]} |    {lam_preds[idx][j]}   | {lam_errs[2*idx+j]: 7.3f} |")            
            print("-"*38)

    return mae_val