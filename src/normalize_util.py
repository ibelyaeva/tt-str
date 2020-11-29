import numpy as np


def normalize_components(img,mask_img, tr=2.0):
    from nilearn.input_data import NiftiMasker
    masker = NiftiMasker(memory='nilearn_cache', mask_img = mask_img, memory_level=1,
                     standardize=True, detrend=False, t_r = tr)
    
    data_masked = masker.fit_transform(img)
    data_masked_norm = masker.inverse_transform(data_masked)
    print("Normalized: mean %.2f" % np.mean(data_masked))
    print("Normalized: std %.2f" % np.std(data_masked))
    return data_masked_norm


def normalize_tc(img,mask_img, tr=2.0):
    from nilearn.input_data import NiftiMasker
    masker = NiftiMasker(memory='nilearn_cache', mask_img = mask_img, memory_level=1, low_pass=0.15,
                     standardize=True, detrend=True, t_r = tr)
    
    data_masked = masker.fit_transform(img)
    data_masked_norm = masker.inverse_transform(data_masked)
    print("Normalized TC: mean %.2f" % np.mean(data_masked))
    print("Normalized TC: std %.2f" % np.std(data_masked))
    return data_masked_norm

def get_matrix_by_columns(x, column_list):
    result = x[:,column_list]
    return result