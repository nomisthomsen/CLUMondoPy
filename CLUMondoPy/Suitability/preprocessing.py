import numpy as np
import rasterio
from scipy.ndimage import convolve

def convert_list_to_np_stack(file_list):
    raster_arrays = []
    for ras_file in file_list:
        with rasterio.open(ras_file) as asc:
            raster_arrays.append(asc.read(1))
    return np.stack(raster_arrays, axis=0)

def fill_nan_with_adjacent_mean(env_array, region_array, region_mask, no_data_value=-9999):
    filled_array = env_array.copy()
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
    for i in range(env_array.shape[0]):
        layer = filled_array[i]
        missing_mask = (region_array == 0) & ((layer == no_data_value) | np.isnan(layer))
        if np.any(missing_mask):
            valid_mask = (layer != no_data_value) & (~np.isnan(layer))
            sum_adj = convolve(layer * valid_mask, kernel, mode='constant', cval=0)
            count_adj = convolve(valid_mask.astype(int), kernel, mode='constant', cval=0)
            mean_adj = np.divide(sum_adj, count_adj, where=count_adj > 0)
            layer[missing_mask] = mean_adj[missing_mask]
    return filled_array

def standardize_array(env_array, region_array, region_mask):
    standardized_array = env_array.copy()
    for i in range(env_array.shape[0]):
        layer = env_array[i]
        valid_mask = region_array == region_mask
        if np.any(valid_mask):
            valid_values = layer[valid_mask].astype(float)
            mean, std = np.mean(valid_values), np.std(valid_values)
            if std > 0:
                standardized_array[i][valid_mask] = (valid_values - mean) / std
    return standardized_array