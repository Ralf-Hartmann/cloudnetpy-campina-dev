import xarray as xr
import datetime as dt
from os.path import isdir,isfile, join
from os import makedirs
import glob
import numpy as np
from skimage.filters import threshold_otsu



######################################################
# Working File
# Adapt closer to cloudnetpy style later
# E.g adapt .plotting to automaticall plankton product module
######################################################





def prepare_classification_data(
    classification: xr.DataArray,
    target_classes: list = [9, 10],  # Default: both insects (9) and aerosol&insects (10)
    height_min: int = 0,
    height_max: int = 130,
    binarize: bool = True,
) -> xr.DataArray:
    
    """
    Processes classification data by:
    - Selecting specified target classes (e.g., insects, aerosects, or both).
    - Restricting data to a height range.
    - Optionally binarizing the data (0 = non-target, 1 = target).

    Args:
        classification (xr.DataArray): Input classification data.
        target_classes (list): List of target classes (e.g., [9] for insects, [10] for aerosects, [9,10] for both).
        height_min (int): Minimum height bin index.
        height_max (int): Maximum height bin index.
        binarize (bool): If True, convert to binary (0/1) labels.

    Returns:
        xr.DataArray: Processed data with renamed coordinates.
    """


    #Select target classes and binarize if needed
    if binarize:
        processed_data = classification.where(classification.isin(target_classes)).fillna(0)
        processed_data = xr.where(processed_data.isin(target_classes), 1, 0)
    else:
        processed_data = classification.where(classification.isin(target_classes))

    # Apply height gating
    processed_data = processed_data.isel(height=slice(height_min, height_max))

    # Rename the DataArray based on target classes
    if target_classes == [9]:
        processed_data = processed_data.rename("insects")
    elif target_classes == [10]:
        processed_data = processed_data.rename("aerosects")
    elif sorted(target_classes) == [9, 10]:
        processed_data = processed_data.rename("insect_aerosects")
    else:
        processed_data = processed_data.rename("custom_class")

    return processed_data


def pixel_density(time_size, height_size, dataset):
   

    """    Calculate pixel density from a dataset with time and height dimensions.
    Args:       time_size (int): Size of the time window in minutes.
                height_size (int): Size of the height window in height bins NOT METERS.
                dataset (xr.DataArray): Input dataset with 'time' and 'height' dimensions.
    Returns:        xr.DataArray: Density of insect detection pixels.
    """ 

    def time_window(dataset, minutes):

        #time_deltas = dataset['time'].diff("time").values
        
        #30 seconds is timestep used by cloudnet
        window_size = int((minutes * 60) / 30)
    
        return  window_size 

    # window size 
    window_size = time_window(dataset, time_size)  
    
    rolling_sum = dataset.rolling(time=window_size, height=height_size, center=True, min_periods = 1).sum()
    

    # Window area
    window_area = window_size * height_size

    # Detections per time * height -> concentration of insect detection pixels
    density =  rolling_sum / window_area

    density = density.rename(f"{dataset.name}_pixel_density")
    
    # height_resolution = dataset['height'].diff('height').mean().item()
    
    # # Calculate the height size in terms of number of height bins
    # height_window_meters = height_resolution * height_size

    # print(f"Rolling window dimensions (time x height): {window_size} minutes x {height_window_meters} meters")
    # print(f"Total possible pixels per window: {window_size * height_size}")


    return density

def otsu_threshold(dataset):
    """
    Calculate Otsu's threshold for the given dataset.
    
    Args:
        dataset (xr.DataArray): Input dataset with insect concentration data.
        
    Returns:
        float: Otsu's threshold value.
    """
    # Flatten the dataset values and remove NaNs
    flat_values = dataset.values.flatten()
    flat_values = flat_values[~np.isnan(flat_values)]
    
    # Calculate Otsu's threshold
    otsu_value = threshold_otsu(flat_values)
    
    return otsu_value

def threshold(dataset, threshold_value):
    threshold = np.nanpercentile(dataset, threshold_value) 
    return threshold

def otsu_to_percentile(dataset, otsu_value):
    """
    Convert an Otsu threshold value to its equivalent percentile in the dataset.
    """
    flat_values = dataset.values.flatten()
    flat_values = flat_values[~np.isnan(flat_values)]
    return (flat_values < otsu_value).sum() / len(flat_values)

def calculate_pblh_otsu(dataset):

    """    Calculate the Planetary Boundary Layer Height (PBLH) using Otsu's thresholding method.
    Args:        dataset (xr.DataArray): Input dataset with insect concentration data.
    Returns:        tuple: Estimated PBLH and Otsu's threshold value.

    """ 
    
    # Apply Otsu's threshold to the dataset
    otsu_threshold_value = otsu_threshold(dataset)
    percentile = otsu_to_percentile(dataset, otsu_threshold_value)
    thresholded_dataset = dataset.where(dataset >= otsu_threshold_value)
 
    # Calculate PBLH based on the maximum insect concentration
    pblh_raw = thresholded_dataset["height"].where(~np.isnan(thresholded_dataset)).idxmax(dim="height")
    
    return pblh_raw, percentile 




def calculate_pblh(dataset, threshold_value):

    """    Calculate the Planetary Boundary Layer Height (PBLH) based on insect concentration data.
    Args:        dataset (xr.DataArray): Input dataset with insect concentration data.
                threshold_value (float): Threshold value for insect concentration to determine PBLH.
    Returns:        xr.DataArray: Estimated PBLH based on the maximum insect concentration above the threshold.
    """

    percentile = threshold_value * 100

    # Apply threshold, keep only values above the percentile
    thresholded_dataset = dataset.where(dataset < np.nanpercentile(dataset, percentile))

    # Compute PBL
    pblh = thresholded_dataset["height"].where(~np.isnan(thresholded_dataset)).idxmax(dim="height")

    return pblh

'''
Mask PBLH values during precipitation periods and interpolate over these gaps.


'''
def mask_pblh_rain(pblh, precip_mask, time_buffer=30):
    """Mask PBLH values for times with precipitation ±time_buffer."""
    rain_present = precip_mask.notnull() if precip_mask.dtype.kind == 'f' else precip_mask.astype(bool)
    rain_times = rain_present.any(dim="height").values
    nt = len(rain_times)
    mask_times = np.zeros(nt, dtype=bool)
    rain_indices = np.where(rain_times)[0]
    for t in rain_indices:
        t0, t1 = max(0, t - time_buffer), min(nt, t + time_buffer + 1)
        mask_times[t0:t1] = True

    pblh_masked = pblh.copy()
    pblh_masked["pblh"] = pblh["pblh"].where(~mask_times, np.nan)
    return pblh_masked, mask_times


def remove_outliers(pblh, jump_threshold=3):
    """Remove large jumps in PBLH (outlier filter)."""
    pblh_values = pblh["pblh"].values
    diffs = np.abs(np.diff(pblh_values, prepend=np.nan))
    jump_mask = diffs > jump_threshold
    pblh_filtered = pblh.copy()
    pblh_filtered["pblh"] = pblh["pblh"].where(~jump_mask, np.nan)
    return pblh_filtered, jump_mask


def interpolate_over_rain(pblh_masked, rain_mask_times, method):
    """
    Interpolate PBLH only over rain-masked NaNs, including edges.
    """
    pblh_interp = pblh_masked.copy()
    values = pblh_masked["pblh"].values
    original_nan = np.isnan(values)
    
    # Interpolation for internal points
    interp_vals = pblh_masked["pblh"].interpolate_na(dim="time", method= method)
    
    # Fill edges using 
    interp_vals = interp_vals.ffill(dim="time").bfill(dim="time")
    
    # Only apply interpolated values where the rain mask was active 
    pblh_interp["pblh"] = xr.where(rain_mask_times & original_nan, interp_vals, values)
    
    return pblh_interp