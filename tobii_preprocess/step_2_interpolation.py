# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 17:27:12 2026

@author: bbf2518
"""
# pupil_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from pathlib import Path

# =========================
# Core helper functions
# =========================

def detect_bad_data_by_slope(pupil_signal, slope_threshold=0.03, n_passes=120):
    """
    Detect bad pupil data based on slope (derivative). Marks bad points as 0.
    
    Parameters
    ----------
    pupil_signal : np.ndarray
        1D array of pupil diameters (mm)
    slope_threshold : float
        Threshold of derivative to mark a point as bad
    n_passes : int
        Number of passes through the signal to capture peaks
    
    Returns
    -------
    cleaned_signal : np.ndarray
        Signal with bad points marked as 0
    bad_mask : np.ndarray
        Boolean mask of bad points
    """
    cleaned_signal = pupil_signal.copy()
    n_samples = len(cleaned_signal)
    
    for _ in range(n_passes):
        # derivative between consecutive points
        dy = np.diff(cleaned_signal, prepend=cleaned_signal[0])
        # mark large slopes as bad
        bad_indices = np.where(np.abs(dy) > slope_threshold)[0]
        cleaned_signal[bad_indices] = 0
        
        # also mark previous and next points
        prev_indices = bad_indices[bad_indices > 0] - 1
        next_indices = bad_indices[bad_indices < n_samples-1] + 1
        cleaned_signal[prev_indices] = 0
        cleaned_signal[next_indices] = 0
        
        # single points flanked by zeros
        for i in range(1, n_samples-1):
            if cleaned_signal[i-1] == 0 and cleaned_signal[i+1] == 0:
                cleaned_signal[i] = 0
    
    # end-of-recording handling
    if cleaned_signal[-1] == 0:
        mean_valid = safe_nanmean(cleaned_signal[cleaned_signal != 0])
        cleaned_signal[-n_passes:] = mean_valid if not np.isnan(mean_valid) else 0
    
    bad_mask = cleaned_signal == 0
    return cleaned_signal, bad_mask


def interpolate_signal(signal, first_samples_bad=False, n_prepend=120):
    """
    Interpolate zeros in a signal (pupil or gaze) with linear interpolation.
    Replaces fully empty or zero-only signals with the mean of the rest of the signal.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal
    first_samples_bad : bool
        True if first n_prepend samples need prepending
    n_prepend : int
        Number of prepended samples if first samples are bad

    Returns
    -------
    interpolated_signal : np.ndarray
        Interpolated signal with no NaNs
    """
    signal = np.asarray(signal, dtype=float)
    n_samples = len(signal)
    interpolated_signal = signal.copy()

    # Handle first samples if flagged
    if first_samples_bad and n_samples > n_prepend:
        mean_rest = safe_nanmean(interpolated_signal[n_prepend:])
        interpolated_signal[:n_prepend] = mean_rest

    # Find zeros to interpolate
    zero_indices = np.where(interpolated_signal == 0)[0]

    all_indices = np.arange(n_samples)
    valid_indices = np.setdiff1d(all_indices, zero_indices)
    valid_values = interpolated_signal[valid_indices]

    # Edge case: all zeros or empty
    if len(valid_indices) == 0:
        mean_signal = safe_nanmean(signal)
        if np.isnan(mean_signal):
            mean_signal = 0
        interpolated_signal[:] = mean_signal
        return interpolated_signal

    # Linear interpolation over zeros
    if len(valid_indices) > 1:
        interp_func = interp1d(valid_indices, valid_values, kind='linear',
                               bounds_error=False, fill_value='extrapolate')
        interpolated_signal[zero_indices] = interp_func(zero_indices)
    else:
        # Only one valid value → fill zeros with that value
        interpolated_signal[zero_indices] = valid_values[0]

    # Reapply first sample correction if needed
    if first_samples_bad and n_samples > n_prepend:
        interpolated_signal[:n_prepend] = safe_nanmean(interpolated_signal[n_prepend:])

    return interpolated_signal



def correct_for_distance(pupil_diameter_mm, measured_distance_cm, reference_distance_cm=70):
    """
    Correct pupil size for distance from screen.
    
    Parameters
    ----------
    pupil_diameter_mm : np.ndarray
        Original pupil diameter
    measured_distance_cm : float
        Distance at which pupil was measured
    reference_distance_cm : float
        Distance to which we want to normalize pupil size
    
    Returns
    -------
    corrected_diameter : np.ndarray
        Pupil diameter corrected for distance
    """
    pupil_diameter_mm = np.asarray(pupil_diameter_mm)
    corrected_diameter = pupil_diameter_mm * (reference_distance_cm / measured_distance_cm)
    return corrected_diameter


def safe_nanmean(array, axis=None, fill=np.nan):
    """
    Compute mean ignoring NaNs. Returns 'fill' if the array is empty or all NaNs.
    
    Parameters
    ----------
    array : np.ndarray
        Input array
    axis : int or None
        Axis to compute mean along
    fill : float
        Value to return if all NaNs or empty
    
    Returns
    -------
    mean_result : float or np.ndarray
    """
    array = np.asarray(array)
    if array.size == 0 or np.all(np.isnan(array)):
        if axis == 0 and array.ndim > 1:
            return np.full(array.shape[1], fill)
        else:
            return fill
    else:
        return np.nanmean(array, axis=axis)



# =========================
# Interpolation
# =========================


def pupil_interpolation(df, delta_slope=0.03, n_passes=120, reference_distance_cm=70):
    """
    Preprocess a pupil dataframe: detect bad data by slope, interpolate, and correct for distance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        - "Pupil diameter left [mm]"
        - "Pupil diameter right [mm]"
    delta_slope : float
        Slope threshold for bad data detection
    n_passes : int
        Number of passes for slope detection
    measured_distance_cm : float
        Original distance from participant to screen
    reference_distance_cm : float
        Distance to normalize pupil size to
    
    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with new columns:
        - "Pupil diameter left corrected [mm]"
        - "Pupil diameter right corrected [mm]"
        - "Pupil diameter combined [mm]"
        - "baddata_left" (boolean mask)
        - "baddata_right" (boolean mask)
    """
    df_out = df.copy()
    
    # --- eye distance ----
    eye_z_left_mm = pd.to_numeric(
        df["Eye position left Z [DACS mm]"],
        errors="coerce"
    ).values
    
    eye_z_right_mm = pd.to_numeric(
        df["Eye position right Z [DACS mm]"],
        errors="coerce"
    ).values
    
    eye_z_left_mm = interpolate_signal(
        pd.to_numeric(
            df["Eye position left Z [DACS mm]"],
            errors="coerce"
        ).values
    )

    eye_z_right_mm = interpolate_signal(
        pd.to_numeric(
            df["Eye position right Z [DACS mm]"],
            errors="coerce"
        ).values
    )


    # --- Left eye ---
    pupil_left = pd.to_numeric(df_out["Pupil diameter left [mm]"], errors="coerce").values
    pupil_left_clean, baddata_left = detect_bad_data_by_slope(
        pupil_left, slope_threshold=delta_slope, n_passes=n_passes
    )
    pupil_left_interp = interpolate_signal(pupil_left_clean, first_samples_bad=True, n_prepend=n_passes)
    pupil_left_corrected = correct_for_distance(
        pupil_left_interp, measured_distance_cm=eye_z_left_mm / 10.0, reference_distance_cm=reference_distance_cm
    )
    
    # --- Right eye ---
    pupil_right = pd.to_numeric(df_out["Pupil diameter right [mm]"], errors="coerce").values
    pupil_right_clean, baddata_right = detect_bad_data_by_slope(
        pupil_right, slope_threshold=delta_slope, n_passes=n_passes
    )
    pupil_right_interp = interpolate_signal(pupil_right_clean, first_samples_bad=True, n_prepend=n_passes)
    pupil_right_corrected = correct_for_distance(
        pupil_right_interp, measured_distance_cm=eye_z_right_mm / 10.0, reference_distance_cm=reference_distance_cm
    )
    
    # --- Combine left and right eyes safely ---
    pupil_combined = safe_nanmean(np.vstack([pupil_left_corrected, pupil_right_corrected]), axis=0)
    
    # --- Add results to dataframe ---
    df_out["Pupil diameter left corrected [mm]"] = pupil_left_corrected
    df_out["Pupil diameter right corrected [mm]"] = pupil_right_corrected
    df_out["Pupil diameter combined [mm]"] = pupil_combined
    df_out["baddata_left"] = baddata_left
    df_out["baddata_right"] = baddata_right
    
    return df_out

def gaze_interpolation(df, n_prepend=120):
    """
    Interpolate gaze signals (left/right, X/Y) over zeros, independently of pupil preprocessing.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain gaze columns:
        - "Gaze point left X [DACS mm]"
        - "Gaze point left Y [DACS mm]"
        - "Gaze point right X [DACS mm]"
        - "Gaze point right Y [DACS mm]"
    n_prepend : int
        Number of first samples to replace if zeros

    Returns
    -------
    df_out : pd.DataFrame
        Same dataframe with added columns:
        - "Gaze left X corrected"
        - "Gaze left Y corrected"
        - "Gaze right X corrected"
        - "Gaze right Y corrected"
    """
    df_out = df.copy()

    gaze_columns = [
        ("Gaze point left X [DACS mm]", "Gaze left X corrected"),
        ("Gaze point left Y [DACS mm]", "Gaze left Y corrected"),
        ("Gaze point right X [DACS mm]", "Gaze right X corrected"),
        ("Gaze point right Y [DACS mm]", "Gaze right Y corrected"),
    ]

    for raw_col, corrected_col in gaze_columns:
        gaze_signal = pd.to_numeric(df_out[raw_col], errors="coerce").values
        
        # first_samples_bad if first n_prepend points are zeros
        first_samples_bad = np.any(gaze_signal[:n_prepend] == 0)
        
        gaze_interp = interpolate_signal(
            gaze_signal,
            first_samples_bad=first_samples_bad,
            n_prepend=n_prepend
        )
        df_out[corrected_col] = gaze_interp

    return df_out



# =========================
# Gaze correction
# =========================

def pupil_gaze_residuals(df, pupil_eye="combined"):
    """
    Compute pupil residuals after regressing out gaze positions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - Pupil columns (corrected)
        - Gaze columns (corrected)
        - 'Recording Time stamp' and 'Event'
    pupil_eye : str
        'left', 'right', or 'combined'

    Returns
    -------
    df_resid : pd.DataFrame
        Columns: 'Recording Time stamp', 'Event', 'Pupil dilation residual'
    """
    # Select pupil signal and corresponding gaze
    if pupil_eye == "left":
        pupil_signal = pd.to_numeric(df["Pupil diameter left corrected [mm]"], errors="coerce").values
        gaze_X = pd.to_numeric(df["Gaze left X corrected"], errors="coerce").values
        gaze_Y = pd.to_numeric(df["Gaze left Y corrected"], errors="coerce").values
    elif pupil_eye == "right":
        pupil_signal = pd.to_numeric(df["Pupil diameter right corrected [mm]"], errors="coerce").values
        gaze_X = pd.to_numeric(df["Gaze right X corrected"], errors="coerce").values
        gaze_Y = pd.to_numeric(df["Gaze right Y corrected"], errors="coerce").values
    else:  # combined
        pupil_signal = pd.to_numeric(df["Pupil diameter combined [mm]"], errors="coerce").values
        gaze_X_left = pd.to_numeric(df["Gaze left X corrected"], errors="coerce").values
        gaze_Y_left = pd.to_numeric(df["Gaze left Y corrected"], errors="coerce").values
        gaze_X_right = pd.to_numeric(df["Gaze right X corrected"], errors="coerce").values
        gaze_Y_right = pd.to_numeric(df["Gaze right Y corrected"], errors="coerce").values
        gaze_X = safe_nanmean(np.vstack([gaze_X_left, gaze_X_right]), axis=0)
        gaze_Y = safe_nanmean(np.vstack([gaze_Y_left, gaze_Y_right]), axis=0)

    # Prepare regression
    X = np.column_stack([gaze_X, gaze_Y])
    valid_mask = ~np.isnan(pupil_signal) & ~np.isnan(X).any(axis=1)

    residuals = np.full_like(pupil_signal, np.nan, dtype=float)
    if np.sum(valid_mask) > 0:
        reg = LinearRegression().fit(X[valid_mask], pupil_signal[valid_mask])
        predicted = reg.predict(X[valid_mask])
        residuals[valid_mask] = pupil_signal[valid_mask] - predicted

    # Build output DataFrame
    df_resid = pd.DataFrame({
        "Timestamp [μs]": df["Timestamp [μs]"],
        "Event": df["Event"],
        "Pupil residual": residuals
    })

    return df_resid


