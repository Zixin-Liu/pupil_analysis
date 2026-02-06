# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 11:03:43 2026

@author: bbf2518
"""

import os
import traceback
import pandas as pd
from pathlib import Path
from tobii_preprocess.step_1_cleanup import (
    extract_relevant_rows_1,
    extract_relevant_rows_2,
    remove_empty_columns_tobii
)
from tobii_preprocess.step_2_interpolation import ( 
    pupil_interpolation, 
    gaze_interpolation, 
    pupil_gaze_residuals
)

# ---- configure raw data folder ----
#raw_folder = Path(r"Z:\pb\EPSY\EPSY-Allgemein\Forschung\DynamicBeliefUpdating_Children (Harry und Nina)\03_Data\Tobii_Data_tsv")  # your raw TSV folder
# Base output directory
base_output_dir = Path(
    r"Z:\pb\EPSY\EPSY-Allgemein\Forschung\DynamicBeliefUpdating_Children (Harry und Nina)\04_Analysis\pupil_analysis"
)
# test raw TSV folder (inside the base output directory)
raw_folder = base_output_dir / "raw_tsv"

processed_folder = base_output_dir /"preprocessed"  # base folder for First step outputs
processed_folder.mkdir(exist_ok=True)

# ---- get all TSV files in the raw folder ----
tsv_files = sorted(raw_folder.glob("*.tsv"))

# ---- loop over files for cleaning up ----
for file_path in tsv_files:
    print(f"\nProcessing {file_path.name} ...")
    try:
        # ---- 1. remove completely empty columns + decimal symbol swap ----
        step1_output = remove_empty_columns_tobii(file_path, output_dir=processed_folder)

        # ---- 2. Keep relevant rows ----
        step2_output = extract_relevant_rows_1(step1_output, output_dir=processed_folder)

        # ---- 3. Further tidy up the files ----
        # This function returns two lists: baselines and blocks
        baseline_files, block_files = extract_relevant_rows_2(step2_output, output_dir=processed_folder.parent)

        
        # delete Step 1 output
        if step1_output.exists():
            os.remove(step1_output)

       
        # delete Step 2 output
        if step2_output.exists():
            os.remove(step2_output)
            
        print(f"Finished processing {file_path.name}")
        

    except Exception as e:
        print(f"⚠️ Error processing {file_path.name}: {e}")
        traceback.print_exc()
        print("Skipping to next file...")
        continue

# ---- loop over files for interpolation ----

delta = 0.03        # slope threshold for blink detection, equivalent to 25 pixel change
n_interps = 120       # number of passes
reference_distance_cm = 70.0  # reference distance for correction


# ---- Step 4: Interpolate pupil + gaze ----
for file_path in Path(base_output_dir).glob("preprocessed/*/*.tsv"):
     print(f"\nProcessing {file_path.name} ...")
     try:
         # Load TSV
         df = pd.read_csv(file_path, sep="\t")
 
         # 1: Pupil interpolation
         df_pupil = pupil_interpolation(
             df,
             delta_slope=delta,
             n_passes=n_interps,
             reference_distance_cm=reference_distance_cm
         )
 
         # 2: Gaze interpolation
         df_gaze = gaze_interpolation(
             df,
             n_prepend=n_interps
         )
 
         # Merge pupil + gaze interpolated data
         df_merged = pd.DataFrame({
             "Timestamp [μs]": df["Recording timestamp [μs]"],
             # Pupil
             "Pupil diameter left corrected [mm]": df_pupil["Pupil diameter left corrected [mm]"],
             "Pupil diameter right corrected [mm]": df_pupil["Pupil diameter right corrected [mm]"],
             "Pupil diameter combined [mm]": df_pupil["Pupil diameter combined [mm]"],
             "baddata_left": df_pupil["baddata_left"].astype(int),
             "baddata_right": df_pupil["baddata_right"].astype(int),
             # Gaze
             "Gaze left X corrected": df_gaze["Gaze left X corrected"],
             "Gaze left Y corrected": df_gaze["Gaze left Y corrected"],
             "Gaze right X corrected": df_gaze["Gaze right X corrected"],
             "Gaze right Y corrected": df_gaze["Gaze right Y corrected"],
         })
 
         # Save interpolated TSV
         participant_id = file_path.parent.name
         out_dir_interp = Path(base_output_dir) / "interpolated" / participant_id
         out_dir_interp.mkdir(parents=True, exist_ok=True)
 
         out_file_interp = out_dir_interp / f"{file_path.stem}_interpolated.tsv"
         df_merged.to_csv(out_file_interp, sep="\t", index=False)
         print(f"Saved interpolated pupil + gaze: {out_file_interp}")
 
     except Exception as e:
         print(f"Error processing {file_path}: {e}")


# ---- Step 5: Compute pupil residuals from interpolated files ----
for interp_file in Path(base_output_dir).glob("interpolated/*/*.tsv"):
    print(f"\nComputing residuals for {interp_file.name} ...")
    try:
        # Load interpolated TSV
        df_interp = pd.read_csv(interp_file, sep="\t")

        # Compute residuals using the logic function
        df_resid = pupil_gaze_residuals(df_interp, pupil_eye="combined")

        # Save residual TSV
        participant_id = interp_file.parent.name
        out_dir_resid = Path(base_output_dir) / "residual" / participant_id
        out_dir_resid.mkdir(parents=True, exist_ok=True)

        out_file_resid = out_dir_resid / f"{interp_file.stem}_residual.tsv"
        df_resid.to_csv(out_file_resid, sep="\t", index=False)
        print(f"Saved pupil residuals: {out_file_resid}")

    except Exception as e:
        print(f"Error processing {interp_file}: {e}")

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            