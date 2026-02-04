# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 11:03:43 2026

@author: bbf2518
"""

import os
import traceback
from pathlib import Path
from tobii_preprocess.step_1_cleanup import (
    extract_relevant_rows_1,
    extract_relevant_rows_2,
    remove_empty_columns_tobii
)

# ---- configure raw data folder ----
#raw_folder = Path(r"Z:\pb\EPSY\EPSY-Allgemein\Forschung\DynamicBeliefUpdating_Children (Harry und Nina)\03_Data\Tobii_Data_tsv")  # your raw TSV folder
raw_folder = Path(r"Z:\pb\EPSY\EPSY-Allgemein\Forschung\DynamicBeliefUpdating_Children (Harry und Nina)\04_Analysis\pupil_analysis\raw_tsv")  # your raw TSV folder
processed_folder = Path(r"Z:\pb\EPSY\EPSY-Allgemein\Forschung\DynamicBeliefUpdating_Children (Harry und Nina)\04_Analysis\pupil_analysis\preprocessed")  # base folder for outputs
processed_folder.mkdir(exist_ok=True)

# ---- get all TSV files in the raw folder ----
tsv_files = sorted(raw_folder.glob("*.tsv"))

# ---- loop over files for cleaning up ----
for file_path in tsv_files:
    print(f"\nProcessing {file_path.name} ...")
    try:
        # ---- Step 1 ----
        step1_output = remove_empty_columns_tobii(file_path, output_dir=processed_folder)

        # ---- Step 2 ----
        step2_output = extract_relevant_rows_1(step1_output, output_dir=processed_folder)

        # ---- Step 3 ----
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

# ---- loop over files for down smapling ----
