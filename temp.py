# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# change dir
import os
os.chdir("D:/data_et/")

# import relevant libraries
import scipy.io
import pandas as pd
import numpy as np

def extract_mat_struct(mat_obj, parent_key="data"):
    """Extracts all nested MATLAB structs from a .mat file into a dictionary of DataFrames."""
    result = {}

    # Ensure we unwrap single-element arrays
    if isinstance(mat_obj, np.ndarray) and mat_obj.size == 1:
        mat_obj = mat_obj.item()

    # Check if it's a MATLAB struct (has dtype.names)
    if isinstance(mat_obj, np.ndarray) and mat_obj.dtype.names:
        for name in mat_obj.dtype.names:
            value = mat_obj[name]
            new_key = f"{parent_key}.{name}"  # Dot-separated key

            # If value is another struct, recursively extract it
            if isinstance(value, np.ndarray) and value.dtype.names:
                result.update(extract_mat_struct(value, new_key))
            else:
                # Convert to DataFrame if possible
                result[new_key] = pd.DataFrame(value) if isinstance(value, np.ndarray) else pd.DataFrame([value])

    return result

def find_struct_columns(df):
    """Finds columns in a DataFrame that contain MATLAB structs."""
    struct_cols = []
    for col in df.columns:
        if isinstance(df[col].iloc[0], np.ndarray) and df[col].iloc[0].dtype.names:
            struct_cols.append(col)
    return struct_cols

def unpack_struct_columns(df):
    """Recursively unpacks all struct-type columns within a DataFrame."""
    while True:
        struct_columns = find_struct_columns(df)
        if not struct_columns:
            break  # Stop when no struct columns remain

        for col in struct_columns:
            # Convert struct field into a DataFrame
            struct_data = pd.DataFrame.from_records(df[col].apply(lambda x: {name: x[name] for name in x.dtype.names}))

            # Rename columns to avoid conflicts
            struct_data.columns = [f"{col}.{subcol}" for subcol in struct_data.columns]

            # Drop the original struct column and merge expanded columns
            df = df.drop(columns=[col]).join(struct_data)

    return df

def fully_unpack_dataframes(extracted_dfs):
    """Recursively unpacks all structs inside extracted DataFrames."""
    for key in list(extracted_dfs.keys()):
        extracted_dfs[key] = unpack_struct_columns(extracted_dfs[key])

    return extracted_dfs

# Load .mat file
mat_data = scipy.io.loadmat("commonConfetti_00015_et_4.mat", struct_as_record=False, squeeze_me=True)

# Extract the "data" struct
if "data" in mat_data:
    data_struct = mat_data["data"]

    # Step 1: Extract all MATLAB structs into DataFrames
    extracted_dfs = extract_mat_struct(data_struct)

    # Step 2: Fully unpack struct columns inside each DataFrame
    extracted_dfs = fully_unpack_dataframes(extracted_dfs)

    # Print extracted DataFrames
    print("Extracted and Fully Unpacked DataFrames:", extracted_dfs.keys())

    # Example: Accessing a specific DataFrame
    df_example = extracted_dfs.get("data.field1.subfield1", None)
    if df_example is not None:
        print(df_example.head())
    else:
        print("Key not found in extracted data.")

else:
    print("The 'data' struct is not found in the .mat file.")
