# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:49:08 2025

@author: Liu
"""
# change dir
import os
os.chdir("D:/")

# import relevant libraries
import pandas as pd
import matplotlib.pyplot as plt

def read_large_tsv(file_path, chunk_size=100000):
    """Reads a large TSV file in chunks and processes it efficiently."""
    chunks = []
    
    # Read the file in chunks
    for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size, low_memory=False):
        chunks.append(chunk)  # Store chunk in a list (if you have enough memory)

    # Combine all chunks into a single DataFrame (optional)
    full_data = pd.concat(chunks, ignore_index=True)
    
    return full_data

# Replace with your file path
file_path = "pilot.tsv"

# Read the entire file
df = read_large_tsv(file_path)

# Display basic info
print(df.info())  # Check memory usage and structure
print(df.head())  # Preview first few rows

