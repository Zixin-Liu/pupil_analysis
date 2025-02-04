# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:34:01 2025

@author: Liu
"""

# change dir
import os
os.chdir("D:/")

import pandas as pd
import matplotlib.pyplot as plt

# File path
file_path = "pilot.tsv"

# Define the participant to filter
participant_name = "commonConfetti_00015_et_4"

# Initialize lists to store filtered data
timestamps = []
gaze_x = []
gaze_y = []

# Read the file in chunks
chunk_size = 50000  # Adjust based on your system capacity
for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size, encoding="utf-8", low_memory=False):
    # Keep only relevant columns (saves memory)
    chunk = chunk[["Participant name", "Recording timestamp", "Gaze point left X", "Gaze point left Y"]]
    
   
    
    # Filter for the specific participant
    chunk = chunk[chunk["Participant name"] == participant_name]
    
    # Debugs 
    print(chunk.columns)  # Debug: See actual column names
    print(chunk["Participant name"].unique())  # Debug: See all participant names
    print(chunk.head())  # Debug: Check if data exists
    print("Before filtering:", len(chunk))  # Debug: Total rows in chunk
    
   
    # Drop rows where either gaze point is missing
    chunk = chunk.dropna(subset=["Gaze point left X", "Gaze point left Y"])

    # Convert timestamp to numeric
    chunk["Recording timestamp"] = pd.to_numeric(chunk["Recording timestamp"], errors='coerce')
    
    # Debug
    print(chunk.head())  # Debug: Check cleaned data
    print("After dropping NaN:", len(chunk))  # Debug: Rows after cleaning
    break  # Stop after first chunk

    # Append filtered data to lists
    timestamps.extend(chunk["Recording timestamp"].tolist())
    gaze_x.extend(chunk["Gaze point left X"].tolist())
    gaze_y.extend(chunk["Gaze point left Y"].tolist())

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(timestamps, gaze_x, label="Fixation Point X", alpha=0.7)
plt.plot(timestamps, gaze_y, label="Fixation Point Y", alpha=0.7)

plt.xlabel("Recording Timestamp")
plt.ylabel("Fixation Coordinates")
plt.title(f"Fixation Point Data for {participant_name}")
plt.legend()
plt.grid()
plt.show()
