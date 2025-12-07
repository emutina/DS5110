#Final Project DS5110: Fall 2025
#Liz Mutina
#Madi Augustine
# Data file preparation

# CODE FILE: 1 OF 4

import pandas as pd
import numpy as np

"""
The purpose of this code file is to separately prepare the csv file by
replacing any NaN cells with participant responses to the same question during
the completion of their second EPSi screener
(T1 responses were collected 3-7 days prior to T2)

NOTE: This file is for informational purposes.
The code does not need to be run again as the data file was adequately prepared
via this program and the resulting file was used throughout the analyses

Referenced:
https://pytutorial.com/python-pandas-set_index-set-dataframe-index/
"""

# ============================================================
# Data prep to ensure no missing values
#-------------------------------------------------------------
t1 = pd.read_csv("Data/epsi_t1_all.csv")
t2 = pd.read_csv("Data/t2_EPSI_data.csv")

# Identify EPSI variables
epsi_vars = [col for col in t1.columns if col.startswith("epsi")]

# Replace missing t1 values with t2 equivalents (match by ID)
t1.set_index("record_id", inplace=True)
t2.set_index("record_id", inplace=True)

# Define matching variable pairs (t1 to t2)
t1_epsi_vars = [col for col in t1.columns if col.startswith("epsi")]

# Fill missing t1 responses using matching t2 variables 
for t1_col in t1_epsi_vars:
    t2_col = t1_col.replace("", "t2_")
    if t2_col in t2.columns:
        t1[t1_col] = t1[t1_col].fillna(t2[t2_col])
    else:
        print(f"! No matching column for {t1_col} in t2 (skipped)")
print("* Finished filling t1 missing values with t2 responses where available.")

# Check how many replacements were made
missing_after = t1[epsi_vars].isna().sum().sum()
print(f"Remaining missing EPSI responses after replacement: {missing_after}")

# Create the working EPSI dataframe
data_epsi = t1[epsi_vars].copy()
all_data_csv_epsi = data_epsi.to_csv("Data/all_data_csv_epsi.csv")
