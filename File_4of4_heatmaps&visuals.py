#Final Project DS5110: Fall 2025
#Liz Mutina
#Madi Augustine
# Generation of visuals (heatmaps)

# CODE FILE: 4 OF 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

"""
Referenced:
https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap
https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LinearSegmentedColormap.html
"""

epsi_loadings_df = pd.read_csv("Data/np_all_epsi_loadings.csv", index_col=0)
epsi_loadings_df = epsi_loadings_df.apply(pd.to_numeric, errors='coerce')
 
plt.figure(figsize=(10, 8))
sns.heatmap(epsi_loadings_df.abs(), cmap="Blues", annot=False)
plt.title("EPsi Factor Loadings (Cleaned)")
plt.xlabel("Factor")
plt.ylabel("Variables")
plt.tight_layout()
plt.show()

#===================================
#  colored yellow 0.7
#-----------------------------------

colors = ["#e0f2ff", "#7bbcff", "#00509e", "#ffea00"]  # last = yellow
cmap = LinearSegmentedColormap.from_list("highlight_high", colors, N=256)

plt.figure(figsize=(12, 10))
sns.heatmap(epsi_loadings_df.abs(), cmap=cmap, vmin=0, vmax=1, linewidths=0.5)
plt.title("Factor Loadings (> .70 highlighted)")
plt.show()

#==================================
# with values in cell
#----------------------------------

plt.figure(figsize=(12, 10))
sns.heatmap(
    epsi_loadings_df.abs(),
    cmap="Blues",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar=True
)
plt.title("Factor Loadings with Values")
plt.show()

#=========================================
# sorted highest to lowest - NOT used
# ----------------------------------------

sorted_loadings = epsi_loadings_df.abs().iloc[np.argsort(-epsi_loadings_df.abs().max(axis=1))]

plt.figure(figsize=(12, 11))
sns.heatmap(sorted_loadings, cmap="Blues", linewidths=0.5)
plt.title("Factor Loadings (Variables Sorted by Highest Loading)")
plt.show()

#===================================================
# categorical threshold bands
# ----------------------------------------

bounds = [0, .30, .50, .70, 1.0]
colors = ["#e0e0e0", "#90caf9", "#42a5f5", "#ffeb3b"]
cmap = LinearSegmentedColormap.from_list("banded", colors)

plt.figure(figsize=(12, 10))
sns.heatmap(epsi_loadings_df, cmap=cmap, linewidths=0.5)
plt.title("Factor Loadings Categorized by Strength")
plt.show()

#=========================================
# per factor loadings bar chart - NOT used
# ----------------------------------------

##factor = "Factor3"  # change factor #
##epsi_loadings_df[factor].sort_values().plot(kind="barh", figsize=(6, 10))
##plt.title(f"Loadings for {factor}")
##plt.xlabel("Loading")
##plt.show()

#========================================
# binary heatmap 
# ----------------------------------------

binary = (epsi_loadings_df.abs() > .70).astype(int)

plt.figure(figsize=(10, 10))
sns.heatmap(binary, cmap="Greys", linewidths=0.5, annot=True)
plt.title("High Loadings (Binary ≥ .70) Included in Final Items")
plt.show()


#===============================================
# LOADING DATA FROM PREVIOUS PROGRAM — DataFrame
#
# (must have index = variable names, columns = factor names)
# ---------------------------------------------

loadings = epsi_loadings_df.copy()  # rename for clarity


## Variables to annotate with values:

annot_vars = [
    "epsi_hear_speaking",
    "epsi_hear_voices_alone",
    "epsi_voice_uncertain_real",
    "epsi_thought_voice_real",
    "epsi_more_one_voice",
    "epsi_voice_about_me",
    "epsi_voice_clear",
    "epsi_people_animals",
    "epsi_felt_touching",
    "epsi_energy_trouble",
    "epsi_energy_control",
    "epsi_read_others_minds",
    "epsi_gods_messenger",
    "epsi_famous_relationship",
    "epsi_famous_romantic",
    "epsi_romantic_messages",
]

## Create annotation matrix: only values for selected variables
## ---------------------------------------------
annot_matrix = loadings.copy().astype(str)

for var in annot_matrix.index:
    for fac in annot_matrix.columns:
        if var in annot_vars:
            annot_matrix.loc[var, fac] = f"{loadings.loc[var, fac]:.2f}"
        else:
            annot_matrix.loc[var, fac] = ""


# Custom colormap with yellow for high loadings
# ---------------------------------------------

colors = [
    (0.90, 0.95, 1.0),   # light blue
    (0.40, 0.60, 0.90),  # medium blue
    (0.00, 1.0, 0.8),  # I put  a 1 here and it looks sweet!
    (1.0, 1.0, 0.00),  # bright yellow for >0.7
]

cmap = LinearSegmentedColormap.from_list("custom_map", colors, N=256)


# Plot
# ---------------------------------------------
plt.figure(figsize=(10, 14))
sns.heatmap(
    loadings.astype(float),
    cmap=cmap,
    annot=annot_matrix,
    fmt="",
    cbar=True,
    linewidths=0.4,
    linecolor="gray",
    vmin=0,
    vmax=1,
)

plt.title("Factor Loadings Heatmap (Final Model Items Annotated)", fontsize=16)
plt.ylabel("Items")
plt.xlabel("Factors")
plt.tight_layout()
plt.show()

