#Final Project DS5110: Fall 2025
#Liz Mutina
#Madi Augustine
# Factor analysis 

# CODE FILE: 2 OF 4

import pandas as pd
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from semopy import Model, calc_stats
from semopy.inspector import inspect

"""
Referenced:
https://factor-analyzer.readthedocs.io/en/latest/index.html
https://pypi.org/project/factor-analyzer/
https://semopy.com/
https://pypi.org/project/semopy/
"""

#=================================================
#PART 1: Read in file, check if fit for analysis
#-------------------------------------------------

all_data_epsi = pd.read_csv("Data/np_all_data_csv_epsi.csv")

# KMO Test (ideal >= 0.6, adequate data)
kmo_all, kmo_model = calculate_kmo(all_data_epsi)

# Bartlett’s Test (ideal < 0.05, variables correlated sufficiently)
chi_square_value, p_value = calculate_bartlett_sphericity(all_data_epsi)

print(f"\n PART I:\n KMO Measure: {kmo_model:.3f}, goal is >= 0.6")
print(f"Bartlett’s test: Chi-square = {chi_square_value:.2f}, p-value = {p_value:.3e}")

#==============================================================
#PART II: Loadings using factor analyzer
#--------------------------------------------------------------

epsi_vars = [col for col in all_data_epsi.columns if col.startswith("epsi")] #variable columns ONLY

just_epsi = [v for v in epsi_vars]
data_epsi_mod_vars = all_data_epsi[just_epsi]

fa = FactorAnalyzer(n_factors=3, rotation="varimax") # factors updated based on scree
fa.fit(data_epsi_mod_vars)

#============================================================
 #Get eigenvalues and variance explained
#------------------------------------------------------------

ev, v = fa.get_eigenvalues()
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ev) + 1), ev, marker='o')
plt.title("Scree Plot for EPSi Data")
plt.xlabel("Factor")
plt.ylabel("Eigenvalue")
plt.axhline(3, color='r')  # 'elbow' of plot where intersects
plt.grid(True)
plt.show()

loadings = pd.DataFrame(
    fa.loadings_,
    index=data_epsi_mod_vars.columns,
    columns=[f"Factor{i+1}" for i in range(fa.n_factors)]
)

#============================================================
#PART III:low and cross loading variables:
#--------------------------------------------------------------

low_loading_vars = loadings[(loadings.abs() < 0.4).all(axis=1)].index.tolist()
cross_loading_vars = loadings[(loadings.abs() >= 0.4).sum(axis=1) > 1].index.tolist()

print("\n ALL loadings- before cleaning:\n",(loadings.to_string())) # ALL loading values will print (pre-cleaning)
print("\n PART III: \nLow-loading variables (<0.4 on all factors):\n",(low_loading_vars))
print("\nCross-loading variables (≥0.4 on more than one factor):\n", (cross_loading_vars))

# ============================================================
 # to check stats before cleaning (optional):
#--------------------------------------------------------------

variance1 = pd.DataFrame({
    "SS Loadings": fa.get_factor_variance()[0], # Eigenvalue for factor variance only
    "Proportion Var": fa.get_factor_variance()[1], # percentage of variance individually
    "Cumulative Var": fa.get_factor_variance()[2] # variance explained with addition of each factor
}, index=[f"Factor{i+1}" for i in range(fa.n_factors)])

print("\nVariance Explained by Each Factor_:")
print(variance1.round(3))

# =========================CLEANING============================
# PART IV:  Cleaned Factor Model:
#--------------------------------------------------------------

clean_vars = [v for v in epsi_vars if v not in low_loading_vars + cross_loading_vars]
data_clean = all_data_epsi[clean_vars]

fa_clean = FactorAnalyzer(n_factors=3, rotation="varimax") # set n_factors based off scree plot
fa_clean.fit(data_clean)

loadings_clean = pd.DataFrame(
    fa_clean.loadings_,
    index=data_clean.columns,
    columns=[f"Factor{i+1}" for i in range(fa_clean.n_factors)]) # creates working dataframe of cleaned loading values

print("\nPART IV: \n All loadings CLEANED:\n",(loadings_clean.to_string())) #uncomment to have ALL loading values print

# ============================================================
 #Variance and fit Info
#--------------------------------------------------------------

variance = pd.DataFrame({
    "SS Loadings": fa_clean.get_factor_variance()[0],
    "Proportion Var": fa_clean.get_factor_variance()[1],
    "Cumulative Var": fa_clean.get_factor_variance()[2]
}, index=[f"Factor{i+1}" for i in range(fa_clean.n_factors)])

print("\nVariance Explained by Each Factor:")
print(variance.round(3))

#=============================================================
# Model based on factor loadings
#--------------------------------------------------------------
 #(optional to save csv of loadings)
###np_all_epsi_loadings = loadings_clean.to_csv("Data/np_all_epsi_loadings.csv")

##original_model = """
##F1 =~ epsi_voice_about_me+epsi_hear_speaking+epsi_hear_voices_alone+epsi_voice_uncertain_real+epsi_thought_voice_real+epsi_more_one_voice+epsi_voice_clear+epsi_flashes_flames+epsi_saw_unsure_real+epsi_people_animals+epsi_felt_touching
##
##F2 =~ epsi_energy_trouble+epsi_energy_control+epsi_active_trouble+epsi_act_without_think+epsi_more_talkative+epsi_many_ideas+epsi_owed_money+epsi_spent_beyond_means+epsi_bought_expensive
##
##F3 =~ epsi_hear_my_thoughts+epsi_read_my_mind+epsi_read_others_minds+epsi_messages_things+epsi_gods_messenger+epsi_gods_work+epsi_thoughts_removed
####"""

np_point7_model = """

F1=~ epsi_hear_speaking+epsi_hear_voices_alone+epsi_voice_uncertain_real+epsi_thought_voice_real+epsi_more_one_voice+epsi_voice_about_me+epsi_voice_clear+epsi_people_animals+epsi_felt_touching

F2=~ epsi_energy_trouble+epsi_energy_control

F3 =~ epsi_read_others_minds+epsi_gods_messenger+epsi_famous_relationship+epsi_famous_romantic+epsi_romantic_messages
"""

###============GET=FIT=STATS======================
#build model as desribed
#---------------------------------------------------

model = Model(original_model)
model.load_dataset(data_clean)
model.fit()

stats = calc_stats(model)   
print(stats.to_string())
