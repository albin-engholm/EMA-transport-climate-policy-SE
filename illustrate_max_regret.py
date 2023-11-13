# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:52:07 2023

@author: aengholm
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulating the data: for each policy, we have a list of outcomes across different scenarios.
np.random.seed(7)
policy_a_outcomes = np.random.randn(100).cumsum() + 50
policy_b_outcomes = np.random.randn(100).cumsum() + 50.5
policy_c_outcomes = np.random.randn(100).cumsum() + 52

# Calculate the regret for each policy in each scenario
regret_a = policy_a_outcomes - np.minimum.reduce([policy_a_outcomes, policy_b_outcomes, policy_c_outcomes])
regret_b = policy_b_outcomes - np.minimum.reduce([policy_a_outcomes, policy_b_outcomes, policy_c_outcomes])
regret_c = policy_c_outcomes - np.minimum.reduce([policy_a_outcomes, policy_b_outcomes, policy_c_outcomes])

# Identify the scenario with the maximum regret for each policy
max_regret_scenario_a = np.argmax(regret_a)
max_regret_scenario_b = np.argmax(regret_b)
max_regret_scenario_c = np.argmax(regret_c)

fig, ax = plt.subplots(figsize=(10, 6))

# Make the background of the plot transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.grid(False)
# Plotting the outcomes for each policy with blue circles for each scenario
ax.plot(np.arange(len(policy_a_outcomes)), policy_a_outcomes, label='Policy A', color='grey')
ax.scatter(np.arange(len(policy_a_outcomes)), policy_a_outcomes, color='lightgrey', s=50, label=None)
ax.plot(np.arange(len(policy_b_outcomes)), policy_b_outcomes, label='Policy B', color='g')
ax.scatter(np.arange(len(policy_b_outcomes)), policy_b_outcomes, color='lightgreen', s=50, label=None)
ax.plot(np.arange(len(policy_c_outcomes)), policy_c_outcomes, label='Policy C', color='grey')
ax.scatter(np.arange(len(policy_c_outcomes)), policy_c_outcomes, color='lightgrey', s=50, label=None)

# Highlighting the maximum regret for each policy with a larger circle
ax.scatter(max_regret_scenario_a, policy_a_outcomes[max_regret_scenario_a], color='grey', s=100, zorder=5)
ax.scatter(max_regret_scenario_b, policy_b_outcomes[max_regret_scenario_b], color='g', s=100, zorder=5)
ax.scatter(max_regret_scenario_c, policy_c_outcomes[max_regret_scenario_c], color='grey', s=100, zorder=5)

# Determine the best-performing policy in the maximum regret scenario for each policy
best_outcome_at_max_regret_a = np.min([policy_a_outcomes[max_regret_scenario_a], policy_b_outcomes[max_regret_scenario_a], policy_c_outcomes[max_regret_scenario_a]])
best_outcome_at_max_regret_b = np.min([policy_a_outcomes[max_regret_scenario_b], policy_b_outcomes[max_regret_scenario_b], policy_c_outcomes[max_regret_scenario_b]])
best_outcome_at_max_regret_c = np.min([policy_a_outcomes[max_regret_scenario_c], policy_b_outcomes[max_regret_scenario_c], policy_c_outcomes[max_regret_scenario_c]])

# Draw vertical lines to visualize the regret
ax.vlines(x=max_regret_scenario_a, ymin=best_outcome_at_max_regret_a, ymax=policy_a_outcomes[max_regret_scenario_a], color='grey', linestyle='--', zorder=4)
ax.vlines(x=max_regret_scenario_b, ymin=best_outcome_at_max_regret_b, ymax=policy_b_outcomes[max_regret_scenario_b], color='g', linestyle='--', zorder=4)
ax.vlines(x=max_regret_scenario_c, ymin=best_outcome_at_max_regret_c, ymax=policy_c_outcomes[max_regret_scenario_c], color='grey', linestyle='--', zorder=4)

# Determine the midpoint of the regret for annotation positioning
midpoint_a = (policy_a_outcomes[max_regret_scenario_a] + best_outcome_at_max_regret_a) / 2
midpoint_b = (policy_b_outcomes[max_regret_scenario_b] + best_outcome_at_max_regret_b) / 2
midpoint_c = (policy_c_outcomes[max_regret_scenario_c] + best_outcome_at_max_regret_c) / 2

# Adding annotations with semi-transparent boxes
# bbox_props = dict(boxstyle="round,pad=0.0", facecolor="#F9F7F6", alpha=0.6, edgecolor="none")

# ax.annotate('Max Regret Policy A', (max_regret_scenario_a, midpoint_a), 
#             color='black',
#             textcoords="offset points", 
#             xytext=(5,0), 
#             ha='left', 
#             va='center',
#             fontsize=20,
#             bbox=bbox_props)

# ax.annotate('Max Regret  Policy B', (max_regret_scenario_b, midpoint_b), 
#             color='green',
#             textcoords="offset points", 
#             xytext=(5,0), 
#             ha='left', 
#             va='center',
#             fontsize=20,
#             bbox=bbox_props)

# ax.annotate('Max Regret Policy C', (max_regret_scenario_c, midpoint_c), 
#             color='black',
#             textcoords="offset points", 
#             xytext=(5,0), 
#             ha='left', 
#             va='center',
#             fontsize=18,
#             bbox=bbox_props)

# Set the color of the axes to black
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)


ax.set_xlabel('Scenario')
ax.set_ylabel('Outcome $M_i$ (minimize)')
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.tight_layout()
plt.show()
