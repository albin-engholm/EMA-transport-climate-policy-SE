# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:19:26 2023

@author: aengholm
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set global font size for Seaborn and Matplotlib
sns.set(font_scale=2)  # This scales the font size for Seaborn plots

# Create synthetic data using the gamma distribution to ensure positive values
np.random.seed(0)

policy_a = np.random.gamma(2, 2, 100)
policy_b = np.random.gamma(3, 3, 100)
policy_c = np.random.gamma(1.5, 1.5, 100)

# Compute the robustness metric for each policy
r_a = (np.mean(policy_a) + 1) * np.std(policy_a + 1)
r_b = (np.mean(policy_b) + 1) * np.std(policy_b + 1)
r_c = (np.mean(policy_c) + 1) * np.std(policy_c + 1)

metrics = [r_a, r_b, r_c]
policies = [policy_a, policy_b, policy_c]
labels = ["Policy A", "Policy B", "Policy C"]

# Sort the policies and metrics based on their metric values
sorted_indices = np.argsort(metrics)
sorted_policies = [policies[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]

# Create a color map: best policy in green, others in grey
colors = ['green'] + ['grey'] * (len(metrics) - 1)

fig, ax = plt.subplots(figsize=(10, 6))

for policy, color in zip(sorted_policies, colors):
    sns.kdeplot(policy, ax=ax, fill=True, color=color)

# # Now annotate the peak of each KDE with its label and metric
# for idx, collection in enumerate(ax.collections):
#     x = collection.get_paths()[0].vertices[:, 0]
#     y = collection.get_paths()[0].vertices[:, 1]
#     max_y = np.max(y)
#     max_x = x[np.argmax(y)]
#     label = f"{sorted_labels[idx]} (R={metrics[sorted_indices[idx]]:.2f})"
#     if sorted_labels[idx] == "Policy C":
#         annotation_color = 'green'
#         arrow_color = 'green'
#     else:
#         annotation_color = 'black'
#         arrow_color = 'black'
#     ax.annotate(label,
#                 xy=(max_x, max_y), xycoords='data',
#                 xytext=(40, -5), textcoords='offset points',
#                 arrowprops=dict(arrowstyle="-", lw=1.5, color=arrow_color),
#                 size=20, color=annotation_color)  # Font size for annotations

sns.despine()
ax.set_xlabel("Outcome $M_{i}$ (minimize)", fontsize=20)  # Font size for x label

# Remove y-label and tick marks
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_ylabel('')

# Set the background transparency for figure and axes
fig.patch.set_alpha(0.0)           # for figure
ax.patch.set_alpha(0.0)            # for axis

# Remove grid
ax.grid(False)

plt.tight_layout()
plt.show()

