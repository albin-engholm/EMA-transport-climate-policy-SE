# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:02:28 2024
Simple visualization of the challenge with curse of dimensionality for scenario forecasting
@author: aengholm
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
x = np.linspace(0.1, 0.9, 9*10).round(decimals=2)

results = pd.DataFrame()
for n in np.linspace(5, 20, 4):
    for p in x:
        results.loc[n, p] = p**n

# Resetting index and melting the DataFrame
results_reset = results.reset_index().rename(columns={'index': 'n'})
long_format = results_reset.melt(id_vars='n', var_name='p', value_name='P(all correct)')

# Plotting with Seaborn
sns.set_style("whitegrid")
sns.lineplot(data=long_format, x='p', y='P(all correct)', hue='n', palette='viridis')

# Show the plot
plt.show()
