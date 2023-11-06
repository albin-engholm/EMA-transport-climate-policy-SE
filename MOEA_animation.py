# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:23:09 2023

@author: aengholm
"""

from ema_workbench.em_framework.optimization import (HypervolumeMetric,
                                                    EpsilonProgress,
                                                    ArchiveLogger,epsilon_nondominated)
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load the data
file_str="All levers test5"
archives = ArchiveLogger.load_archives("./archives_animation/"+file_str+".tar.gz")

# Filter out empty dataframes
archives = {k: v for k, v in archives.items() if not v.empty}

# Sort the dataframes and remove the first which might be empty
dataframes = [df for _, df in sorted(archives.items())]

# Calculate global min and max for the axes and the color variable
x_min = min(df['M1_CO2_TTW_total'].min() for df in dataframes)
x_max = max(df['M1_CO2_TTW_total'].max() for df in dataframes)
y_min = min(df['M4_energy_use_bio'].min() for df in dataframes)
y_max = max(df['M4_energy_use_bio'].max() for df in dataframes)
c_min = min(df['M2_driving_cost_car'].min() for df in dataframes)
c_max = max(df['M2_driving_cost_car'].max() for df in dataframes)

fig, ax = plt.subplots()

# Prepare a scatter plot that will be updated at each frame.
# Initialize with empty data, and map colors to M2 outcome
scatter = ax.scatter([], [], c=[], s=15, cmap='viridis', vmin=c_min, vmax=c_max)

# Add a color bar
cbar = plt.colorbar(scatter, ax=ax, label="M2_driving_cost_car")

# Set the labels for your plot
ax.set_xlabel("M1_CO2_TTW_total")
ax.set_ylabel("M4_energy_use_bio")

# Set the limits for the x and y axes
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

def animate(i):
    # Update the data for the scatter plot.
    scatter.set_offsets(dataframes[i][["M1_CO2_TTW_total", "M4_energy_use_bio"]].values)
    # Set color according to M2 outcome
    scatter.set_array(dataframes[i]["M2_driving_cost_car"].values)
    ax.set_title(f'Number of function evaluations: {sorted(archives.keys())[i]}')

ani = animation.FuncAnimation(fig, animate, frames=len(dataframes), interval=200, repeat=True)

# Save the animation
ani.save('animation'+file_str+'.gif', writer='imagemagick', fps=10, dpi=150)


