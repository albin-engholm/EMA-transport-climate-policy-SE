import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the BEV_fleet.xlsx file as a pandas dataframe
bev_fleet_df = pd.read_excel("BEV_fleet.xlsx").transpose()

# Assign the values in the first row as the column headers
bev_fleet_df = bev_fleet_df.set_axis(bev_fleet_df.iloc[0], axis=1)

# Remove the first row
bev_fleet_df = bev_fleet_df.iloc[1:]

# Interpolate missing values
bev_fleet_i_df = bev_fleet_df.astype(float).interpolate(method="linear")


# Create a dictionary to map the column names to the line styles
linestyle_dict = {
    'Cars BEV share of vehicles sold': (1,1),
    'Cars VKT share electric': (1, 0),
    "LCV BEV share of vehicles sold": (1, 1),
    "LCV VKT share electric": (1, 0),
    "HGV BEV share of vehicles sold": (1, 1),
    "HGV VKT share electric": (1, 0),
}

# Create a dictionary to map the column names to the colors
palette_dict = {
    'Cars BEV share of vehicles sold': "blue",
    'Cars VKT share electric': "blue",
    "LCV BEV share of vehicles sold": "green",
    "LCV VKT share electric": "green",
    "HGV BEV share of vehicles sold": "red",
    "HGV VKT share electric": "red",
}

# Plot the data
sns.set_theme(style="darkgrid")
ax = sns.lineplot(data=bev_fleet_i_df, dashes=linestyle_dict, palette=palette_dict)

# Add vertical line at x=2040
ax.axvline(x=2040, color="black", linestyle="--")

# Show the plot
plt.legend(title="")
plt.show()
