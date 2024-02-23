# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:18:42 2023
Script for performing uncertainty and robustness analysis.
@author: aengholm
"""
import math
import pickle
import statistics
import pandas as pd
import numpy as np
import re
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
import matplotlib.patches as patches
from ema_workbench.analysis import feature_scoring

from ema_workbench.analysis import prim
from matplotlib.patches import Patch
from ema_workbench.analysis import parcoords
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")

policy_types = ["All levers", "No transport efficiency"]
# policy_types=["All levers"]
# ,"No transport efficiency"]
load_results = 1
load_results = 1
if load_results == 1:
    from ema_workbench import load_results
    date = "2024-02-23"
    n_scenarios = 105
    # for policy_type in policy_types:
    t1 = f"./output_data/robustness_analysis_results/X_XP{n_scenarios}_scenarios_MORDM_OE_{date}.p"

    data = pickle.load(open(t1, "rb"))
    if len(data) > 1:
        data_x = data[0][0]  # Load output data for run for parameter set X
        data_xp = data[0][1]  # Load output data for run for parameter set XP

        experiments_x = data_x[1]
        outcomes_x = data_x[0]
        experiments_xp = data_xp[1]
        outcomes_xp = data_xp[0]
        df_outcomes = pd.concat([pd.DataFrame(outcomes_x), pd.DataFrame(outcomes_xp)],
                                axis=0, join="inner").reset_index(drop=True)
        experiments = pd.concat([experiments_x, experiments_xp], axis=0, join="outer").reset_index(drop=True)
    else:
        data_xp = data[0][0]  # Load output data for run for parameter set XP

        experiments_xp = data_xp[1]
        outcomes_xp = data_xp[0]
        df_outcomes = pd.DataFrame(outcomes_xp).reset_index(drop=True)
        experiments = experiments_xp
    # experiments, outcomes = pickle.load(open(t1, "rb"))
    t2 = f"./output_data/robustness_analysis_results/{n_scenarios}_scenarios_MORDM_OE_{date}model_.p"
    model = pickle.load(open(t2, "rb"))

    # DF with both experiments and outcomes
    df_full = pd.concat([experiments, df_outcomes], axis=1, join="inner")
    df_outcomes['policy'] = experiments["policy"]
    for index, row in df_full.iterrows():
        if math.isnan(row["R1_fuel_price_to_car_electrification"]):
            df_full.loc[index, "Uncertainty set"] = "X"
        else:
            df_full.loc[index, "Uncertainty set"] = "XP"
    df_full = df_full[df_full["policy"] != "Reference policy"]
    df_full["CO2 target not met"] = df_full['M1_CO2_TTW_total'] > 1.89
    df_full["Bio share of fuel energy"] = df_full["M4_energy_use_bio"] / \
        (df_full["Energy total"]-df_full["M5_energy_use_electricity"])
    # Create helper lists of outcomes, levers and uncertainties
    # Different outcome sets
    all_outcomes = model.outcomes.keys()
    key_outcomes = [
        'M1_CO2_TTW_total',
        'M2_driving_cost_car',
        'M3_driving_cost_truck',
        'M4_energy_use_bio',
        'M5_energy_use_electricity']
    objective_outcomes = [
        'M2_driving_cost_car',
        'M3_driving_cost_truck',
        'M4_energy_use_bio',
        'M5_energy_use_electricity']

    # Levers
    levers = model.levers.keys()

    # Uncertainties
    uncertainties = model.uncertainties.keys()

    # Define color coding mapping
    color_coding = {
        "All levers": '#0005CC',
        "No transport efficiency": '#05CC00',
        "STA": '#CC0005'}
    light_color_coding = {
        "All levers": '#6064CC',
        "No transport efficiency": '#7ECC60',
        "STA": '#C77F5D'
    }

    ultra_light_color_coding = {
        "All levers": '#EDEDFF',
        "No transport efficiency": '#EDFFED',
        "STA": '#FFEDED'
    }
    df_full_sta = df_full[df_full["Policy type"] == "STA"]
    df_sta_levers = df_full_sta[list(levers)+["policy"]].drop_duplicates()
# %% Visualization of reference scenario performance
df_reference_subset = df_full[(df_full["scenario"] == "Reference") & (df_full["Uncertainty set"] == "XP")]

limits_outcomes = pd.DataFrame()  # Create a dataframe for lever-based limits
for item in objective_outcomes:
    limits_outcomes.loc[0, item] = min(df_reference_subset[item])  # Get lower bound
    limits_outcomes.loc[1, item] = max(df_reference_subset[item])  # Get upper bound
paraxes = parcoords.ParallelAxes(limits_outcomes, formatter={"maxima": ".1f", "minima": ".1f"}, fontsize=20, rot=90)

sta_policies = df_full[(df_full["Policy type"] == "STA")]["policy"].unique()
# Plot selected policies with the manually specified colors
for idx, policy_type in enumerate(df_reference_subset['Policy type'].unique()):
    selected_data = df_reference_subset[df_reference_subset['Policy type'] == policy_type]
    lines = paraxes.plot(selected_data, label=f'{policy_type}', color=color_coding[policy_type], linewidth=2)
    # Annotate only the lines with 'STA' policy type
    if policy_type == 'STA':
        # Get the axis limits for the last outcome
        last_axis_limits = paraxes.limits[objective_outcomes[-1]]

        for i, (line, policy_name) in enumerate(zip(selected_data[objective_outcomes].values, sta_policies)):
            # Normalize the y-position within the last axis limits
            y_value = line[-1]  # This is the data value at the last axis
            y_relative = (y_value - last_axis_limits[0]) / (last_axis_limits[1] - last_axis_limits[0])

            # Calculate the x-position as a relative position beyond the last axis
            # Using the axes' number to get a position just after the last axis
            x_relative = len(objective_outcomes) + 0.02  # The -0.5 here is to give some space from the last axis

            # Now we use the axes' transform to place the text correctly
            # The axes for the parallel coordinates plot are in a list, paraxes.axes
            # We take the transform of the last axis
            last_axis_transform = paraxes.axes[-1].transData

            text = policy_name
            fontsize = 18
            paraxes.fig.text(x_relative, y_relative, text, transform=last_axis_transform,
                             fontsize=fontsize, color=color_coding["STA"], ha='left', va='center',
                             bbox=dict(facecolor='white', alpha=0, edgecolor='none', boxstyle='round,pad=0.2'))

# Get the figure that parcoords is using
parcoords_fig = plt.gcf()  # 'gcf' stands for 'Get Current Figure'
# for ax in paraxes.axes:
#     ax.set_xticklabels([])  # This removes the x-axis tick labels
#     ax.set_yticklabels([])  #
# Set figure size and facecolor
parcoords_fig.set_size_inches(10, 12)
# parcoords_fig.patch.set_facecolor((1, 1, 1, 0))  # Set transparency

# Add legend
paraxes.legend()

# Get the figure that parcoords is using
parcoords_fig = plt.gcf()
legend = parcoords_fig.legends[0]

# Set the legend location by updating the bounding box. Adjust the coordinates as needed.
legend.set_bbox_to_anchor((1.01, 0.7, 0, 0))

# Update the fontsize of the legend text and the linewidth of the legend's lines.
for text in legend.get_texts():
    text.set_fontsize(18)
for line in legend.get_lines():
    line.set_linewidth(6)

# Remove the frame (box) of the legend
legend.get_frame().set_edgecolor('none')

# Save to file
parcoords_fig.savefig("./figs/parcoords_candidate_policies_reference_outcomes.png",
                      dpi=300, format="png", bbox_inches="tight", transparent=True)

# %% More plots on reference scenario performance
# Pairplot outcomes on outcomes
df_reference_subset = df_reference_subset.copy()
df_reference_subset['Policy type'] = df_reference_subset['Policy type'].astype('category')
df_reference_subset['Policy type'] = df_reference_subset['Policy type'].cat.reorder_categories(
    ['STA', 'All levers', 'No transport efficiency'], ordered=True
)
# Now create the pairplot.
g = sns.pairplot(
    data=df_reference_subset,
    x_vars=objective_outcomes,
    y_vars=objective_outcomes,
    hue="Policy type",
    palette=color_coding,
    diag_kind="hist",
    diag_kws={"common_norm": False}
)

# Move the legend to the right of the figure.
# You can adjust the values inside bbox_to_anchor to best fit your figure's size.
g._legend.remove()
g.add_legend(title='', bbox_to_anchor=(1, .5), loc='center right', fontsize=16, ncols=1)

for ax in g.axes.flatten():
    # Set new font sizes here
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)  # Adjust fontsize as needed for x labels
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)
    ax.tick_params(axis='x', labelsize=12)  # Adjust labelsize as needed for x ticks
    ax.tick_params(axis='y', labelsize=12)  # Adjust labelsize as needed for y ticks


# Adjust the figure to make space for the legend
# plt.subplots_adjust(right=0.9)  # You might need to tweak this value.

# %%
# Pairplot levers on levers
plt.figure(figsize=(8, 6))
sns.pairplot(data=df_reference_subset, x_vars=levers, y_vars=levers,
             hue="Policy type", palette=color_coding, diag_kws={"common_norm": False})


# %% Parcoords plot on levers
limits_levers = pd.DataFrame()  # Create a dataframe for lever-based limits
for item in levers:
    limits_levers.loc[0, item] = min(df_reference_subset[item])  # Get lower bound
    limits_levers.loc[1, item] = max(df_reference_subset[item])  # Get upper bound
paraxes = parcoords.ParallelAxes(limits_levers, formatter={"maxima": ".1f", "minima": ".1f"}, fontsize=20, rot=90)
# Plot selected policies with the manually specified colors
for idx, policy_type in enumerate(df_reference_subset['Policy type'].unique()):
    selected_data = df_reference_subset[df_reference_subset['Policy type'] == policy_type]
    paraxes.plot(selected_data, label=f'{policy_type}', color=color_coding[policy_type], linewidth=2)

# Get the figure that parcoords is using
parcoords_fig = plt.gcf()  # 'gcf' stands for 'Get Current Figure'
# for ax in paraxes.axes:
#     ax.set_xticklabels([])  # This removes the x-axis tick labels
#     ax.set_yticklabels([])  #
# Set figure size and facecolor
parcoords_fig.set_size_inches(10, 10)
# parcoords_fig.patch.set_facecolor((1, 1, 1, 0))  # Set transparency

# Optionally, you can add a legend if you need it
paraxes.legend()

# Get the figure that parcoords is using
parcoords_fig = plt.gcf()
legend = parcoords_fig.legends[0]

# Set the legend location by updating the bounding box. Adjust the coordinates as needed.
legend.set_bbox_to_anchor((1, 0.8, 0, 0))

# Update the fontsize of the legend text and the linewidth of the legend's lines.
for text in legend.get_texts():
    text.set_fontsize(18)
for line in legend.get_lines():
    line.set_linewidth(6)

# Remove the frame (box) of the legend
legend.get_frame().set_edgecolor('none')

# Instead of saving 'fig', now we save 'parcoords_fig' which is the actual figure containing the plot.
parcoords_fig.savefig("./figs/parcoords_candidate_policies_reference_levers.png",
                      dpi=300, format="png", bbox_inches="tight", transparent=True)

# %%Relationship between biofuels and electrification rate in reference scenario
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_full[df_full["scenario"] == "Reference"], x='Electrified VKT share light vehicles',
                y='M4_energy_use_bio', hue="Policy type", palette=color_coding, legend=False, s=25)
# Annotating STA policies
bbox_props = dict(boxstyle="round,pad=0.1", fc="white", ec="none", lw=0, alpha=0.5)

for sta_policy in df_full[df_full["Policy type"] == "STA"]["policy"].unique():
    df_sta_policy = df_full[df_full['policy'] == sta_policy]

    plt.annotate(
        sta_policy,
        (df_sta_policy["Electrified VKT share light vehicles"].values[0],
         df_sta_policy["M4_energy_use_bio"].values[0]),
        textcoords="offset points",
        xytext=(0, -15),
        ha='center', color=color_coding["STA"], fontsize=12, bbox=bbox_props
    )
plt.axvline(x=0.68, linestyle="--", color="black", linewidth=1, zorder=10)
plt.text(s="X6_car_electrification_rate: 0.68", x=0.68+.001, y=4, color="black")
plt.xlim([0.66, .85])
sns.despine()

# %%Relationship between biofuels and electrification rate in reference scenario
# First, create a scatter plot with all policy types
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_full[df_full["scenario"] == "Reference"],
    x='Electrified VKT share light vehicles',
    y='M4_energy_use_bio',
    hue="Policy type",
    palette=color_coding,
    legend=False,
    s=25
)

# Then, draw regression lines for two specific policy types
# You'll need to adjust 'PolicyType1' and 'PolicyType2' to match your actual policy type names
for policy_type in ['All levers', 'No transport efficiency']:
    # Filter the data
    df_subset = df_full[(df_full["scenario"] == "Reference") & (df_full["Policy type"] == policy_type)]

    # Plot the regression line
    sns.regplot(
        data=df_subset,
        x='Electrified VKT share light vehicles',
        y='M4_energy_use_bio',
        scatter=False,  # Do not draw scatter again
        color=color_coding[policy_type],
        label=policy_type
    )

for sta_policy in df_full[df_full["Policy type"] == "STA"]["policy"].unique():
    df_sta_policy = df_full[df_full['policy'] == sta_policy]

    plt.annotate(
        sta_policy,
        (df_sta_policy["Electrified VKT share light vehicles"].values[0],
         df_sta_policy["M4_energy_use_bio"].values[0]),
        textcoords="offset points",
        xytext=(0, -15),
        ha='center', color=color_coding["STA"], fontsize=12, bbox=bbox_props
    )

# Add reference line and text - you can continue to use your existing code here
plt.axvline(x=0.68, linestyle="--", color="black", linewidth=1, zorder=10)
plt.text(s="Reference scenario input car \n electrification rate: 0.68", x=0.68+.001, y=4, color="black")

# Set the x and y limits of the plot
plt.xlim([0.66, .85])

# Despine the plot
sns.despine()

# Show the legend
plt.legend()

# Show the plot
plt.show()

# %% Distribution of electrification outcomes
vars_pairplot = ["Electrified VKT share light vehicles", "Bio share of fuel energy"]
g = sns.pairplot(data=df_full, x_vars=vars_pairplot,
                 y_vars=vars_pairplot, hue="CO2 target not met",
                 palette=["green", "red"], plot_kws=dict(s=.01, alpha=1))
# %% Distribution of electrification and bio outcomes
g = sns.pairplot(data=df_full, x_vars=vars_pairplot, kind="kde",
                 y_vars=vars_pairplot, hue="CO2 target not met",
                 palette=["green", "red"], plot_kws=dict(fill=True, alpha=0.5, levels=10, gridsize=100))

# %% Calculate robustness metrics

# Define robustness metrics
all_robustness_metrics = ["90th percentile", "Reference", "90_percentile_deviation",
                          "Max", "Mean", "Standard deviation", "Mean_stdev", "Max_regret"]
robustness_metrics = ["90_percentile_deviation", "Mean_stdev"]

# Initial setup - Calculating Zero regret values

results = []
grouped = df_full.groupby(['Uncertainty set', 'scenario'])
for (uncertainty_set, scenario), group in grouped:
    for outcome in key_outcomes:
        zero_regret_value = group[outcome].min()
        results.append({
            "Outcome": outcome,
            "Uncertainty set": uncertainty_set,
            "scenario": scenario,
            "Zero regret": zero_regret_value
        })

zero_regret_df = pd.DataFrame(results)


# Calculate other metrics for each policy
all_policies = df_full["policy"].unique()
all_uncertainty_sets = df_full["Uncertainty set"].unique()

metrics_data = []
long_metrics_data = []  # For creating the long format DataFrame

for policy in all_policies:
    if policy != "Reference policy":
        for uncertainty_set in all_uncertainty_sets:
            df_temp = df_full[(df_full["policy"] == policy) & (df_full["Uncertainty set"] == uncertainty_set)]
            policy_data = {
                "policy": policy,
                "Policy type": df_temp["Policy type"].iloc[0],
                "Uncertainty set": uncertainty_set
            }

            co2_outcome_data = df_temp["M1_CO2_TTW_total"]
            satisficing_metric = (co2_outcome_data < 1.89).sum() / len(co2_outcome_data)

            # Add satisficing metric to policy_data
            policy_data["Satisficing metric M1_CO2_TTW_total"] = satisficing_metric

            for outcome in key_outcomes:
                outcome_data = df_temp[outcome]
                ref_outcome_data = df_temp[df_temp["scenario"] == "Reference"][outcome]

                metrics = {
                    f"{metric} {outcome}": (
                        outcome_data.quantile(0.9),
                        ref_outcome_data.values[0],
                        (outcome_data.quantile(0.9) - ref_outcome_data.values[0]) / abs(ref_outcome_data.values[0]),
                        outcome_data.max(),
                        outcome_data.mean(),
                        outcome_data.std(),
                        (outcome_data.mean()) * (outcome_data.std()),
                        None  # Placeholder for Max_regret which will be calculated later
                    )[all_robustness_metrics.index(metric)] for metric in all_robustness_metrics
                }

                policy_data.update(metrics)

                regrets = abs(outcome_data - zero_regret_df.loc[
                    (zero_regret_df["Uncertainty set"] == uncertainty_set) &
                    (zero_regret_df["Outcome"] == outcome) &
                    zero_regret_df["scenario"].isin(df_temp["scenario"].unique()),
                    "Zero regret"
                ].values)

                policy_data[f"Max_regret {outcome}"] = regrets.max()
                metrics[f"Max_regret {outcome}"] = regrets.max()  # Update the metrics dictionary

                # Append data to long_metrics_data
                for metric in all_robustness_metrics:
                    long_metrics_data.append({
                        "policy": policy,
                        "Policy type": df_temp["Policy type"].iloc[0],
                        "Uncertainty set": uncertainty_set,
                        "Outcome": outcome,
                        "Robustness metric": metric,
                        "Value": metrics[f"{metric} {outcome}"]
                    })

            metrics_data.append(policy_data)

policy_metrics_df = pd.DataFrame(metrics_data)
policy_metrics_df_long = pd.DataFrame(long_metrics_data)
policy_metrics_sta = policy_metrics_df[policy_metrics_df["Policy type"] == "STA"]
policy_metrics_sta_long = policy_metrics_df_long[policy_metrics_df_long["Policy type"] == "STA"]
# %% visualize relationship between robustness metrics
# Use Whitegrid
sns.set_style("whitegrid")

all_uncertainty_sets = policy_metrics_df["Uncertainty set"].unique()

# Loop over each outcome and uncertainty set combination
for outcome in key_outcomes:
    for uncertainty_set in all_uncertainty_sets:
        # Filter the data
        subset_df = policy_metrics_df[policy_metrics_df["Uncertainty set"] == uncertainty_set]

        # Metrics to visualize in their original form (without newlines)
        metrics = [
            f"90_percentile_deviation {outcome}",
            f"Max_regret {outcome}",
            f"Mean_stdev {outcome}"
        ]
        metrics = [
            f"90_percentile_deviation {outcome}",
            f"Mean_stdev {outcome}"
        ]

        # Create the pairplot
        plot = sns.pairplot(data=subset_df, vars=metrics, diag_kind="kde", hue="Policy type",
                            height=2.5, palette=color_coding, diag_kws={"common_norm": False})
        plot.fig.suptitle(f"Outcome: {outcome}, Uncertainty Set: {uncertainty_set}", y=1.02)

        # Adjust x-labels and y-labels for better readability
        for ax in plot.axes.flat:
            ax.set_xlabel(ax.get_xlabel().replace(" ", "\n"), rotation=0)
            # If y-labels are still overlapping, try using 45 here too.
            ax.set_ylabel(ax.get_ylabel().replace(" ", "\n"), rotation=90)
        # Position of legend relative to axes bounding box (x: 1 is right outside of the plot, y: 0.5 is centered)
        plot._legend.set_bbox_to_anchor((1.3, 0.5))
        plt.tight_layout()
        plt.show()
# %% Visualize satisficing CO2
g = sns.displot(policy_metrics_df, x="Satisficing metric M1_CO2_TTW_total",
                hue="Policy type", kind="kde", common_norm=False, bw_adjust=0.5, cut=0,
                palette=color_coding, fill=True)

g._legend.set_bbox_to_anchor((0.7, 0.9, 0, 0))
# Remove the legend title
g._legend.set_title('')

# %% Visualize distributions of different policy metrics
# Create the FacetGrid
sns.set_style("white")
g = sns.FacetGrid(policy_metrics_df_long[policy_metrics_df_long["Robustness metric"].isin(
    robustness_metrics)], col='Robustness metric', row='Outcome', sharey=False)

# Map the boxplot to the grid, specify the width of the boxplot
g.map_dataframe(sns.boxplot, x="Policy type", y="Value", hue="Policy type",
                palette=color_coding, width=0.7, dodge=False)

# Remove the x-ticks as they are redundant with the legend
for ax in g.axes.flat:
    ax.set_xticks([])

# Adjust the legend. Here we re-add it with adjusted parameters after removing it above.
g.add_legend(title=None, bbox_to_anchor=(0.5, 1.02), loc='upper center', borderaxespad=0, ncol=3, fontsize=12)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set_axis_labels("", "")
# Improve the layout
plt.tight_layout()
g.despine()
# for robustness_metric in robustness_metrics:
#     for outcome in key_outcomes:

#         # Visualization
#         sns.displot(policy_metrics_df, x=f"{robustness_metric} {outcome}",
#                     hue="Policy type", common_norm=False, kind="kde")
#         plt.title(f'Distribution of {robustness_metric} for {outcome}')
# %% Pairplot of robustness

for robustness_metric in robustness_metrics:
    robustness_metric_outcomes = []
    for outcome in key_outcomes:
        robustness_metric_outcomes.append(f"{robustness_metric} {outcome}")
    sns.pairplot(policy_metrics_df, x_vars=robustness_metric_outcomes,
                 y_vars=robustness_metric_outcomes, hue="Policy type", palette=color_coding)
# %% Perform statistical test of robustness metrics differences ALl levers No transport efficiency
for robustness_metric in robustness_metrics:
    for outcome in key_outcomes:

        # Filter data for the two specific policy types
        filtered_data = policy_metrics_df_long[policy_metrics_df_long['Policy type'].isin(
            ['All levers', 'No transport efficiency'])]

        # Statistical Test - Mann-Whitney U Test
        group1 = filtered_data[(filtered_data['Policy type'] == 'All levers') & (
            filtered_data['Outcome'] == outcome) & (filtered_data["Robustness metric"] == f"{robustness_metric}")]
        group2 = filtered_data[(filtered_data['Policy type'] == 'No transport efficiency') & (
            filtered_data['Outcome'] == outcome) & (filtered_data["Robustness metric"] == f"{robustness_metric}")]

        # Calculate Descriptive Statistics
        group1_stats = group1.describe()
        group2_stats = group2.describe()

        print(
            f"{robustness_metric} {outcome} - All levers: Median = {group1_stats.loc['50%'][0]}, Mean = {group1_stats.loc['mean'][0]}")
        print(
            f"{robustness_metric} {outcome} - No transport efficiency: Median = {group2_stats.loc['50%'][0]}, Mean = {group2_stats.loc['mean'][0]}")

        # Perform the test (use alternative='two-sided' for a two-tailed test)
        u_statistic, p_value = stats.mannwhitneyu(
            group1['Value'].dropna(), group2['Value'].dropna(), alternative='two-sided')
        print(f'Mann-Whitney U test for {robustness_metric} {outcome}: p-value = {p_value}')

# %% Parcoords of all policies and robustness metrics
# Save the original default size
original_figsize = plt.rcParams["figure.figsize"]

# Set the figure size
plt.rcParams["figure.figsize"] = [8, 8]


highlight_policies = [969, 457, 1044, 1294, 2395, 1507]
linestyles = ["solid", "dotted", "dashdot", "solid", "dotted", "dashdot"]

for robustness_metric in robustness_metrics:
    dynamic_outcomes = [f"{robustness_metric} {outcome}" for outcome in key_outcomes]  # Dynamic outcomes

    # for uncertainty_set in policy_metrics_df["Uncertainty set"].unique():
    for uncertainty_set in ["XP"]:
        policy_metrics_subset = policy_metrics_df[policy_metrics_df["Uncertainty set"] == uncertainty_set]

        limits_outcomes = pd.DataFrame()  # Create a dataframe for limits
        for i, item in enumerate(dynamic_outcomes):
            limits_outcomes.loc[0, item] = min(policy_metrics_subset[item])  # Get lower bound
            limits_outcomes.loc[1, item] = max(policy_metrics_subset[item])  # Get upper bound
        # Create a dictionary for renaming columns
        rename_dict = {long: short for long, short in zip(dynamic_outcomes, key_outcomes)}

        # Rename columns of limits_outcomes for display purposes
        limits_outcomes.rename(columns=rename_dict, inplace=True)
        limits = limits_outcomes

        # Create the parallel coordinates
        paraxes = parcoords.ParallelAxes(limits)
        paraxes.ticklabels = key_outcomes
        # Define your color coding for the legend
        labels = list(color_coding.keys())
        legend_elements = [Patch(facecolor=color_coding[label], label=label) for label in labels]
        sta_policies = df_full[(df_full["Policy type"] == "STA")]["policy"].unique().tolist()

        # Loop over rows in dataframe. Plot all policies
        for i, row in policy_metrics_subset.iterrows():
            row_renamed = row.rename(rename_dict)
            policy_type = row["Policy type"]
            data = row_renamed[key_outcomes]
            color = ultra_light_color_coding[policy_type]
            paraxes.plot(data, color=color)

        # Then plot the highlighted policies. This is to ensure highlighted policies are plot on top
        count = 0
        for i, row in policy_metrics_subset.iterrows():
            policy_type = row["Policy type"]
            if policy_type == 'STA':
                row_renamed = row.rename(rename_dict)
                data = row_renamed[key_outcomes]
                color = color_coding[policy_type]
                paraxes.plot(data, color=color)

                # Get the axis limits for the last outcome
                last_axis_limits = paraxes.limits.iloc[:, -1]

                # Normalize the y-position within the last axis limits
                y_value = data[rename_dict.get(dynamic_outcomes[-1])]  # This is the data value at the last axis
                y_relative = (y_value - last_axis_limits[0]) / (last_axis_limits[1] - last_axis_limits[0])

                # Calculate the x-position as a relative position beyond the last axis
                x_relative = len(dynamic_outcomes) + 0.05  # Adjust offset to make it pretty

                # Now we use the axes' transform to place the text correctly
                last_axis_transform = paraxes.axes[-1].transData

                text = policy_metrics_subset.loc[i, "policy"]  # Use the policy name from the current row
                fontsize = 16
                paraxes.fig.text(x_relative, y_relative, text, transform=last_axis_transform,
                                 fontsize=fontsize, color='red', ha='left', va='center',
                                 bbox=dict(facecolor='white', alpha=0, edgecolor='none', boxstyle='round,pad=0.2'))
            # elif (int(row["policy"]) in highlight_policies):

            #     row_renamed = row.rename(rename_dict)
            #     data = row_renamed[key_outcomes]
            #     color = color_coding[policy_type]
            #     paraxes.plot(data, color=color, linestyle=linestyles[count],linewidth=2.5)
            #     count+=1
        # Add and adjustlegend manually
        plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc="lower right", fontsize=12, frameon=False)

        # for i, ax in enumerate(paraxes.axes):

        #     ax.set_xticklabels([key_outcomes[i]])  # This removes the x-axis tick labels
        #     ax.set_yticklabels([])  #

        plt.suptitle(f"Robustness metric: {robustness_metric}", fontsize=14, x=0, y=1.02, horizontalalignment="left")
        plt.show()

plt.rcParams["figure.figsize"] = original_figsize

# %% Parcoords of all metrics with brushing

# Save the original default size
original_figsize = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = [10, 10]

outcomes = [
    'M1_CO2_TTW_total',
    'M2_driving_cost_car',
    'M3_driving_cost_truck',
    'M4_energy_use_bio',
    'M5_energy_use_electricity'
]
metrics = ["90_percentile_deviation", "Max_regret", "Mean_stdev"]

# 1. Normalize the metrics for each outcome
for metric in metrics:
    policy_metrics_df[f"sum_normalized_{metric}"] = 0  # initialize at 0
    for outcome in outcomes:
        col_name = f"{metric} {outcome}"
        policy_metrics_df[f"normalized_{col_name}"] = (policy_metrics_df[col_name] - policy_metrics_df[col_name].min()) / \
            (policy_metrics_df[col_name].max() - policy_metrics_df[col_name].min())
        policy_metrics_df[f"sum_normalized_{metric}"] = policy_metrics_df[f"sum_normalized_{metric}"] + \
            policy_metrics_df[f"normalized_{metric} {outcome}"]

# Main loop to plot each metric-uncertainty set combination
for metric in metrics:
    dynamic_outcomes = [f"{metric} {outcome}" for outcome in outcomes]

    # for uncertainty_set in policy_metrics_df["Uncertainty set"].unique():
    for uncertainty_set in ["XP"]:
        subset_uncertainty = policy_metrics_df[policy_metrics_df["Uncertainty set"] == uncertainty_set].copy()

        # Normalize data for the robustness metrics
        normalized_data = subset_uncertainty[dynamic_outcomes].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        subset_uncertainty['sum_metric'] = normalized_data.sum(axis=1)

        # Determine top 5 policies for this metric and uncertainty set, for each policy type, excluding "STA"
        top_policies = []
        for ptype in color_coding.keys():
            if ptype != "STA":
                ptype_policies = subset_uncertainty[subset_uncertainty["Policy type"]
                                                    == ptype].nsmallest(5, 'sum_metric').index.tolist()
                top_policies.extend(ptype_policies)
            else:
                ptype_policies = subset_uncertainty[subset_uncertainty["Policy type"] == ptype].index.tolist()
                top_policies.extend(ptype_policies)

        # Plotting
        color_list = [color_coding[ptype] if i in top_policies else light_color_coding[ptype]
                      for i, ptype in subset_uncertainty["Policy type"].items()]
        alpha_list = [0.9 if i in top_policies else 0.1 for i, ptype in subset_uncertainty["Policy type"].items()]
        linewidth_list = [2 if i in top_policies else 1 for i in subset_uncertainty.index]
        zorder_list = [10 if i in top_policies else 1 for i in subset_uncertainty.index]

        limits_outcomes = pd.DataFrame(
            {col: [subset_uncertainty[col].min(), subset_uncertainty[col].max()] for col in dynamic_outcomes})
        paraxes = parcoords.ParallelAxes(limits_outcomes)
        labels = list(color_coding.keys())
        legend_elements = [Patch(facecolor=color_coding[label], label=label) for label in labels]

        # Plotting logic remains unchanged
        for index, (i, row) in enumerate(subset_uncertainty.iterrows()):
            data = pd.DataFrame([row[dynamic_outcomes]])
            color = color_list[index]
            alpha = alpha_list[index]
            linewidth = linewidth_list[index]
            zorder = zorder_list[index]
            paraxes.plot(data.iloc[0], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
            # Annotate only the lines with 'STA' policy type

        plt.legend(handles=legend_elements, loc='upper right')
        #plt.title(f"Metric: {metric}, Uncertainty set: {uncertainty_set}")
        plt.title(f"Metric: {metric}")
        plt.show()

plt.rcParams["figure.figsize"] = original_figsize

# %% Analyze whether we can find a policy that is more robust in all outcomes for all STA policies

# Assuming policy_metrics_df is a pre-defined DataFrame and metrics is a list of metrics
policy_metrics_df_sta = policy_metrics_df[policy_metrics_df["Policy type"] == "STA"]

# Initialize columns for each metric to indicate if a policy is better than all STA policies
for metric in metrics:
    policy_metrics_df[f"{metric}_Is_Better"] = False

for policy in policy_metrics_df["policy"].unique():
    policy_data = policy_metrics_df[policy_metrics_df["policy"] == policy]

    for metric in metrics:
        is_metric_better_for_all_sta = True

        dynamic_outcomes = [f"{metric} {outcome}" for outcome in key_outcomes]

        for sta_policy in policy_metrics_df_sta["policy"].unique():
            sta_policy_data = policy_metrics_df_sta[policy_metrics_df_sta["policy"] == sta_policy]

            is_better_for_this_sta_policy = True
            for outcome in dynamic_outcomes:
                policy_outcome_value = policy_data[outcome].values[0] if not policy_data[outcome].empty else None
                sta_outcome_value = sta_policy_data[outcome].values[0] if not sta_policy_data[outcome].empty else None

                if policy_outcome_value is None or sta_outcome_value is None or not (policy_outcome_value < sta_outcome_value):
                    is_better_for_this_sta_policy = False
                    break

            if not is_better_for_this_sta_policy:
                is_metric_better_for_all_sta = False
                break

        # Update the DataFrame
        policy_metrics_df.loc[policy_metrics_df["policy"] == policy,
                              f"{metric}_Is_Better"] = is_metric_better_for_all_sta

policy_metrics_df
# %% Prepare data for in-dpeth analysis of STA policies
columns_90th_percentile = [
    col for col in policy_metrics_sta.columns if '90_percentile_deviation' in col and 'normalized' not in col]

columns_90th_percentile.append("policy")
policy_metrics_sta_90 = policy_metrics_sta[columns_90th_percentile]
# Set "policy" column as the index for policy_metrics_sta_90
policy_metrics_sta_90.set_index("policy", inplace=True)
# Extract the "Mean/stdev M...." columns
columns_mean_stdev = [col for col in policy_metrics_sta.columns if (
    ('Mean_stdev' in col) & ("normalized" not in col) & ("sum" not in col))]+["policy"]

policy_metrics_mean_stdev = policy_metrics_sta[columns_mean_stdev]
policy_metrics_mean_stdev.set_index("policy", inplace=True)

df_full_sta = df_full[df_full["scenario"] == "Reference"]
df_full_sta = df_full_sta[df_full_sta["Policy type"] == "STA"][key_outcomes+["policy"]]
# Set "policy" column as the index for df_full_sta
df_full_sta.set_index("policy", inplace=True)


df_sta = pd.concat([df_full_sta, policy_metrics_sta_90], axis=1)
df_sta = pd.concat([df_sta, policy_metrics_mean_stdev], axis=1)
df_sta["policy"] = df_sta.index

x_vars = key_outcomes
y_vars = ['90_percentile_deviation M1_CO2_TTW_total',
          '90_percentile_deviation M2_driving_cost_car',
          '90_percentile_deviation M3_driving_cost_truck',
          '90_percentile_deviation M4_energy_use_bio',
          '90_percentile_deviation M5_energy_use_electricity']

y_vars += columns_mean_stdev
# Create a FacetGrid with scatterplots for each metric

# Your existing code for creating df_long
df_long = pd.melt(df_sta, id_vars=["policy"],
                  value_vars=['M1_CO2_TTW_total', 'M2_driving_cost_car', 'M3_driving_cost_truck',
                              'M4_energy_use_bio', 'M5_energy_use_electricity'],
                  var_name='Metric', value_name='Outcome in reference scenario')

# Add the 90th percentile deviation values
# Adjust this based on how your data is structured
df_long['90th Percentile Deviation'] = df_long.apply(
    lambda row: df_sta.at[row['policy'], '90_percentile_deviation ' + row['Metric']], axis=1
)

df_long['Mean_stdev'] = df_long.apply(
    lambda row: df_sta.at[row['policy'], 'Mean_stdev ' + row['Metric']], axis=1
)

# %%Ranking of reference scenario and 90th percentile deviation


def rank_values(series):
    return series.rank(method='dense')


# Apply the ranking within each group for each column
df_long['Ranking outcome in reference scenario'] = df_long.groupby(
    'Metric')['Outcome in reference scenario'].transform(rank_values)
df_long['Ranking 90th percentile deviation'] = df_long.groupby(
    'Metric')['90th Percentile Deviation'].transform(rank_values)

df_long['Ranking mean_stdev'] = df_long.groupby(
    'Metric')['Mean_stdev'].transform(rank_values)

# Sample over different values of w_ref_rank
weights = [0, 0.25, 0.5, 0.75, 1]
weights = [1, 0.5, 0]
df_samples = pd.DataFrame()

for w_ref in weights:
    w_rob = 1 - w_ref
    temp_df = df_long.copy()
    temp_df["Weighted ranking"] = w_ref * temp_df['Ranking outcome in reference scenario'] + \
        w_rob * temp_df['Ranking 90th percentile deviation']
    temp_df['w ref'] = w_ref

    df_samples = pd.concat([df_samples, temp_df])

df_samples["Ranking weighted"] = df_samples.groupby([
    'Metric', "w ref"])["Weighted ranking"].transform(rank_values)

# Plot using FacetGrid
color_sta_policies = ["green", "lightcoral", "red", "darkred", "coral", "cornflowerblue", "blue", "darkblue"]
g = sns.FacetGrid(df_samples, col='w ref', height=3.5, aspect=1.25, sharex=True, sharey=False)
g.map(sns.lineplot, 'Metric', 'Ranking weighted', 'policy', marker='o', palette=color_sta_policies)

g.fig.subplots_adjust(hspace=0.8, wspace=0.3)
# Enhancing the plot
g.set_xticklabels(rotation=90)
plt.subplots_adjust(top=0.9)
#g.fig.suptitle('Weighted Ranking of Policies for Different w_ref Values 90th percentile dev', size=16)
g.add_legend(title="STA policy", bbox_to_anchor=(0.5, 1.02), loc='center right', borderaxespad=0.1, ncol=8)
g.set_titles(col_template="Reference scenario performance weight: {col_name}", row_template="{row_name}")
g.set_axis_labels("", "Ranking")
plt.show()
plt.figure()
# weights = [0, 0.25, 0.5, 0.75, 1]
# df_samples = pd.DataFrame()
# for w_ref in weights:
#     w_rob = 1 - w_ref
#     temp_df = df_long.copy()
#     temp_df["Weighted ranking"] = w_ref * temp_df['Ranking outcome in reference scenario'] + \
#         w_rob * temp_df['Ranking mean_stdev']
#     temp_df['w ref'] = w_ref
#     df_samples = pd.concat([df_samples, temp_df])

# # Plot using FacetGrid
# color_map = ["green", "lightcoral", "red", "darkred", "coral", "cornflowerblue", "blue", "darkblue"]
# g = sns.FacetGrid(df_samples, col='w ref', col_wrap=3, height=4, aspect=1.5, sharex=False)
# g.map(sns.lineplot, 'Metric', 'Weighted ranking', 'policy', marker='o', palette=color_map)
# g.fig.subplots_adjust(hspace=0.6, wspace=0.3)
# # Enhancing the plot
# g.add_legend()
# g.set_xticklabels(rotation=45)
# plt.subplots_adjust(top=0.9)
# g.fig.suptitle('Weighted Ranking of Policies for Different w_ref Values mean/stdev', size=16)

# plt.show()

# %% Visualize robustness and reference scenario performance 90th percentile


def reference_robustness_scatter_plot(df, y_variable, plot_title):
    # Create the FacetGrid
    g = sns.FacetGrid(df, col='Metric', col_wrap=3, sharex=False, sharey=False)
    g = g.map(plt.scatter, 'Outcome in reference scenario', y_variable, color='red')
    g.fig.subplots_adjust(hspace=0.38, wspace=0.25)

    # Annotate each point with its policy
    for ax, (metric, subset) in zip(g.axes.flatten(), df.groupby('Metric')):
        subset.apply(lambda row: ax.text(row['Outcome in reference scenario'],
                                         row[y_variable],
                                         str(row['policy']),
                                         fontsize=10,
                                         ha='left', va="bottom"), axis=1)

    # Set the subplot titles and x-axis labels for all subplots
    for ax, title in zip(g.axes.flatten(), x_vars):
        ax.set_title(title)
        ax.set_xlabel('Outcome in reference scenario')

    # Set the subplot titles
    g.set_titles("{col_name}", size=10)
    g.fig.suptitle(plot_title, size=10, y=1.02)


for robustness_metric, plot_title in [
    ('90th Percentile Deviation', 'STA policies: 90th percentile deviation vs reference scenario performance'),
    ('Mean_stdev', 'STA policies: Mean-stdev vs reference scenario performance')
]:
    reference_robustness_scatter_plot(df_long, robustness_metric, plot_title)
# %% Visualie relationship between mean and stdev
for metric in key_outcomes:
    plt.figure()
    sns.scatterplot(policy_metrics_sta, x=f"Mean {metric}", y=f"Standard deviation {metric}", hue="policy")

# %% Vulnerability analysis
# Policies selected for analysis
# vulnerability_policies = [291, 3051, "B"]

# # Parcoords of vulnerability policies
# paraxes = parcoords.ParallelAxes(limits_levers, formatter={"maxima": ".1f", "minima": ".1f"}, fontsize=20, rot=90)
# # Plot selected policies with the manually specified colors
# for idx, policy in enumerate(vulnerability_policies):
#     selected_data = df_full[df_full['policy'] == str(policy)].iloc[0]
#     policy_type = selected_data["Policy type"]
#     paraxes.plot(selected_data, label=f'Policy type: {policy_type}', color=color_coding[policy_type])

# # Get the figure that parcoords is using
# parcoords_fig = plt.gcf()  # 'gcf' stands for 'Get Current Figure'
# # for ax in paraxes.axes:
# #     ax.set_xticklabels([])  # This removes the x-axis tick labels
# #     ax.set_yticklabels([])  #
# # Set figure size and facecolor
# parcoords_fig.set_size_inches(10, 10)
# # parcoords_fig.patch.set_facecolor((1, 1, 1, 0))  # Set transparency

# # Optionally, you can add a legend if you need it
# paraxes.legend()

# %% Find most common vulnerabilities

df_policy_vulnerabilities = pd.DataFrame(index=df_full["policy"].unique())
n_vulnerabilities = 2
vulnerabilities_list = []
for policy in df_full["policy"].unique():
    df_policy_vulnerabilities.loc[policy, "policy"] = policy
    df_policy_vulnerabilities.loc[policy, "Policy type"] = df_full[df_full["policy"] == policy]["Policy type"].unique()[
        0]
    df_policy_vulnerabilities.loc[policy, "CO2 target not met count"] = sum(
        df_full[df_full["policy"] == policy]["CO2 target not met"])
    df_v = df_full[df_full["policy"] == policy]
    df_v = df_v[df_v["scenario"] != "Reference"]
    df_v = df_v.reset_index(drop=True)

    x = df_v[model.uncertainties.keys()]
    y = df_v["CO2 target not met"]
    prim_alg = prim.Prim(x, y, threshold=0.5)
    box1 = prim_alg.find_box()

    # Track the found vulnerabilities to avoid duplicates
    found_vulnerabilities = []

    # Determine the maximum number of dimensions (vulnerabilities) found
    max_res_dim = max(box1.peeling_trajectory["res_dim"], default=0)

    for i in range(1, min(n_vulnerabilities, max_res_dim) + 1):
        box_i = box1.peeling_trajectory[box1.peeling_trajectory["res_dim"]
                                        == i].sort_values(by="density", ascending=False).head(1)
        if not box_i.empty:
            # Choose point for inspection
            i1 = int(box_i["id"].iloc[0])
            box_data = box1.inspect(i1, style="data")[0][1]
            box_data.columns = box_data.columns.droplevel(0)

            # Get the vulnerability that hasn't been found yet
            for v in box_data.index:
                if v not in found_vulnerabilities:
                    found_vulnerabilities.append(v)
                    df_policy_vulnerabilities.loc[policy, f"Vulnerability {i}"] = v
                    break
            if len(found_vulnerabilities) < i:
                # If we didn't find a new vulnerability, we break out of the loop
                break
        else:
            # If there are no more vulnerabilities to find, we break out of the loop
            break
df_policy_vulnerabilities["CO2 target not met share"] = df_policy_vulnerabilities["CO2 target not met count"] / \
    len(df_full["scenario"].unique())
# %% countplot of vulnerabilities

sns.countplot(df_policy_vulnerabilities, x="Vulnerability 1", hue="Policy type")
plt.figure()
sns.countplot(df_policy_vulnerabilities, x="Vulnerability 2", hue="Policy type")

# Melt the dataframe
df_policy_vulnerabilities_long = pd.melt(df_policy_vulnerabilities, id_vars=['Policy type'],
                                         value_vars=['Vulnerability 1', 'Vulnerability 2'],
                                         var_name='Vulnerability Type', value_name='Vulnerability')

sns.catplot(

    data=df_policy_vulnerabilities_long, x="Vulnerability", row="Vulnerability Type",
    kind="count", hue="Policy type", height=4, aspect=1.5, sharex=True, sharey=True
)
plt.xticks(rotation=90)  # Rotate labels for better readability

# Across all policy types
# Create a combination count table
combination_counts = df_policy_vulnerabilities.groupby(
    ['Vulnerability 1', 'Vulnerability 2']).size().reset_index(name='counts')

# Pivot the table to create a matrix form suitable for a heatmap
matrix_counts = combination_counts.pivot_table(
    index='Vulnerability 1', columns='Vulnerability 2', values='counts', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(12, 10))
# You can change cmap to another colormap like 'viridis' if you prefer
sns.heatmap(matrix_counts, annot=True, cmap='viridis', fmt='d')
plt.title('Heatmap of Policy Vulnerability Combinations')
plt.xlabel('Vulnerability 2')
plt.ylabel('Vulnerability 1')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Per policy type
# Iterate through each Policy type and create a heatmap

# Step 1: Identify the unique vulnerabilities for 'Vulnerability 1' and 'Vulnerability 2', excluding 'nan'
unique_vuln1 = df_policy_vulnerabilities['Vulnerability 1'].dropna().unique()
unique_vuln2 = df_policy_vulnerabilities['Vulnerability 2'].dropna().unique()

# Step 2: Create standardized pivot tables and plot heatmaps for each Policy type
for policy_type in df_policy_vulnerabilities['Policy type'].unique():
    # Filter the DataFrame for the current Policy type
    df_filtered = df_policy_vulnerabilities[df_policy_vulnerabilities['Policy type'] == policy_type]

    # Create a combination count table for this Policy type, using the unique vulnerabilities
    combination_counts = df_filtered.groupby(['Vulnerability 1', 'Vulnerability 2']).size().reset_index(name='counts')

    # Calculate the total number of occurrences for the current Policy type
    total_counts = combination_counts['counts'].sum()

    # Calculate the share of each combination
    combination_counts['share'] = combination_counts['counts'] / total_counts

    # Create a full combination matrix with all unique vulnerabilities in 'Vulnerability 1' and 'Vulnerability 2'
    combination_matrix = pd.DataFrame(index=unique_vuln1, columns=unique_vuln2).fillna(0)

    # Populate the matrix with the shares
    for _, row in combination_counts.iterrows():
        vuln1 = row['Vulnerability 1']
        vuln2 = row['Vulnerability 2']
        combination_matrix.at[vuln1, vuln2] = row['share']

    # Step 3: Plot the heatmap for this Policy type
    plt.figure(figsize=(5, 4))
    sns.heatmap(combination_matrix, annot=True, cmap='viridis', fmt='.0%', vmax=.45)  # Use percentage format
    plt.title(f'{policy_type}')
    plt.xlabel('Vulnerability 2')
    plt.ylabel('Vulnerability 1')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()
# The number of top combinations you want to extract
n = 5

# Assuming df_policy_vulnerabilities is your DataFrame that contains the raw data
# Calculate the share of each vulnerability combination within each policy type
policy_vuln_combinations = (
    df_policy_vulnerabilities
    .groupby(['Policy type', 'Vulnerability 1', 'Vulnerability 2'])
    .size()
    .reset_index(name='counts')
)

# Calculate the total counts for each policy type to normalize and get the share
policy_totals = policy_vuln_combinations.groupby('Policy type')['counts'].transform('sum')
policy_vuln_combinations['share'] = policy_vuln_combinations['counts'] / policy_totals

# Now create a DataFrame to hold the top n combinations per policy type
top_combinations = pd.DataFrame()

# For each policy type, sort by share and take the top n
for policy in policy_vuln_combinations['Policy type'].unique():
    # Get the sorted combinations for the current policy
    sorted_combinations = (
        policy_vuln_combinations[policy_vuln_combinations['Policy type'] == policy]
        .sort_values(by='share', ascending=False)
    )

    # Get the top n combinations for the current policy
    top_for_policy = sorted_combinations.head(n)

    # Calculate cumulative count and share
    top_for_policy['cumulative_count'] = top_for_policy['counts'].cumsum()
    top_for_policy['cumulative_share'] = top_for_policy['share'].cumsum()

    # Append to the top_combinations DataFrame
    top_combinations = pd.concat([top_combinations, top_for_policy], ignore_index=True)

# Optionally, sort the final DataFrame for better readability
top_combinations = top_combinations.sort_values(by=['Policy type', 'share'], ascending=[True, False])

# Reset the index for the final DataFrame
top_combinations.reset_index(drop=True, inplace=True)

# Show the DataFrame
print(top_combinations)
# %% Vulnerability analysis and comparison comapred to policy B
vulnerability_analysis_policy = "B"  # 99 All levers, #183 no transport efficiency "B" STA

df_v = df_full[df_full["policy"].astype(str) == str(vulnerability_analysis_policy)]
df_v = df_v[df_v["scenario"] != "Reference"]
# Perform PRIM analysis
# Set up PRIM
# Reset the index to ensure alignment
df_v = df_v.reset_index(drop=True)

# Set up PRIM with aligned indices
x = df_v[model.uncertainties.keys()]
y = df_v["CO2 target not met"]
x_rotated, rotation_matrix = prim.pca_preprocess(x, y)
prim_alg = prim.Prim(x, y, threshold=0.9)

# Find 1st box
box1 = prim_alg.find_box()
#
# Visualizations of Box1
box1.show_tradeoff()

# for i in range(0, len(box1.peeling_trajectory.T.columns)):
#     s = int(box1.peeling_trajectory.T[i].id)
#     if (i % 2) == 0:
#         plt.text(box1.peeling_trajectory.T[i].coverage, box1.peeling_trajectory.T[i].density + .0, s, fontsize=8)
#     else:
#         plt.text(box1.peeling_trajectory.T[i].coverage, box1.peeling_trajectory.T[i].density + .0, s, fontsize=8)
plt.show()
# Inspect box with restrictions on 2 dimensions and max density (i.e. "last" box with res_dim=2)
box_i = box1.peeling_trajectory[box1.peeling_trajectory["res_dim"]
                                == 2].sort_values(by="density", ascending=False).head(1)

# Choose point for inspection
i1 = int(box_i["id"].iloc[0])
# or choose box manually
#i1 = 15
box1.inspect(i1)
box_data = box1.inspect(i1, style="data")[0][1]
# box data is now a df with each row data for each of the restricted dimensions
box_data.columns = box_data.columns.droplevel(0)

box1.inspect(i1, style='graph')
plt.show()
box1.show_ppt()
# Save the original palette
original_palette = sns.color_palette()
custom_palette = ['green', 'red']
sns.set_palette(sns.color_palette(custom_palette))
ax = box1.show_pairs_scatter(i1)
ax.legend.set_title("")
# for row in ax.axes:
#     for subplot in row:
#         subplot.set_facecolor('none')  # For the subplot background
#         subplot.grid(False)  # To remove the grid lines
#         for spine in subplot.spines.values():
#             spine.set_edgecolor('none')  # For the edges of the subplot
#plt.savefig('test_prim_box.png', transparent=True)

# Calculate statistics for boxes
# Initialize the filtered DataFrame to all True
mask = pd.Series(True, index=df_full.index)

# Apply the filtering criteria for each row in box_data
for index, row in box_data.iterrows():
    min_val = row['min']
    max_val = row['max']
    # Update the mask to check if each entry is within the 'min' and 'max' range
    mask = mask & (df_full[index] >= min_val) & (df_full[index] <= max_val)

# Apply the mask to df_full to filter the rows
df_v_box = df_full[mask]
# %% Visualize loadings in case PCA PRIM was used
# Get loadings
loadings = {
    "Variable": rotation_matrix.index,
    "r_19": rotation_matrix["r_19"],
    "r_20": rotation_matrix["r_20"]
}
df_loadings = pd.DataFrame(loadings)

# Plot for r_19
plt.figure(figsize=(10, 8))
plt.barh(df_loadings["Variable"], df_loadings["r_19"], color='skyblue')
plt.xlabel('Loadings')
plt.ylabel('Variables')
plt.title('PCA Loadings for Component r_19')
plt.tight_layout()
plt.show()

# Plot for r_20
plt.figure(figsize=(10, 8))
plt.barh(df_loadings["Variable"], df_loadings["r_20"], color='lightgreen')
plt.xlabel('Loadings')
plt.ylabel('Variables')
plt.title('PCA Loadings for Component r_20')
plt.tight_layout()
plt.show()

# Preparing the data
variables = rotation_matrix.index  # The original variables
r_19_loadings = rotation_matrix['r_19']
r_20_loadings = rotation_matrix['r_20']

# Setting up the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 8), sharey=True)
fig.suptitle('Comparison of PCA Loadings for Components r_19 and r_20')

# Plot for r_19
axes[0].barh(variables, r_19_loadings, color='skyblue')
axes[0].set_title('Component r_19')
axes[0].set_xlabel('Loadings')
axes[0].set_ylabel('Variables')

# Plot for r_20
axes[1].barh(variables, r_20_loadings, color='lightgreen')
axes[1].set_title('Component r_20')
axes[1].set_xlabel('Loadings')
# No need to set y-label for the second plot as it shares the axis with the first

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the global title
plt.show()
# %% Calculate  coverage and density for all policies for the scenario box

for index, row in df_policy_vulnerabilities.iterrows():

    subset_box = df_v_box[df_v_box["policy"] == index]
    subset_full = df_full[df_full["policy"] == index]
    df_policy_vulnerabilities.loc[index, "Coverage"] = sum(
        subset_box["CO2 target not met"])/sum(subset_full["CO2 target not met"])
    df_policy_vulnerabilities.loc[index, "Density"] = sum(
        subset_box["CO2 target not met"])/len(subset_box["CO2 target not met"])


# %% Scatterplot of coverage and density
# Plotting the scatterplot with the specified color coding
plt.figure(figsize=(8, 6))
sns.set_style("white")
sns.scatterplot(
    data=df_policy_vulnerabilities,
    x="Coverage",
    y="Density",
    hue="Policy type",
    palette=color_coding,
    size="CO2 target not met share",
)
sns.despine()
# Annotating and emphasizing the policy named "B"
for sta_policy in df_full[df_full["Policy type"] == "STA"]["policy"].unique():
    df_sta_policy = df_policy_vulnerabilities[df_policy_vulnerabilities['policy'] == sta_policy]
    # if sta_policy == "B":
    #     plt.scatter(
    #         df_sta_policy["Coverage"],
    #         df_sta_policy["Density"],
    #         color='red',  # Or any color that makes it stand out
    #         s=100,          # Size of the marker
    #         #label='Policy B',
    #         edgecolor='black',
    #         linewidth=2,
    #         zorder=5  # Make sure the point is on top
    #     )

    plt.annotate(
        sta_policy,
        (df_sta_policy["Coverage"].values[0], df_sta_policy["Density"].values[0]),
        textcoords="offset points",
        xytext=(0, -13),
        ha='center', color=color_coding["STA"]
    )

# Show the legend
plt.legend(title='')
#plt.title('Density vs Coverage by Policy Type')
plt.legend(frameon=False)
plt.show()
# %% Feature scorings


def alphanumeric_sorter(key):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(k): return [convert(c) for c in re.split('([0-9]+)', k)]
    return alphanum_key(key)


# Prepare your data
x_raw = experiments.drop(columns=["Policy type", "policy", "model"])
sorted_columns = sorted(x_raw.columns, key=alphanumeric_sorter)
x_sorted = x_raw[sorted_columns]

# Define outcomes and calculate feature scores
keys_to_include = outcomes
y = {key: value for key, value in outcomes_xp.items() if key in keys_to_include}
fs = feature_scoring.get_feature_scores_all(x_sorted, y)

# Sort both the index and columns of the fs DataFrame using the alphanumeric sorter
fs_sorted = fs.sort_index(key=alphanumeric_sorter)
sorted_fs_columns = sorted(fs_sorted.columns, key=alphanumeric_sorter)
fs_sorted = fs_sorted[sorted_fs_columns]
print(fs_sorted.index)  # Should show the sorted feature names
print(fs_sorted.columns)  # Should show the sorted outcome names
# Now create the heatmap
sns.heatmap(fs_sorted, cmap="viridis", annot=True, fmt=".0%")


# all X and L on all outcomes
x = experiments.drop(columns=["Policy type", "policy", "model"])
keys_to_remove = ["Delta CS light vehicles", "Delta CS trucks", "Delta CS total", "Delta tax increase total"]
y = {key: value for key, value in outcomes_xp.items() if key not in keys_to_remove}

fs = feature_scoring.get_feature_scores_all(x, y)
sns.heatmap(fs, cmap="viridis", annot=True)
plt.show()

# all X and L on all outcomes
x = experiments.drop(columns=["Policy type", "policy", "model"])
y = pd.DataFrame(outcomes_xp['M1_CO2_TTW_total'] > 1.89).to_dict()

fs = feature_scoring.get_feature_scores_all(x, y)
sns.heatmap(fs, cmap="viridis", annot=True)
plt.show()

# %% plot for poster

# Define your color coding for the legend
color_coding = {
    "All levers": '#610100',
    "No transport efficiency": '#E9C2C0',

}

# Create a dictionary for the light colors
light_color_coding = {
    "All levers": 'lightgrey',
    "No transport efficiency": 'lightgrey',
}

# Save the original default size
original_figsize = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = [16, 12]

outcomes = [
    'M1_CO2_TTW_total',
    'M2_driving_cost_car',
    'M3_driving_cost_truck',
    'M4_energy_use_bio',
    'M5_energy_use_electricity'
]
metrics = ["Mean/stdev"]
policy_metrics_df2 = policy_metrics_df[(policy_metrics_df["Policy type"] == "All levers") | (
    policy_metrics_df["Policy type"] == "No transport efficiency")]
# 1. Normalize the metrics for each outcome
for metric in metrics:
    policy_metrics_df[f"sum_normalized_{metric}"] = 0  # initialize at 0
    for outcome in outcomes:
        col_name = f"{metric} {outcome}"
        policy_metrics_df2[f"normalized_{col_name}"] = (policy_metrics_df2[col_name] - policy_metrics_df2[col_name].min()) / \
            (policy_metrics_df2[col_name].max() - policy_metrics_df2[col_name].min())
        policy_metrics_df2[f"sum_normalized_{metric}"] = policy_metrics_df2[f"sum_normalized_{metric}"] + \
            policy_metrics_df2[f"normalized_{metric} {outcome}"]

# Main loop to plot each metric-uncertainty set combination
for metric in metrics:
    dynamic_outcomes = [f"{metric} {outcome}" for outcome in outcomes]

    for uncertainty_set in policy_metrics_df2["Uncertainty set"].unique():
        subset_uncertainty = policy_metrics_df2[policy_metrics_df2["Uncertainty set"] == uncertainty_set].copy()

        # Normalize data for the robustness metrics
        normalized_data = subset_uncertainty[dynamic_outcomes].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        subset_uncertainty['sum_metric'] = normalized_data.sum(axis=1)

        # Determine top 5 policies for this metric and uncertainty set, for each policy type including "STA"
        top_policies = []
        for ptype in color_coding.keys():
            ptype_policies = subset_uncertainty[subset_uncertainty["Policy type"]
                                                == ptype].nsmallest(10, 'sum_metric').index.tolist()
            top_policies.extend(ptype_policies)

        # Plotting
        color_list = [color_coding[ptype] if i in top_policies else light_color_coding[ptype]
                      for i, ptype in subset_uncertainty["Policy type"].items()]
        alpha_list = [0.8 if i in top_policies else 0.2 for i, ptype in subset_uncertainty["Policy type"].items()]
        linewidth_list = [5 if i in top_policies else 2 for i in subset_uncertainty.index]

        limits_outcomes = pd.DataFrame(
            {col: [subset_uncertainty[col].min(), subset_uncertainty[col].max()] for col in dynamic_outcomes})
        paraxes = parcoords.ParallelAxes(limits_outcomes, formatter={
                                         "maxima": ".1f", "minima": ".1f"}, fontsize=20, rot=45)
        labels = list(color_coding.keys())
        # legend_elements = [Patch(facecolor=color_coding[label], label=label) for label in labels]

        # Plotting logic remains unchanged
        for index, (i, row) in enumerate(subset_uncertainty.iterrows()):
            data = pd.DataFrame([row[dynamic_outcomes]])
            color = color_list[index]
            alpha = alpha_list[index]
            linewidth = linewidth_list[index]
            paraxes.plot(data.iloc[0], color=color, alpha=alpha, linewidth=linewidth)

        # plt.legend(handles=legend_elements, loc='upper right')
        # plt.title(f"Metric: {metric}, Uncertainty set: {uncertainty_set}")

    # Get the figure that parcoords is using
    parcoords_fig = plt.gcf()  # 'gcf' stands for 'Get Current Figure'
    # for ax in paraxes.axes:
    #     ax.set_xticklabels([])  # This removes the x-axis tick labels
    #     ax.set_yticklabels([])  #
    # Set figure size and facecolor
    parcoords_fig.set_size_inches(15, 10)
    parcoords_fig.patch.set_facecolor((1, 1, 1, 0))  # Set transparency

    # Optionally, you can add a legend if you need it
    # paraxes.legend()

    # Instead of saving 'fig', now we save 'parcoords_fig' which is the actual figure containing the plot.
    parcoords_fig.savefig("./figs/robustness metrics.png", dpi=500,
                          format="png", bbox_inches="tight", transparent=True)
    plt.show()
plt.rcParams["figure.figsize"] = original_figsize
