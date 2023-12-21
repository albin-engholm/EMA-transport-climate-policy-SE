# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:18:42 2023

@author: aengholm
"""
from ema_workbench.analysis import parcoords
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

policy_types = ["All levers", "No transport efficient society"]
# policy_types=["All levers"]#,"No transport efficient society"]
load_results = 1
load_results = 1
if load_results == 1:
    from ema_workbench import load_results
    date = "2023-12-07"
    n_scenarios = 50
    # for policy_type in policy_types:
    t1 = './output_data/'+"X_XP"+str(n_scenarios)+'_scenarios_MORDM_OE_'+date+".p"
    # =str(nfe)+'_nfe_directed_search_sequential_'+str(date.today())+'_'+str(n_scenarios)+'_scenarios'
    import pickle
    data = pickle.load(open(t1, "rb"))
    data_x = data [0][0] #Load output data for run for parameter set X
    data_xp = data [0][1] #Load output data for run for parameter set XP
    
    experiments_x = data_x[1]
    outcomes_x = data_x[0]
    experiments_xp = data_xp[1]
    outcomes_xp = data_xp[0]
    #experiments, outcomes = pickle.load(open(t1, "rb"))
    t2 = './output_data/'+str(n_scenarios) + \
    '_scenarios_MORDM_OE_'+date+"model_.p"
    model = pickle.load(open(t2, "rb"))
    # results = load_results(t1+'.tar.gz')
    # experiments=results[0]
    # outcomes=results[1]
    df_outcomes = pd.concat([pd.DataFrame(outcomes_x),pd.DataFrame(outcomes_xp)],axis=0, join="inner").reset_index(drop=True)

    import pickle
    import math
    experiments=pd.concat([experiments_x,experiments_xp],axis=0, join="outer").reset_index(drop=True)
    # DF with both experiments and outcomes
    df_full = pd.concat([experiments, df_outcomes], axis=1, join="inner")
    df_outcomes['policy'] = experiments["policy"]
    for index,row in df_full.iterrows():
        if math.isnan(row["R1_fuel_price_to_car_electrification"]):
            df_full.loc[index,"Uncertainty set"] = "X"
        else:
            df_full.loc[index,"Uncertainty set"] = "XP"
    df_full=df_full[df_full["policy"]!="Reference policy"]
#%% Visualization of reference scenario performance
df_reference_subset=df_full[(df_full["scenario"]=="Reference")&(df_full["Uncertainty set"]=="XP")]
# Step 2: Plot parallel coordinates
outcomes = model.outcomes.keys()
outcomes = ['M1_CO2_TTW_total',
        'M2_driving_cost_car',
        'M3_driving_cost_truck',
        'M4_energy_use_bio',
        'M5_energy_use_electricity']
limits_outcomes = pd.DataFrame()  # Create a dataframe for lever-based limits
for item in outcomes:
    limits_outcomes.loc[0, item] = min(df_reference_subset[item])  # Get lower bound
    limits_outcomes.loc[1, item] = max(df_reference_subset[item])  # Get upper bound
paraxes = parcoords.ParallelAxes(limits_outcomes,formatter={"maxima":".1f","minima":".1f"},fontsize=20,rot=90)

# Non-selected policies in gray

# Create a colormap for unique policy types using viridis
n_unique_policies = len(df_reference_subset['Policy type'].unique())
# Manually specify colors: Dark Plum and Dark Gold
colors = ["blue", "orange","red"]

# Plot selected policies with the manually specified colors
for idx, policy_type in enumerate(df_reference_subset['Policy type'].unique()):
    selected_data = df_reference_subset[df_reference_subset['Policy type'] == policy_type]
    paraxes.plot(selected_data, label=f'Policy type: {policy_type}', color=colors[idx])

# Get the figure that parcoords is using
parcoords_fig = plt.gcf()  # 'gcf' stands for 'Get Current Figure'
# for ax in paraxes.axes:
#     ax.set_xticklabels([])  # This removes the x-axis tick labels
#     ax.set_yticklabels([])  #
# Set figure size and facecolor
parcoords_fig.set_size_inches(10, 10)
#parcoords_fig.patch.set_facecolor((1, 1, 1, 0))  # Set transparency

# Optionally, you can add a legend if you need it
paraxes.legend()

# Instead of saving 'fig', now we save 'parcoords_fig' which is the actual figure containing the plot.
parcoords_fig.savefig("parcoords_candidate_policies_reference_outcomes.png", dpi=300, format="png", bbox_inches="tight", transparent=True)
# Pairplot outcomes on outcomes
sns.pairplot(data=df_reference_subset,x_vars=outcomes,y_vars=outcomes,hue="Policy type", palette = colors, diag_kws={"common_norm":False})
    
# The same plot but over the levers
levers = model.levers.keys()

limits_levers = pd.DataFrame()  # Create a dataframe for lever-based limits
for item in levers:
    limits_levers.loc[0, item] = min(df_reference_subset[item])  # Get lower bound
    limits_levers.loc[1, item] = max(df_reference_subset[item])  # Get upper bound
paraxes = parcoords.ParallelAxes(limits_levers,formatter={"maxima":".1f","minima":".1f"},fontsize=20,rot=90)

# Non-selected policies in gray

# Create a colormap for unique policy types using viridis


# Plot selected policies with the manually specified colors
for idx, policy_type in enumerate(df_reference_subset['Policy type'].unique()):
    selected_data = df_reference_subset[df_reference_subset['Policy type'] == policy_type]
    paraxes.plot(selected_data, label=f'Policy type: {policy_type}', color=colors[idx])

# Get the figure that parcoords is using
parcoords_fig = plt.gcf()  # 'gcf' stands for 'Get Current Figure'
# for ax in paraxes.axes:
#     ax.set_xticklabels([])  # This removes the x-axis tick labels
#     ax.set_yticklabels([])  #
# Set figure size and facecolor
parcoords_fig.set_size_inches(10, 10)
#parcoords_fig.patch.set_facecolor((1, 1, 1, 0))  # Set transparency

# Optionally, you can add a legend if you need it
paraxes.legend()

# Instead of saving 'fig', now we save 'parcoords_fig' which is the actual figure containing the plot.
parcoords_fig.savefig("parcoords_candidate_policies_reference_levers.png", dpi=300, format="png", bbox_inches="tight", transparent=True)

# If you want to show the plot, you would now use
plt.show()  
# Pairplot levers on levers
sns.pairplot(data=df_reference_subset,x_vars=outcomes,y_vars=outcomes,hue="Policy type", palette = colors, diag_kws={"common_norm":False})

for outcome in outcomes:

    # Plot both distributions
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    sns.histplot(df_reference_subset[outcome], kde=True, ax=ax2,
                 label="Reference scenario, right axis", color='red')
    ax2.set_ylabel("Reference", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.axvline(df_reference_subset[outcome].quantile(0.5), color="red")

    sns.histplot(df_full[outcome], kde=True, ax=ax1,
                 label="All scenarios, left axis", color='blue')
    ax1.set_ylabel('All scenarios', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    plt.axvline(df_full[outcome].quantile(0.5), color="blue")
    plt.axvline(df_full[outcome].quantile(0.9), color="blue",linestyle="--")

    fig.legend(loc='upper right')

    plt.show()
# %% Calculate robustness metrics
import statistics

import pandas as pd

# Calculate Zero Regret
outcomes_M = ['M1_CO2_TTW_total', 'M2_driving_cost_car', 'M3_driving_cost_truck', 'M4_energy_use_bio', 'M5_energy_use_electricity']

results = []

grouped = df_full.groupby(['Uncertainty set', 'scenario'])

for (uncertainty_set, scenario), group in grouped:
    for outcome in outcomes_M:
        zero_regret_value = group[outcome].min()
        
        results.append({
            "Outcome": outcome,
            "Uncertainty set": uncertainty_set,
            "scenario": scenario,
            "Zero regret": zero_regret_value
        })

zero_regret_df = pd.DataFrame(results)

# Now, calculate metrics for each policy
all_policies = df_full["policy"].unique()
all_uncertainty_sets = df_full["Uncertainty set"].unique()

metrics_data = []

for policy in all_policies:
    if policy != "Reference policy":
        for uncertainty_set in all_uncertainty_sets:
            df_temp = df_full[(df_full["policy"] == policy) & (df_full["Uncertainty set"] == uncertainty_set)]

            policy_data = {
                "policy": policy,
                "Policy type": df_temp["Policy type"].iloc[0],
                "Uncertainty set": uncertainty_set
            }

            for outcome in outcomes_M:
                outcome_data = df_temp[outcome]
                ref_outcome_data = df_temp[df_temp["scenario"] == "Reference"][outcome]

                policy_data.update({
                    f"90th percentile {outcome}": outcome_data.quantile(0.9),
                    f"Reference {outcome}": ref_outcome_data.values[0],
                    f"90_percentile_deviation {outcome}": (outcome_data.quantile(0.9) - ref_outcome_data.values[0]) / abs(ref_outcome_data.values[0]),
                    f"Max {outcome}": outcome_data.max(),
                    f"Mean {outcome}": outcome_data.mean(),
                    f"Standard deviation {outcome}": outcome_data.std(),
                    f"Mean/stdev {outcome}": (outcome_data.mean() + 1) * (outcome_data.std() + 1),
                })

                regrets = abs( outcome_data - zero_regret_df.loc[
                    (zero_regret_df["Uncertainty set"] == uncertainty_set) &
                    (zero_regret_df["Outcome"] == outcome) &
                    zero_regret_df["scenario"].isin(df_temp["scenario"].unique()), 
                    "Zero regret"
                ].values)
                
                policy_data[f"Max_regret {outcome}"] = regrets.max()

            metrics_data.append(policy_data)

policy_metrics_df = pd.DataFrame(metrics_data)

#%% visualize metrics

# Set the aesthetics for better visibility
sns.set_style("whitegrid")

outcomes_M = ['M1_CO2_TTW_total', 'M2_driving_cost_car', 'M3_driving_cost_truck', 'M4_energy_use_bio', 'M5_energy_use_electricity']
all_uncertainty_sets = policy_metrics_df["Uncertainty set"].unique()

# Loop over each outcome and uncertainty set combination
for outcome in outcomes_M:
    for uncertainty_set in all_uncertainty_sets:
        # Filter the data
        subset_df = policy_metrics_df[policy_metrics_df["Uncertainty set"] == uncertainty_set]
        
        # Metrics to visualize in their original form (without newlines)
        metrics = [
            f"90_percentile_deviation {outcome}", 
            f"Max_regret {outcome}", 
            f"Mean/stdev {outcome}"
        ]
        
        # Create the pairplot
        plot = sns.pairplot(data=subset_df, vars=metrics, diag_kind="kde", hue="Policy type", height=2.5,palette=colors,diag_kws={"common_norm":False})
        plot.fig.suptitle(f"Outcome: {outcome}, Uncertainty Set: {uncertainty_set}", y=1.02)
        
        # Adjust x-labels and y-labels for better readability
        for ax in plot.axes.flat:
            ax.set_xlabel(ax.get_xlabel().replace(" ", "\n"), rotation=0)
            ax.set_ylabel(ax.get_ylabel().replace(" ", "\n"), rotation=90)  # If y-labels are still overlapping, try using 45 here too.
        plot._legend.set_bbox_to_anchor((1.3, 0.5))  # Position of legend relative to axes bounding box (x: 1 is right outside of the plot, y: 0.5 is centered)
        plt.tight_layout()
        plt.show()


#%%
#sns.pairplot(df_full,hue="Policy type",x_vars=list(model.levers.keys()),y_vars=list(model.uncertainties.keys()))
# %% Parcoords of percentage deviation  metrics
from matplotlib.patches import Patch

# Define your color coding for the legend
color_coding = {"All levers": 'blue', "No transport efficient society": 'r', "Transport efficient society not realized": 'g', "Trv":'black'}

# Get the limits from lever ranges
outcomes = ['90_percentile_deviation M1_CO2_TTW_total',
       '90_percentile_deviation M2_driving_cost_car', '90_percentile_deviation M3_driving_cost_truck',
       '90_percentile_deviation M4_energy_use_bio', '90_percentile_deviation M5_energy_use_electricity']  # Get outcomes


# Save the original default size
original_figsize = plt.rcParams["figure.figsize"]

# Set the figure size you want
plt.rcParams["figure.figsize"] = [16, 12]

for uncertainty_set in policy_metrics_df["Uncertainty set"].unique():
    policy_metrics_subset = policy_metrics_df[policy_metrics_df["Uncertainty set"]==uncertainty_set]

    color_list = [color_coding[pt] for pt in policy_metrics_subset["Policy type"]]

    limits_outcomes = pd.DataFrame()  # Create a dataframe for lever-based limits
    for item in outcomes:
        limits_outcomes.loc[0, item] = min(policy_metrics_subset[item])  # Get lower bound
        limits_outcomes.loc[1, item] = max(policy_metrics_subset[item])  # Get upper bound

    limits = limits_outcomes

    # Create the parallel coordinates 
    paraxes = parcoords.ParallelAxes(limits)

    # Define your color coding for the legend
    labels = list(color_coding.keys())
    legend_elements = [Patch(facecolor=color_coding[label], label=label) for label in labels]

    count = 0
    # Loop over rows in dataframe
    for i, row in policy_metrics_subset.iterrows():
        data = row[['90_percentile_deviation M1_CO2_TTW_total',
                    '90_percentile_deviation M2_driving_cost_car', 
                    '90_percentile_deviation M3_driving_cost_truck',
                    '90_percentile_deviation M4_energy_use_bio', 
                    '90_percentile_deviation M5_energy_use_electricity']]
        print(policy_metrics_subset.loc[i,"Policy type"])
        color = color_list[count]  # Corresponding color from color list
        paraxes.plot(data, color=color)
        count=count+1

    # Add legend manually
    plt.legend(handles=legend_elements, loc='upper right')

    # Show the plot
    plt.title(f"Uncertainty set: {uncertainty_set}")
    plt.show()

plt.rcParams["figure.figsize"] = original_figsize

#%% Parcoords of all policies
from matplotlib.patches import Patch

# Define your color coding for the legend
color_coding = {"All levers": 'blue', "No transport efficient society": 'orange', "Trv":'red'}

# Save the original default size
original_figsize = plt.rcParams["figure.figsize"]

# Set the figure size you want
plt.rcParams["figure.figsize"] = [10, 10]

metrics = [
    "90_percentile_deviation", 
    "Max_regret", 
    "Mean/stdev"
]

for metric in metrics:
    dynamic_outcomes = [f"{metric} {outcome}" for outcome in ['M1_CO2_TTW_total',
                                                              'M2_driving_cost_car',
                                                              'M3_driving_cost_truck',
                                                              'M4_energy_use_bio',
                                                              'M5_energy_use_electricity']]  # Dynamic outcomes
    
    #for uncertainty_set in policy_metrics_df["Uncertainty set"].unique():
    for uncertainty_set in ["XP"]:
        policy_metrics_subset = policy_metrics_df[policy_metrics_df["Uncertainty set"] == uncertainty_set]
        
        color_list = [color_coding[pt] for pt in policy_metrics_subset["Policy type"]]

        limits_outcomes = pd.DataFrame()  # Create a dataframe for limits
        for item in dynamic_outcomes:
            limits_outcomes.loc[0, item] = min(policy_metrics_subset[item])  # Get lower bound
            limits_outcomes.loc[1, item] = max(policy_metrics_subset[item])  # Get upper bound

        limits = limits_outcomes

        # Create the parallel coordinates 
        paraxes = parcoords.ParallelAxes(limits)

        # Define your color coding for the legend
        labels = list(color_coding.keys())
        legend_elements = [Patch(facecolor=color_coding[label], label=label) for label in labels]

        count = 0
        # Loop over rows in dataframe
        for i, row in policy_metrics_subset.iterrows():
            data = row[dynamic_outcomes]
            print(policy_metrics_subset.loc[i, "Policy type"])
            color = color_list[count]  # Corresponding color from color list
            paraxes.plot(data, color=color)
            count += 1

        # Add legend manually
        plt.legend(handles=legend_elements, loc='upper right')

        # Show the plot
        plt.title(f"Metric: {metric}, Uncertainty set: {uncertainty_set}")
        plt.show()

plt.rcParams["figure.figsize"] = original_figsize
#%% Parcoords of all metrics
from matplotlib.patches import Patch
import pandas as pd
import matplotlib.pyplot as plt
from ema_workbench.analysis import parcoords

# Define your color coding for the legend
color_coding = {
    "All levers": 'blue',
    "No transport efficient society": 'orange',
    "Trv": 'red'
}

# Create a dictionary for the light colors
light_color_coding = {
    "All levers": 'blue',
    "No transport efficient society": 'yellow',
    "Trv": 'pink'
}

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
metrics = ["90_percentile_deviation", "Max_regret", "Mean/stdev"]

# 1. Normalize the metrics for each outcome
for metric in metrics:
    policy_metrics_df[f"sum_normalized_{metric}"] = 0 #initialize at 0
    for outcome in outcomes:
        col_name = f"{metric} {outcome}"
        policy_metrics_df[f"normalized_{col_name}"] = (policy_metrics_df[col_name] - policy_metrics_df[col_name].min()) / \
                                                     (policy_metrics_df[col_name].max() - policy_metrics_df[col_name].min())
        policy_metrics_df[f"sum_normalized_{metric}"] =  policy_metrics_df[f"sum_normalized_{metric}"] + policy_metrics_df[f"normalized_{metric} {outcome}"]                                        

# Main loop to plot each metric-uncertainty set combination
for metric in metrics:
    dynamic_outcomes = [f"{metric} {outcome}" for outcome in outcomes]

    #for uncertainty_set in policy_metrics_df["Uncertainty set"].unique():
    for uncertainty_set in ["XP"]:
        subset_uncertainty = policy_metrics_df[policy_metrics_df["Uncertainty set"] == uncertainty_set].copy()
        
        # Normalize data for the robustness metrics
        normalized_data = subset_uncertainty[dynamic_outcomes].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        subset_uncertainty['sum_metric'] = normalized_data.sum(axis=1)

        # Determine top 5 policies for this metric and uncertainty set, for each policy type, excluding "Trv"
        top_policies = []
        for ptype in color_coding.keys():
            if ptype!= "Trv":
                ptype_policies = subset_uncertainty[subset_uncertainty["Policy type"] == ptype].nsmallest(5, 'sum_metric').index.tolist()
                top_policies.extend(ptype_policies)
            else:
                ptype_policies = subset_uncertainty[subset_uncertainty["Policy type"] == ptype].index.tolist()
                top_policies.extend(ptype_policies)

        # Plotting
        color_list = [color_coding[ptype] if i in top_policies else light_color_coding[ptype] for i, ptype in subset_uncertainty["Policy type"].items()]
        alpha_list = [0.9 if i in top_policies else 0.1 for i, ptype in subset_uncertainty["Policy type"].items()]
        linewidth_list = [2 if i in top_policies else 1 for i in subset_uncertainty.index]
        zorder_list = [10 if i in top_policies else 1 for i in subset_uncertainty.index]

        limits_outcomes = pd.DataFrame({col: [subset_uncertainty[col].min(), subset_uncertainty[col].max()] for col in dynamic_outcomes})
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

        plt.legend(handles=legend_elements, loc='upper right')
        #plt.title(f"Metric: {metric}, Uncertainty set: {uncertainty_set}")
        plt.title(f"Metric: {metric}")
        plt.show()

plt.rcParams["figure.figsize"] = original_figsize
#%% plot for poster
from matplotlib.patches import Patch
import pandas as pd
import matplotlib.pyplot as plt
from ema_workbench.analysis import parcoords

# Define your color coding for the legend
color_coding = {
    "All levers": '#610100',
    "No transport efficient society": '#E9C2C0',

}

# Create a dictionary for the light colors
light_color_coding = {
    "All levers": 'lightgrey',
    "No transport efficient society": 'lightgrey',
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
policy_metrics_df2=policy_metrics_df[(policy_metrics_df["Policy type"]=="All levers") |(policy_metrics_df["Policy type"]=="No transport efficient society") ]
# 1. Normalize the metrics for each outcome
for metric in metrics:
    policy_metrics_df[f"sum_normalized_{metric}"] = 0 #initialize at 0
    for outcome in outcomes:
        col_name = f"{metric} {outcome}"
        policy_metrics_df2[f"normalized_{col_name}"] = (policy_metrics_df2[col_name] - policy_metrics_df2[col_name].min()) / \
                                                     (policy_metrics_df2[col_name].max() - policy_metrics_df2[col_name].min())
        policy_metrics_df2[f"sum_normalized_{metric}"] =  policy_metrics_df2[f"sum_normalized_{metric}"] + policy_metrics_df2[f"normalized_{metric} {outcome}"]                                        

# Main loop to plot each metric-uncertainty set combination
for metric in metrics:
    dynamic_outcomes = [f"{metric} {outcome}" for outcome in outcomes]

    for uncertainty_set in policy_metrics_df2["Uncertainty set"].unique():
        subset_uncertainty = policy_metrics_df2[policy_metrics_df2["Uncertainty set"] == uncertainty_set].copy()
        
        # Normalize data for the robustness metrics
        normalized_data = subset_uncertainty[dynamic_outcomes].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        subset_uncertainty['sum_metric'] = normalized_data.sum(axis=1)

        # Determine top 5 policies for this metric and uncertainty set, for each policy type including "Trv"
        top_policies = []
        for ptype in color_coding.keys():
            ptype_policies = subset_uncertainty[subset_uncertainty["Policy type"] == ptype].nsmallest(10, 'sum_metric').index.tolist()
            top_policies.extend(ptype_policies)

        # Plotting
        color_list = [color_coding[ptype] if i in top_policies else light_color_coding[ptype] for i, ptype in subset_uncertainty["Policy type"].items()]
        alpha_list = [0.8 if i in top_policies else 0.2 for i, ptype in subset_uncertainty["Policy type"].items()]
        linewidth_list = [5 if i in top_policies else 2 for i in subset_uncertainty.index]

        limits_outcomes = pd.DataFrame({col: [subset_uncertainty[col].min(), subset_uncertainty[col].max()] for col in dynamic_outcomes})
        paraxes = parcoords.ParallelAxes(limits_outcomes,formatter={"maxima":".1f","minima":".1f"},fontsize=20,rot=45)
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
    parcoords_fig.savefig("robustness metrics.png", dpi=500, format="png", bbox_inches="tight", transparent=True)
    plt.show()
plt.rcParams["figure.figsize"] = original_figsize


#%% Vulnerability analysis
#Policies selected for analysis
vulnerability_policies=[5, 103,"B"]

# Parcoords of vulnerability policies
paraxes = parcoords.ParallelAxes(limits_levers,formatter={"maxima":".1f","minima":".1f"},fontsize=20,rot=90)
# Plot selected policies with the manually specified colors
for idx, policy in enumerate (vulnerability_policies):
    selected_data = df_full[df_full['policy'] == str(policy)].iloc[0]
    policy_type=selected_data["Policy type"]
    paraxes.plot(selected_data, label=f'Policy type: {policy_type}', color=colors[idx])

# Get the figure that parcoords is using
parcoords_fig = plt.gcf()  # 'gcf' stands for 'Get Current Figure'
# for ax in paraxes.axes:
#     ax.set_xticklabels([])  # This removes the x-axis tick labels
#     ax.set_yticklabels([])  #
# Set figure size and facecolor
parcoords_fig.set_size_inches(10, 10)
#parcoords_fig.patch.set_facecolor((1, 1, 1, 0))  # Set transparency

# Optionally, you can add a legend if you need it
paraxes.legend()
#%%
vulnerability_analysis_policy="B" # #99 ALl levers, #183 no transport efficiency "B" TRV
df_v=df_full[df_full["policy"].astype(str)==str(vulnerability_analysis_policy)]
df_v=df_v[df_v["scenario"]!="Reference"]
#Perform PRIM analysis
from ema_workbench.analysis import prim
#Set up PRIM
# Reset the index to ensure alignment
df_v = df_v.reset_index(drop=True)

# Set up PRIM with aligned indices
x = df_v[model.uncertainties.keys()]
y = df_v['M1_CO2_TTW_total'] > 1.89
prim_alg = prim.Prim(x, y, threshold=0.5)


prim_alg = prim.Prim(x, y, threshold=0.5)
#prim_alg = prim.Prim(x, y, threshold=0.5)
#Find 1st box
box1 = prim_alg.find_box()
#
#Visualizations of Box1
box1.show_tradeoff()

for i in range(0,len(box1.peeling_trajectory.T.columns)):
    s=box1.peeling_trajectory.T[i].id
    if (i%2)==0:
        plt.text(box1.peeling_trajectory.T[i].coverage,box1.peeling_trajectory.T[i].density +.02,s,fontsize=8)
    else:
        plt.text(box1.peeling_trajectory.T[i].coverage,box1.peeling_trajectory.T[i].density +.02 ,s,fontsize=8)
plt.show()
#%%
#Choose point for inspection
i1=round((len(box1.peeling_trajectory.T.columns)-1))
#or choose box manually
i1=19
box1.inspect(i1)
box1.inspect(i1, style='graph')
plt.show()
#%%
box1.show_ppt()
# Save the original palette
original_palette = sns.color_palette()
custom_palette = ['green', 'red'] 
sns.set_palette(sns.color_palette(custom_palette)) 
ax=box1.show_pairs_scatter(i1)
for row in ax.axes:
    for subplot in row:
        subplot.set_facecolor('none')  # For the subplot background
        subplot.grid(False)  # To remove the grid lines
        for spine in subplot.spines.values():
            spine.set_edgecolor('none')  # For the edges of the subplot
plt.savefig('test_prim_box.png', transparent=True)
#%% Comparison of what policies are selected across uncertainty sets and metrics 

for uncertainty in policy_metrics_df["Uncertainty set"].unique():
    print(f"Uncertainty Set: {uncertainty}")
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            if i < j:
                common_policies = set(highlighted_policies_df.loc[(metric1, uncertainty), 'top_policies'][0]) & set(highlighted_policies_df.loc[(metric2, uncertainty), 'top_policies'][0])
                print(f"Common top policies between {metric1} and {metric2}: {len(common_policies)}")
for metric in metrics:
    print(f"Robustness Metric: {metric}")
    for i, uncertainty1 in enumerate(policy_metrics_df["Uncertainty set"].unique()):
        for j, uncertainty2 in enumerate(policy_metrics_df["Uncertainty set"].unique()):
            if i < j:
                common_policies = set(highlighted_policies_df.loc[(metric, uncertainty1), 'top_policies'][0]) & set(highlighted_policies_df.loc[(metric, uncertainty2), 'top_policies'][0])
                print(f"Common top policies between {uncertainty1} and {uncertainty2}: {len(common_policies)}")

from matplotlib_venn import venn2
import matplotlib.pyplot as plt

def visualize_intersection(uncertainty, metric1, metric2):
    # Fetch the top policies for both metrics under a given uncertainty
    policies_metric1 = set(highlighted_policies_df.loc[(metric1, uncertainty), 'top_policies'][0])
    policies_metric2 = set(highlighted_policies_df.loc[(metric2, uncertainty), 'top_policies'][0])
    
    # Plot Venn Diagram
    venn2([policies_metric1, policies_metric2], (metric1, metric2))
    plt.title(f"Overlap of top policies for Uncertainty Set: {uncertainty}")
    plt.show()

# As an example, visualize for 'X' uncertainty and first two metrics
uncertainty = 'X'  # Replace with the desired uncertainty set
metric1 = metrics[0]
metric2 = metrics[1]

visualize_intersection(uncertainty, metric1, metric2)


from matplotlib_venn import venn3
import matplotlib.pyplot as plt

def visualize_intersection_three_metrics(uncertainty, metric1, metric2, metric3):
    # Fetch the top policies for the three metrics under a given uncertainty
    policies_metric1 = set(highlighted_policies_df.loc[(metric1, uncertainty), 'top_policies'][0])
    policies_metric2 = set(highlighted_policies_df.loc[(metric2, uncertainty), 'top_policies'][0])
    policies_metric3 = set(highlighted_policies_df.loc[(metric3, uncertainty), 'top_policies'][0])
    
    # Plot Venn Diagram
    venn3([policies_metric1, policies_metric2, policies_metric3], (metric1, metric2, metric3))
    plt.title(f"Overlap of top policies for Uncertainty Set: {uncertainty}")
    plt.show()

# Now, visualize for 'X' uncertainty and all three metrics
uncertainty = 'X'  # Replace with the desired uncertainty set
metric1 = metrics[0]
metric2 = metrics[1]
metric3 = metrics[2]
#%%
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

def visualize_intersection_three_metrics_for_policy_type(uncertainty, metric1, metric2, metric3, policy_type):
    
    # Filter policy_metrics_df based on given policy type and uncertainty
    filtered_df = policy_metrics_df[policy_metrics_df['Policy type'] == policy_type]
    
    # Extract the top policies for each metric
    policies_metric1 = set(highlighted_policies_df.loc[(metric1, uncertainty), 'top_policies'][0])
    policies_metric2 = set(highlighted_policies_df.loc[(metric2, uncertainty), 'top_policies'][0])
    policies_metric3 = set(highlighted_policies_df.loc[(metric3, uncertainty), 'top_policies'][0])
    
    # Filter these policy sets based on the ones available in our filtered_df
    policies_metric1 = policies_metric1.intersection(set(filtered_df.index))
    policies_metric2 = policies_metric2.intersection(set(filtered_df.index))
    policies_metric3 = policies_metric3.intersection(set(filtered_df.index))
    
    # Plot Venn Diagram
    venn3([policies_metric1, policies_metric2, policies_metric3], (metric1, metric2, metric3))
    plt.title(f"Overlap of top {policy_type} policies for Uncertainty Set: {uncertainty}")
    plt.show()

# Looping over each policy type
for policy_type in color_coding.keys():
    print(f"Policy Type: {policy_type}")
    visualize_intersection_three_metrics_for_policy_type(uncertainty, metric1, metric2, metric3, policy_type)

#%%
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

def visualize_intersection_three_metrics_for_policy_type(uncertainty, metric1, metric2, metric3, policy_type, overlapped_policies):
    
    # Filter policy_metrics_df based on given policy type and uncertainty, but exclude 'Trv'
    filtered_df = policy_metrics_df[(policy_metrics_df['Policy type'] == policy_type) & (policy_metrics_df['Policy type'] != 'Trv')]
    
    # Extract the top policies for each metric
    policies_metric1 = set(highlighted_policies_df.loc[(metric1, uncertainty), 'top_policies'][0])
    policies_metric2 = set(highlighted_policies_df.loc[(metric2, uncertainty), 'top_policies'][0])
    policies_metric3 = set(highlighted_policies_df.loc[(metric3, uncertainty), 'top_policies'][0])
    
    # Filter these policy sets based on the ones available in our filtered_df
    policies_metric1 = policies_metric1.intersection(set(filtered_df.index))
    policies_metric2 = policies_metric2.intersection(set(filtered_df.index))
    policies_metric3 = policies_metric3.intersection(set(filtered_df.index))
    
    # Plot Venn Diagram
    v = venn3([policies_metric1, policies_metric2, policies_metric3], (metric1, metric2, metric3))
    plt.title(f"Overlap of top {policy_type} policies for Uncertainty Set: {uncertainty}")
    plt.show()
    
    # Check if all three metrics agree on the top policies
    if v.get_label_by_id('111'):  # '111' indicates overlap of all three sets
        overlapped_policies.extend(list(policies_metric1 & policies_metric2 & policies_metric3))

overlapped_policies = []

# Looping over each policy type
for policy_type in color_coding.keys():
    if policy_type != 'Trv':
        print(f"Policy Type: {policy_type}")
        visualize_intersection_three_metrics_for_policy_type(uncertainty, metric1, metric2, metric3, policy_type, overlapped_policies)

print("Policies with complete overlap across all metrics:")
print(overlapped_policies)

#%% Parcoords of policies


from ema_workbench.analysis import parcoords
import matplotlib.pyplot as plt

# Get the limits from lever ranges
levers = ['L1_bio_share_diesel', 'L2_bio_share_gasoline', 'L3_additional_car_energy_efficiency', 'L4_additional_truck_energy_efficiency', 'L5_fuel_tax_increase_gasoline', 'L6_fuel_tax_increase_diesel', 'L7_km_tax_cars', 'L8_km_tax_trucks', 'L9_transport_efficient_planning_cars', 'L10_transport_efficient_planning_trucks']  # Get levers
limits_levers = pd.DataFrame()  # Create a dataframe for lever-based limits
for item in levers:
    limits_levers.loc[0, item] = min(df_full[item])  # Get lower bound
    limits_levers.loc[1, item] = max(df_full[item])  # Get upper bound

limits = limits_levers

# Filter the data for the specific policy type
final_policies = policy_metrics_df.loc[overlapped_policies]["policy"] # as Policy ID
filtered_data = df_full[df_full["policy"].isin(final_policies)]
filtered_data = PDf

# Create the parallel coordinates plot for the filtered data
paraxes = parcoords.ParallelAxes(limits_levers)
paraxes.plot(filtered_data)

# Set the title to the policy type


# Show the plot
plt.show()