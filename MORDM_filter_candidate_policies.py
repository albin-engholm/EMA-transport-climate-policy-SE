# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:01:40 2023

@author: aengholm
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:18:42 2023

@author: aengholm
"""
from ema_workbench import (RealParameter, CategoricalParameter,
                           ScalarOutcome, ema_logging,
                           perform_experiments, Constant)

from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)
    # %% Load candidate policies and model from previous optimization
    import pandas as pd
    df_full = pd.DataFrame()
    policy_types = ["All levers", "No transport efficient society"]
    policy_types=["All levers"]#,"No transport efficient society"]
    load_results = 1
    if load_results == 1:
        date = "2023-12-03"
        nfe = 200000
        count = 0
        for policy_type in policy_types:
            if count == 0:
                t1 = './output_data/'+policy_type + \
                    str(nfe)+"_nfe_"+"directed_search_MORDM_"+date+".p"
               # =str(nfe)+'_nfe_directed_search_sequential_'+str(date.today())+'_'+str(n_scenarios)+'_scenarios'
                import pickle
                results_list, convergence, scenarios, epsilons = pickle.load(
                    open(t1, "rb"))
                t2 = './output_data/'+policy_type + \
                    str(nfe)+"_nfe_"+"directed_search_MORDM_"+date+"model_.p"
                model = pickle.load(open(t2, "rb"))
                scenario_count = 0
                for results in results_list:
                    results["Policy type"] = policy_type
                    df_full = pd.concat([df_full, results], ignore_index=True)
                    scenario_count = scenario_count+1

            if count == 1:
                date = "2023-11-14"
                nfe = 150000
                t1 = './output_data/'+policy_type + \
                    str(nfe)+"_nfe_"+"directed_search_MORDM_"+date+".p"
               # =str(nfe)+'_nfe_directed_search_sequential_'+str(date.today())+'_'+str(n_scenarios)+'_scenarios'
                import pickle
                results_list, convergence, scenarios,epsilons = pickle.load(
                    open(t1, "rb"))
                t2 = './output_data/'+policy_type + \
                    str(nfe)+"_nfe_"+"directed_search_MORDM_"+date+"model_.p"
                model = pickle.load(open(t2, "rb"))
                for results in results_list:
                    results["Policy type"] = policy_type
                    df_full = pd.concat([df_full, results], ignore_index=True)
            count = count+1

    # The model object already contains all information about levers and uncertainties, so no need to specify these things again
    df_full_sample=df_full.sample(100)

#%%   Plot policies against policy levers
    from ema_workbench.analysis import parcoords
    import matplotlib.pyplot as plt

    # Get the limits from lever ranges
    levers = ['L1_bio_share_diesel', 'L2_bio_share_gasoline', 'L3_additional_car_energy_efficiency', 'L4_additional_truck_energy_efficiency', 'L5_fuel_tax_increase_gasoline', 'L6_fuel_tax_increase_diesel', 'L7_km_tax_cars', 'L8_km_tax_trucks', 'L9_transport_efficient_planning_cars', 'L10_transport_efficient_planning_trucks']  # Get levers
    limits_levers = pd.DataFrame()  # Create a dataframe for lever-based limits
    for item in levers:
        limits_levers.loc[0, item] = min(df_full[item])  # Get lower bound
        limits_levers.loc[1, item] = max(df_full[item])  # Get upper bound
    
    limits = limits_levers
    

    # List of unique policy types
    policy_types = df_full_sample['Policy type'].unique()
    
    for policy_type in policy_types:
        # Filter the data for the specific policy type
        filtered_data = df_full_sample[df_full_sample['Policy type'] == policy_type]
        
        # Create the parallel coordinates plot for the filtered data
        paraxes = parcoords.ParallelAxes(limits_levers)
        paraxes.plot(filtered_data)
        
        # Set the title to the policy type
        plt.title(f"Policy type: {policy_type}")
        
        # Show the plot
        plt.show()
#%%   Plot policies against outcomes 
    # Get the limits from lever ranges
    outcomes=[]
    for outcome in model.outcomes:
        if outcome.kind != 0: #0 means INFO
            outcomes.append(outcome.name)
    limits_outcomes = pd.DataFrame()  # Create a dataframe for lever-based limits
    for item in outcomes:
        limits_outcomes.loc[0, item] = min(df_full[item])  # Get lower bound
        limits_outcomes.loc[1, item] = max(df_full[item])  # Get upper bound
    
    limits = limits_outcomes
    
    # Create the parallel coordinates plot
    paraxes = parcoords.ParallelAxes(limits_outcomes)
    count = 0
    
    paraxes.plot(df_full_sample)
    
    # Set the legend and show the plot
    plt.show()
    
    #%% pairplots of all policies
    import seaborn as sns
    pairplot_kws={"alpha":0.7,"s":0.5}
    #levers on levers
   # sns.pairplot(data=df_full,x_vars=model.levers.keys(),y_vars=model.levers.keys(),hue="Policy type",plot_kws=pairplot_kws)
    
    #outcomes on outcomes
    sns.pairplot(data=df_full,x_vars=outcomes,y_vars=outcomes,hue="Policy type",plot_kws=pairplot_kws)
    
    #levers on outcomes
    sns.pairplot(data=df_full,x_vars=model.levers.keys(),y_vars=outcomes,hue="Policy type",plot_kws=pairplot_kws)         

#%% Create a dataframe only with policies that meet the climate target
    CO2_target=0.1*18.9 #90% reduction  compared to 2010
    df_CO2 = df_full[df_full["M1_CO2_TTW_total"] < CO2_target].copy()
    df_full["CO2 target met"] = df_full["M1_CO2_TTW_total"] < CO2_target

#%% Compute policy scores
    # Required imports
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    
    
    # List of metrics we are interested in
    outcomes_filter = [
        'M2_driving_cost_car', 
        'M3_driving_cost_truck', 
        'M4_energy_use_bio',
        'M5_energy_use_electricity'
    ]
    
    # Score 1: Sum of the scaled values of metrics M2-M5
    
    # Filter out the metrics of interest from the policies that reach climate targets
    df_M = df_CO2[outcomes_filter]
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Scale the data (fit and transform)
    df_M = pd.DataFrame(scaler.fit_transform(df_M), columns=df_M.columns, index=df_CO2.index)
    
    # Assign the sum of scaled values as Score 1
    df_CO2["Score 1"] = df_M.sum(axis=1)
    
    # Score 2: Identify policies where all metrics are below a given percentile for each policy type
    
    # Define percentiles ranging from 0.05 to 0.9, in steps of 0.05
    percentiles = np.arange(0.05, 0.95, 0.05).tolist()
    percentiles = [round(x, 2) for x in percentiles]
    #print(percentiles)

    # Prepare a results dictionary to store counts and indices of policies below each percentile for each policy type
    results = {
        ptype: {
            'count': {p: 0 for p in percentiles},
            'indices': {p: [] for p in percentiles}
        } for ptype in df_CO2["Policy type"].unique()
    }
    
    # Group data by policy type and compute results for each policy type and percentile
    for policy_type, group in df_M[df_CO2["Policy type"].notna()].groupby(df_CO2["Policy type"]):
        
        for i in percentiles:
            # Compute percentiles specific to the current policy type
            percentile = group.quantile(i)
            
            # Identify rows that fall below the percentile values
            below_percentile = (
                (group['M2_driving_cost_car'] < percentile['M2_driving_cost_car']) &
                (group['M3_driving_cost_truck'] < percentile['M3_driving_cost_truck']) &
                (group['M4_energy_use_bio'] < percentile['M4_energy_use_bio']) &
                (group['M5_energy_use_electricity'] < percentile['M5_energy_use_electricity'])
            )
            
            # Store the count and indices of policies below the percentile
            results[policy_type]['count'][i] = below_percentile.sum()
            results[policy_type]['indices'][i] = group[below_percentile].index.tolist()
    
    # Convert the results dictionary to a multi-index DataFrame for easier referencing
    df_percentile_results = pd.DataFrame.from_dict({
        (policy_type, measure): results[policy_type][measure]
        for policy_type in results.keys()
        for measure in results[policy_type].keys()
    })
    
    # Assigning "Score 2" based on specific percentile (e.g., 0.55) and policy type
    specific_percentile = 0.75
    df_CO2["Score 2"] = False  # Initialize the "Score 2" column
    
    for policy_type in df_CO2["Policy type"].unique():
        # Fetch indices of policies below the specific percentile for the current policy type
        indices_below_percentile = df_percentile_results.loc[specific_percentile, (policy_type, 'indices')]
        
        # Update the "Score 2" column for the identified policies
        df_CO2.loc[indices_below_percentile, "Score 2"] = True
        


    # Number of desired top policies for each policy type
    n = 100  # Change this to the desired number
    
    # Step 1: Filter rows where Score 2 is True
    filtered_df = df_CO2[df_CO2["Score 2"]]
    
    # Step 2: Sort the dataframe based on Score 1 values
    sorted_df = filtered_df.sort_values(by="Score 1", ascending=True)
    
    # Step 3: Group by policy type and pick the top n policies
    grouped = sorted_df.groupby("Policy type").head(n)
    
    # Step 4: Store the result in df_candidate_policies
    df_candidate_policies_score = grouped.copy()
    
    # Generate sampled policies from percentlie based
    df_candidate_policies_sampled = pd.DataFrame()
    for policy_type in df_CO2["Policy type"].unique():       
        df_candidate_policies_sampled = pd.concat([df_candidate_policies_sampled,filtered_df[filtered_df["Policy type"]==policy_type].sample(n)])
    df_candidate_policies_sampled["Method"]="75th percentile sampled"
    #%% Create a dataframe to compare policies derived from different methods
    df_comparison=df_candidate_policies_score.copy()
    df_comparison["Method"] = "75 percentile, score"
    
    # Assigning "Score 2" based on specific percentile (e.g., 0.55) and policy type
    specific_percentile = 0.55
    df_CO2["Score 2"] = False  # Initialize the "Score 2" column
    
    for policy_type in df_CO2["Policy type"].unique():
        # Fetch indices of policies below the specific percentile for the current policy type
        indices_below_percentile = df_percentile_results.loc[specific_percentile, (policy_type, 'indices')]
        
        # Update the "Score 2" column for the identified policies
        df_CO2.loc[indices_below_percentile, "Score 2"] = True
        
    df_candidate_policies_percentile = df_CO2[df_CO2["Score 2"]==True].copy()
    df_candidate_policies_percentile["Method"]="55th percentile"
    
    df_comparison = pd.concat([df_comparison, df_candidate_policies_percentile, df_candidate_policies_sampled], ignore_index=True)
    
    #%% PLot policy comparisons
    #sns.pairplot(data=df_comparison,hue="Method",x_vars=model.levers.keys(),y_vars=model.levers.keys())
   # sns.pairplot(data=df_comparison[df_comparison["Policy type"]=="All levers"],hue="Method",x_vars=model.levers.keys(),y_vars=model.levers.keys())
    #sns.pairplot(data=df_comparison[df_comparison["Policy type"]!="All levers"],hue="Method",x_vars=model.levers.keys(),y_vars=model.levers.keys())
    import seaborn as sns
    # List of variables
    variables = list(model.levers.keys())
    
    # Number of variables
    n = len(variables)
    
    # Calculate the number of rows for the subplots grid
    num_rows = (n + 1) // 2  # this ensures even number of variables get their own row and an odd number gets an extra row
    
    # Plot for "All levers"
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 5*num_rows))
    for idx, var in enumerate(variables):
        sns.kdeplot(data=df_comparison[df_comparison["Policy type"] == "All levers"], x=var, hue='Method', ax=axes[idx//2, idx%2], fill=True)
        axes[idx//2, idx%2].set_title(f"All levers: {var}")
    
    # If there's an odd number of variables, remove the last unused subplot
    if n % 2 != 0:
        fig.delaxes(axes[num_rows-1, 1])
    
    plt.tight_layout()
    plt.show()
    
    # Plot for other "Policy type"
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 5*num_rows))
    for idx, var in enumerate(variables):
        sns.kdeplot(data=df_comparison[df_comparison["Policy type"] != "All levers"], x=var, hue='Method', ax=axes[idx//2, idx%2], fill=True)
        axes[idx//2, idx%2].set_title(f"No transport efficient planning: {var}")
    
    # If there's an odd number of variables, remove the last unused subplot
    if n % 2 != 0:
        fig.delaxes(axes[num_rows-1, 1])
    
    plt.tight_layout()
    plt.show()
    
    # Levers to be plotted
    levers = ['L1_bio_share_diesel', 'L2_bio_share_gasoline', 'L3_additional_car_energy_efficiency', 
              'L4_additional_truck_energy_efficiency', 'L5_fuel_tax_increase_gasoline', 'L6_fuel_tax_increase_diesel', 
              'L7_km_tax_cars', 'L8_km_tax_trucks', 'L9_transport_efficient_planning_cars', 'L10_transport_efficient_planning_trucks']
    
    # Construct lever-based limits
    limits_levers = pd.DataFrame(index=['min', 'max'])
    for item in levers:
        limits_levers[item] = [df_full[item].min(), df_full[item].max()]
    
    # Initialize parallel coordinates plot
    paraxes = parcoords.ParallelAxes(limits_levers)
    #paraxes.plot(df_full[df_full.index.isin(df_candidate_policies.index) == False], color='gray')  # Non-selected policies in gray
    
    # Color map for unique policy types
    colors = plt.cm.tab10(range(len(df_comparison['Method'].unique())))
    
    # Plot selected policies with unique colors
    for idx, method in enumerate(df_comparison['Method'].unique()):
        selected_data = df_comparison[df_comparison['Method'] == method]
        paraxes.plot(selected_data, label=f'Policy type: {method}', color=colors[idx])
    
    # Add legend, title, and display plot
    paraxes.legend()
    plt.title("Parallel Coordinates Plot of Policies")
    plt.show()
    
    #%% Choose what type of policies should be used as candidate policies, i.e. what method
    df_candidate_policies=df_candidate_policies_sampled.copy()
#%%  Parcoords of candidate policies Plot policies against policy levers
    from ema_workbench.analysis import parcoords
    import matplotlib.pyplot as plt
    
    # Levers to be plotted
    levers = ['L1_bio_share_diesel', 'L2_bio_share_gasoline', 'L3_additional_car_energy_efficiency', 
              'L4_additional_truck_energy_efficiency', 'L5_fuel_tax_increase_gasoline', 'L6_fuel_tax_increase_diesel', 
              'L7_km_tax_cars', 'L8_km_tax_trucks', 'L9_transport_efficient_planning_cars', 'L10_transport_efficient_planning_trucks']
    
    # Construct lever-based limits
    limits_levers = pd.DataFrame(index=['min', 'max'])
    for item in levers:
        limits_levers[item] = [df_full[item].min(), df_full[item].max()]
    
    # Initialize parallel coordinates plot
    paraxes = parcoords.ParallelAxes(limits_levers)
    paraxes.plot(df_full[df_full.index.isin(df_candidate_policies.index) == False], color='gray')  # Non-selected policies in gray
    
    # Color map for unique policy types
    colors = plt.cm.tab10(range(len(df_candidate_policies['Policy type'].unique())))
    
    # Plot selected policies with unique colors
    for idx, policy_type in enumerate(df_candidate_policies['Policy type'].unique()):
        selected_data = df_candidate_policies[df_candidate_policies['Policy type'] == policy_type]
        paraxes.plot(selected_data, label=f'Policy type: {policy_type}', color=colors[idx])
    
    # Add legend, title, and display plot
    paraxes.legend()
    plt.title("Parallel Coordinates Plot of Policies")
    plt.show()

    #%%  Plot policies against outcomes 
    outcomes = model.outcomes.keys()
    limits_outcomes = pd.DataFrame()  # Create a dataframe for lever-based limits
    for item in outcomes:
        limits_outcomes.loc[0, item] = min(df_full[item])  # Get lower bound
        limits_outcomes.loc[1, item] = max(df_full[item])  # Get upper bound
    
    # Step 1: Rename columns
    renamed_columns = {
        'M1_CO2_TTW_total': "CO2 [mton] ",
        'M2_driving_cost_car': "Driving cost car [SEK] ",
        'M3_driving_cost_truck': "Driving cost truck [SEK] ",
        'M4_energy_use_bio': "Energy bio. [TWh] ",
        'M5_energy_use_electricity': "Energy el. [TWh] "
    }
    
    limits_outcomes.rename(columns=renamed_columns, inplace=True)
    df_full.rename(columns=renamed_columns, inplace=True)
    df_candidate_policies.rename(columns=renamed_columns, inplace=True)
    #%% Plot for poster
    # Step 2: Plot parallel coordinates
    
    paraxes = parcoords.ParallelAxes(limits_outcomes,formatter={"maxima":".1f","minima":".1f"},fontsize=20,rot=0)
    
    # Non-selected policies in gray
    paraxes.plot(df_full[df_full.index.isin(df_candidate_policies.index) == False], color='lightgrey',linewidth=0.1,alpha=0.5)
    
    # Create a colormap for unique policy types using viridis
    n_unique_policies = len(df_candidate_policies['Policy type'].unique())
    # Manually specify colors: Dark Plum and Dark Gold
    colors = ["#blue", "orange"]

    # Plot selected policies with the manually specified colors
    for idx, policy_type in enumerate(df_candidate_policies['Policy type'].unique()):
        selected_data = df_candidate_policies[df_candidate_policies['Policy type'] == policy_type]
        paraxes.plot(selected_data, label=f'Policy type: {policy_type}', color=colors[idx])
    
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
    parcoords_fig.savefig("parallel_coordinates_plot_transparent_viridis.png", dpi=500, format="png", bbox_inches="tight", transparent=True)
    
    # If you want to show the plot, you would now use
    plt.show()     
    #%% pairplots of all policies
    import seaborn as sns
    #pairplot_kws={"alpha":0.5,"s":1}
    #levers on levers
    #sns.pairplot(data=df_candidate_policies,x_vars=model.levers.keys(),y_vars=model.levers.keys(),hue="Policy type")
    
    #outcomes on outcomes
   # sns.pairplot(data=df_candidate_policies,x_vars=model.outcomes.keys(),y_vars=model.outcomes.keys(),hue="Policy type")
    
    #levers on outcomes
   # sns.pairplot(data=df_candidate_policies,x_vars=model.levers.keys(),y_vars=model.outcomes.keys(),hue="Policy type")  

    # %% Add a new policy type "Transport effieicnt society not realized in the candidate policy data
    # Create a boolean mask for rows where Policy type is 'All levers'
    # mask = df_candidate_policies['Policy type'] == 'All levers'

    # # Copy these rows, modify as needed, and append to the original DataFrame
    # new_rows = df_candidate_policies[mask].copy()
    # new_rows['Policy type'] = 'Transport efficient society not realized'
    # new_rows['L9_transport_efficient_planning_cars'] = 0
    # new_rows['L10_transport_efficient_planning_trucks'] = 0
    # df_candidate_policies = pd.concat([df_candidate_policies, new_rows])
    # %% resetn indices
    df_candidate_policies = df_candidate_policies.reset_index(drop=True)
    # Remove unnecessary columns
    df_candidate_policies = df_candidate_policies.drop(columns=["Score 1","Score 2","Method"])
    #%% Save the dataframe with candidate policies
    filename=date+"_"+str(nfe)+"candidate_policies"+".p"
    pickle.dump(df_candidate_policies,open("./output_data/"+filename,"wb"))