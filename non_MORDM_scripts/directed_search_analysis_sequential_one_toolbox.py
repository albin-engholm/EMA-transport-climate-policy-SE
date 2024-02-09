# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:25:15 2022

@author: aengholm
"""
import pandas as pd
df_full=pd.DataFrame()
policy_types=["B","C","D"]

load_results=1
if load_results==1:
    date="2023-06-08"
    nfe=50000
    n_scenarios=3

    t1='./output_data/'+str(nfe)+"_nfe_"+"directed_search_sequential_"+date+"_"+str(n_scenarios)+"_scenarios"+".p"
    # =str(nfe)+'_nfe_directed_search_sequential_'+str(date.today())+'_'+str(n_scenarios)+'_scenarios'
    import pickle
    results_list,convergence,scenarios=pickle.load( open(t1, "rb" ))
    t2='./output_data/'+str(nfe)+"_nfe_"+"directed_search_sequential_"+date+"_"+str(n_scenarios)+"_scenarios"+"model_.p"
    model=pickle.load( open(t2, "rb" ) )
    scenario_count=0
    for results in results_list:
        results["Scenario"]=scenario_count
        df_full=pd.concat([df_full,results],ignore_index=True)
        scenario_count=scenario_count+1

#%%
multiple_days=False
if multiple_days:
    import pandas as pd
    df_full=pd.DataFrame()
    policy_types=["B","C","D"]
    
    load_results=1
    if load_results==1:
        date="2023-05-04"
        nfe=20000
        n_scenarios=3
        for policy_type in policy_types:
            if policy_type=="D":
                date="2023-05-05"
            t1='./output_data/'+policy_type+str(nfe)+"_nfe_"+"directed_search_sequential_"+date+"_"+str(n_scenarios)+"_scenarios"+".p"
           # =str(nfe)+'_nfe_directed_search_sequential_'+str(date.today())+'_'+str(n_scenarios)+'_scenarios'
            import pickle
            results_list,convergence,scenarios=pickle.load( open(t1, "rb" ))
            t2='./output_data/'+policy_type+str(nfe)+"_nfe_"+"directed_search_sequential_"+date+"_"+str(n_scenarios)+"_scenarios"+"model_.p"
            model=pickle.load( open(t2, "rb" ) )
            scenario_count=0
            for results in results_list:
                results["Scenario"]=scenario_count
                results["Policy type"]=policy_type
                df_full=pd.concat([df_full,results],ignore_index=True)
                scenario_count=scenario_count+1

#%%  Create df with all runs


#%% Create a df with policies used
# 
# df_policies=pd.DataFrame()
# policies=list(range(len(results)))
# df_policies["policy"]=policies
# for i in model.levers.keys():
#     df_policies[str(i)]=results[str(i)]
# df_policies=df_policies.drop_duplicates()
# df_policies.reset_index(drop=True, inplace=True)

#%% Prep visualizations
#Create a colormap for separating scenarios
import seaborn as sns
scenario_colors=sns.color_palette("hls", len(list(scenarios.index)))
policy_colors=sns.color_palette("Set1", len(policy_types))
#%%sample a sub-set of strategies for visualization if numbr of solutions is above thresolhd
threshold=20
df_sampled=pd.DataFrame()
for policy_type in policy_types:
    temp=df_full[df_full["Policy type"]==policy_type]
    for scenario in df_full["Scenario"].unique():
        temp2=temp[temp["Scenario"]==int(scenario)]
      
        if len(temp)>threshold:
            sample=temp2.sample(threshold,replace=True)
            df_sampled=pd.concat([df_sampled,sample])
            #print("Sampled added")
        else:
            df_sampled=pd.concat([df_sampled,temp])
df_sampled=df_sampled.drop_duplicates()      
            #print("Original added")

#%% Identify policy clusters using K-means clustering   
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

cols_to_cluster = ['M1_CO2_TTW_change_total',
'M2_driving_cost_car', 'M3_driving_cost_truck', 'M4_energy_use_bio',
'M5_energy_use_electricity']

# Normalize the data 
scaler = StandardScaler()
df_scaled = df_full.copy()
df_scaled[cols_to_cluster] = scaler.fit_transform(df_full[cols_to_cluster])

# Use K-means clustering
n_clusters = 5 # define the number of clusters
centroids = []

for name, cluster in df_scaled.groupby(["Scenario"]):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(cluster[cols_to_cluster])
    
    # Convert the name tuple to string in a better way
    scenario_str = str(name[0])
    #policy_str = str(name[1])
    name_str = scenario_str
    
    cluster['cluster'] = [name_str + '-cluster' + str(c) for c in kmeans.labels_]
    df_full.loc[cluster.index, 'cluster'] = cluster['cluster']

    # Get centroids and transform them to the original space
    df_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=cols_to_cluster)
    df_centroids_original_scale = pd.DataFrame(scaler.inverse_transform(df_centroids), columns=cols_to_cluster)

    # Append 'scenario', 'policy', 'cluster' back to df_centroids_original_scale
    df_centroids_original_scale['Scenario'] = name[0]
   # df_centroids_original_scale['Policy type'] = name[1]
    df_centroids_original_scale['cluster'] = [name_str + '-cluster' + str(c) for c in range(n_clusters)]
    
    centroids.append(df_centroids_original_scale)

df_centroids_original_scale = pd.concat(centroids).reset_index(drop=True)


import seaborn as sns
import matplotlib.pyplot as plt

# Select the cluster and the features for the pair plot
cols_to_plot = ['cluster', 'M1_CO2_TTW_change_total',
'M2_driving_cost_car', 'M3_driving_cost_truck', 'M4_energy_use_bio',
'M5_energy_use_electricity']

# Create pairplot
#sns.pairplot(df_full[cols_to_plot], hue='cluster')
#plt.show()
plt.figure()
sns.pairplot(df_centroids_original_scale[cols_to_plot], hue='cluster')

import seaborn as sns
import matplotlib.pyplot as plt

# Set the style of the plot
sns.set(style="ticks")

# Create a copy of the original DataFrame
df_plot = df_full.copy()

# Combine original data and centroids
df_combined = pd.concat([df_plot, df_centroids_original_scale], ignore_index=True)

# Create the pairplot with hue separation by cluster
g = sns.pairplot(df_combined, vars=cols_to_cluster, hue='cluster', plot_kws={'alpha': 0.6})

# Plot the centroids in each subplot
for ax in g.axes.flat:
    if ax.is_first_row() and ax.is_last_col():
        continue  # Skip the empty subplot at the top right
        
    for cluster_name in df_centroids_original_scale['cluster'].unique():
        centroid = df_centroids_original_scale[df_centroids_original_scale['cluster'] == cluster_name]
        ax.scatter(centroid[cols_to_cluster[0]], centroid[cols_to_cluster[1]], marker='X', s=100, color='black')

# Display the plot
plt.show()

#%% visualize pareto frontier per policy type and per scenario
import seaborn as sns
import matplotlib.pyplot as plt

#hue_order_policy=["D","C","B"]
outcome_keys=model.outcomes.keys()
for outcome in outcome_keys:
    if (model.outcomes[outcome].kind!=0) and (outcome!="M1_CO2_TTW_change_total"):
        plt.figure()
        plt.rcParams['figure.dpi'] = 300
        sns.set_theme(style="whitegrid")
        g=sns.scatterplot(data=df_full,x=outcome,y="M1_CO2_TTW_change_total",hue="Scenario",palette=scenario_colors)
        plt.axhline(y=-0.9,color="black",linestyle="--")
        g.set(ylim=[0,-1])

for outcome in outcome_keys:
    if (model.outcomes[outcome].kind!=0) and (outcome!="CO2 TTW change total"):
        plt.figure()
        plt.rcParams['figure.dpi'] = 300
        sns.set_theme(style="whitegrid")
        g=sns.scatterplot(data=df_full,x=outcome,y="M1 CO2 TTW change total",hue="Policy type",style="Scenario",palette=policy_colors)
        plt.axhline(y=-0.9,color="black",linestyle="--")
        g.set(ylim=[0,-1])

plt.rcParams['figure.dpi'] = 300


plt.figure()
g=sns.scatterplot(data=df_sampled,x="Energy bio total",y="CO2 TTW change total",hue="Scenario",style="Policy type",palette=scenario_colors)
g.set(ylim=[0,-1])
# as factgrid per  type
f=sns.FacetGrid(df_sampled,col="Policy type",hue="Scenario",sharex=True,sharey=True,palette=scenario_colors,col_wrap=1) 
f.map_dataframe(sns.scatterplot, x="Energy bio total",y="CO2 TTW change total")
f.set(ylim=[0,-1])
f.set_xlabels(rotation=15)
f.add_legend()
#%%
# as factgrid per scenario type
f=sns.FacetGrid(df_sampled,col="Scenario",hue="Policy type",sharex=True,sharey=True,palette=policy_colors,col_wrap=1)
#f.map_dataframe(sns.regplot, x="Energy bio total",y="CO2 TTW change total",ci=None,order=2)
f.map_dataframe(sns.scatterplot, x="Energy bio total",y="CO2 TTW change total")
f.set(ylim=[0,-1])
f.set_xlabels(rotation=15)
f.add_legend()


# as factgrid per scenario type


f=sns.FacetGrid(df_sampled,col="Scenario",hue="Policy type",sharex=True,sharey=True,palette=policy_colors,col_wrap=1)
#f.map_dataframe(sns.regplot, x="Delta CS light vehicles",y="CO2 TTW change total",order=2,ci=None)
f.map_dataframe(sns.scatterplot, x="Driving cost trucks",y="CO2 TTW change total")
f.set(ylim=[0,-1])
f.set_xlabels(rotation=15)
f.add_legend()

#%% linear model paretofront bio-CO2
import numpy as np
from sklearn.linear_model import LinearRegression
linear_coefs=[]
linear_scores=[]
for scenario in df_full["Scenario"].unique():
    x=np.array(df_full[(df_full["Scenario"]==scenario) & (df_full["Policy type"]=="B")]["Energy bio total"]).reshape(-1,1)
    y=np.array(df_full[(df_full["Scenario"]==scenario) & (df_full["Policy type"]=="B")]["CO2 TTW change total"])
    reg=LinearRegression().fit(x,y)
    linear_scores.append(reg.score(x,y))
    linear_coefs.append(reg.coef_)
    

    #%%
# import plotly.express as px
# from plotly.offline import plot
# fig=px.scatter_3d(data_frame=results,x="Driving cost light vehicles relative reference",y="Energy total",z="Driving cost trucks relative reference")
# fig.show()
# plot(fig)
#%%
  # Visualize each policy as a dot diagram
# sns.set_theme(style="whitegrid")
# sns.set(font_scale=2)
# g = sns.PairGrid(df_policies.sort_values("policy", ascending=False),
#                  x_vars=df_policies.columns[1:], y_vars=["policy"],
#                  palette="colorblind", height=15, aspect=.15)

# g.map(sns.stripplot, size=8, orient="h", jitter=False, linewidth=1,
#       edgecolor="w")
# for ax, title in zip(g.axes.flat, df_policies.columns[1:]):
#     # Set a different title for each axes
#     ax.set(title=title)
#     # Make the grid horizontal instead of vertical
#     ax.xaxis.grid(False)
#     ax.yaxis.grid(True)
#     ax.tick_params(axis='x', rotation=90)
# sns.despine(left=True, bottom=True)
# sns.set(font_scale=1)

# #df_policies["ICE CO2 reduction ambition level"]=df_policies["ICE CO2 reduction ambition level"].cat.as_ordered()
# #df_policies["policy"]=df_policies["policy"].cat.as_ordered()

# %%Parcoords for policy levers
df=df_sampled
# get the minimum and maximum values as present in the dataframe
from ema_workbench.analysis import parcoords
limits = parcoords.get_limits(df.drop(columns=["Scenario","Policy type"]))

#get limits from lever ranges
levers=model.levers._data #Get levers
limits_levers=pd.DataFrame() #Create a dataframe for lever-based limits
for item in levers: 
    limits_levers.loc[0,item]=levers[item].lower_bound #get limits per lever and add to dataframe
    limits_levers.loc[1,item]=levers[item].upper_bound

limits=limits_levers.drop(columns=["ICE CO2 reduction ambition level","Bus energy consumption"])
# COde example for manually specifying limits
#limits.loc[0, ["Share HVO diesel", "Share FAME diesel", "Share HVO gasoline", ]] = 0
# inertia and reliability are defined on unit interval, so their theoretical maximum is 1
#%%
#Parcorrds of policies all scenarios
paraxes = parcoords.ParallelAxes(limits)
count=0
for i in reversed(df["Policy type"].unique()):
    
    label="Policy type " + str(i)
    paraxes.plot(df[df["Policy type"]==i],color=policy_colors[count], label=label)
    paraxes.legend()
    count=count+1
plt.show()

#%%#Parcorrds of each policy type per scenario

count=0
for j in list(scenarios.index):
    df_temp=df[df["Scenario"]==j]
    count=0
    for i in reversed(df["Policy type"].unique()):
        paraxes = parcoords.ParallelAxes(limits)
        label="Policy type " + str(i)
        paraxes.plot(df[df["Policy type"]==i],color=policy_colors[count], label=label)
        paraxes.legend()
        count=count+1
        plt.title("Scenario: "+str(j))
        plt.show()
    

#Parcoords of policies per scenario
for j in list(scenarios.index):
    df_temp=df[df["Scenario"]==j]
    paraxes = parcoords.ParallelAxes(limits)
    count=0
    for i in reversed(df_temp["Policy type"].unique()):
        label="Policy type " + str(i)
        paraxes.plot(df_temp[df_temp["Policy type"]==i],color=policy_colors[count], label=label)
        paraxes.legend()
        count=count+1
        plt.title("Scenario: "+str(j))
    plt.show()
    
#%%#Parcorrds of each policy type per scenario meeting the CO2 targets
df_old=df
df=df_sampled[df_sampled["CO2 TTW change total"]<-0.9]
count=0
for j in list(scenarios.index):
    df_temp=df[df["Scenario"]==j]
    count=0
    for i in reversed(df["Policy type"].unique()):
        paraxes = parcoords.ParallelAxes(limits)
        label="Policy type " + str(i)
        paraxes.plot(df[df["Policy type"]==i],color=policy_colors[count], label=label)
        paraxes.legend()
        count=count+1
        plt.title("Scenario: "+str(j))
        plt.show()
    

#Parcoords of policies per scenario
for j in list(scenarios.index):
    df_temp=df[df["Scenario"]==j]
    paraxes = parcoords.ParallelAxes(limits)
    count=0
    for i in reversed(df_temp["Policy type"].unique()):
        label="Policy type " + str(i)
        paraxes.plot(df_temp[df_temp["Policy type"]==i],color=policy_colors[count], label=label)
        paraxes.legend()
        count=count+1
        plt.title("Scenario: "+str(j))
    plt.show()    
df=df_old
# %%Parcoords for scenarios
# get the minimum and maximum values as present in the dataframe
df=scenarios
limits = parcoords.get_limits(df)


paraxes = parcoords.ParallelAxes(limits)
scenario_colors=["g","r","b"]
count=0
scenario_labels=["Best case","Worst case","Reference case"]
for i in list(scenarios.index):
    label="Scenario " + str(i)
    paraxes.plot(df.loc[i,:],color=scenario_colors[i], label=scenario_labels[i])
    paraxes.legend()
    count=count+1
plt.show()
#%% visualizaiton of policies
sns.displot(data=df_full,x="Share HVO diesel",y="Share HVO gasoline",hue="Policy type",palette="bright",kind="kde")
#%% select sub-set of experiments where cliate target is met
df_target=df_full[df_full["CO2 TTW change total"]<-0.9]
sns.displot(data=df_target,x="Energy bio total", col="Policy type",hue="Scenario")


for policy in df_full["Policy type"].unique():
    sns.displot(data=df_target[df_target["Policy type"]==policy],x="Share HVO diesel",y="Share HVO gasoline",kind="kde")

sns.displot(data=df_target,x="Energy bio total", col="Scenario",hue="Policy type",kind="kde",col_wrap=3)
sns.displot(data=df_target,x="Energy bio total", y="Driving cost light vehicles relative reference",col="Scenario",hue="Policy type",col_wrap=3)

sns.displot(data=df_target,x="Energy bio total", kind="kde",hue="Policy type")
plt.xlabel("Bio fuel use [TWh/y]")
plt.title("Bio fuel use for solutions where climate target is met")
# for scenario in [3,4]:

#     df=df_target[df_target["Scenario"]==scenario]
#     sns.displot(data=df,x="Energy bio total", col="Policy type",hue="Scenario")
#     sns.displot(data=df,x="Energy bio total", kind="kde",hue="Policy type")
#     plt.xlabel("Bio fuel use [TWh/y]")
#     plt.title("Bio fuel use for solutions where climate target is met in scenario "+str(scenario))

#%%
sns.pairplot(data=df_full, hue="Scenario")