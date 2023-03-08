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
    date="2023-03-08"
    nfe=500
    n_scenarios=5
    for policy_type in policy_types:
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
scenario_colors=sns.color_palette("Paired", len(list(scenarios.index)))
policy_colors=sns.color_palette("Set1", len(policy_types))
#%%sample a sub-set of strategies for visualization if numbr of solutions is above thresolhd
threshold=10
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
    

#%% visualize pareto frontier per policy type and per scenario
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure()
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="whitegrid")
g=sns.scatterplot(data=df_full,x="Energy bio total",y="CO2 TTW change total",hue="Scenario",style="Policy type",palette=scenario_colors)
plt.axhline(y=-0.9,color="black",linestyle="--")
g.set(ylim=[0,-1])


plt.figure()
g=sns.scatterplot(data=df_sampled,x="Energy bio total",y="CO2 TTW change total",hue="Scenario",style="Policy type",palette=scenario_colors)
g.set(ylim=[0,-1])
# as factgrid per  type
f=sns.FacetGrid(df_sampled,col="Policy type",hue="Scenario",sharex=True,sharey=True,palette=scenario_colors) 
f.map_dataframe(sns.scatterplot, x="Energy bio total",y="CO2 TTW change total")
f.set(ylim=[0,-1])
f.set_xlabels(rotation=15)
f.add_legend()

# as factgrid per scenario type
f=sns.FacetGrid(df_sampled,col="Scenario",hue="Policy type",sharex=True,sharey=True,palette=policy_colors,col_wrap=3)
f.map_dataframe(sns.scatterplot, x="Energy bio total",y="CO2 TTW change total")
f.set(ylim=[0,-1])
f.set_xlabels(rotation=15)
f.add_legend()


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

#Parcorrds of policies all scenarios
paraxes = parcoords.ParallelAxes(limits)
count=0
for i in reversed(df["Policy type"].unique()):
    label="Policy type " + str(i)
    paraxes.plot(df[df["Policy type"]==i],color=policy_colors[count], label=label)
    paraxes.legend()
    count=count+1
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

# %%Parcoords for scenarios
# get the minimum and maximum values as present in the dataframe
df=scenarios
limits = parcoords.get_limits(df)


paraxes = parcoords.ParallelAxes(limits)
scenario_colors=sns.color_palette("Paired", len(list(scenarios.index)))
count=0
for i in list(scenarios.index):
    label="Scenario " + str(i)
    paraxes.plot(df.loc[i,:],color=scenario_colors[i], label=label)
    paraxes.legend()
    count=count+1
#paraxes.invert_axis("max_P")
plt.show()
#%% visualizaiton of policies
sns.displot(data=df_full,x="Share HVO diesel",y="Share HVO gasoline",hue="Policy type",palette="bright",kind="kde")

#%% select sub-set of experiments where cliate target is met
df_target=df_full[df_full["CO2 TTW change total"]<-0.9]
sns.displot(data=df_target,x="Energy bio total", col="Policy type",hue="Scenario")
sns.displot(data=df_target,x="Energy bio total", col="Scenario",hue="Policy type",kind="kde",col_wrap=3)
sns.displot(data=df_target,x="Energy bio total", kind="kde",hue="Policy type")
plt.xlabel("Bio fuel use [TWh/y]")
plt.title("Bio fuel use for solutions where climate target is met")
# for scenario in [3,4]:

#     df=df_target[df_target["Scenario"]==scenario]
#     sns.displot(data=df,x="Energy bio total", col="Policy type",hue="Scenario")
#     sns.displot(data=df,x="Energy bio total", kind="kde",hue="Policy type")
#     plt.xlabel("Bio fuel use [TWh/y]")
#     plt.title("Bio fuel use for solutions where climate target is met in scenario "+str(scenario))