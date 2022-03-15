# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:20:38 2021

@author: aengholm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ema_workbench.analysis.pairs_plotting import (pairs_scatter)
from ema_workbench.analysis import feature_scoring

#%% plotting settings
SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['figure.dpi'] = 300

sns.set(rc={"figure.dpi":300})

sns.set_palette("bright")
#%% Load data
#Should previously saved result data be loaded? If not, data from workspace is used

n_policies=8
n_scenarios=500
load_results=1
if load_results==1:
    from ema_workbench import load_results
    t1='./output_data/'+str(n_scenarios)+'_scenarios_'+str(n_policies)+'_policies_2022-03-15'
    results = load_results(t1+'.tar.gz')
    experiments=results[0]
    outcomes=results[1]
    df_outcomes=pd.DataFrame(outcomes)
    import pickle
    model=pickle.load( open(t1+"model_.p", "rb" ) )
    df_full=pd.concat([experiments,df_outcomes],axis=1,join="inner") #DF with both experiments and outcomes
    df_outcomes['policy'] = experiments ["policy"]
    #%% Fix nan bug 
    if experiments.isnull().values.any(): #Check if experiments contains nans. 
        #Assume it is the last row issue, drop last row (Experiment)
        experiments_as_loaded=experiments
        print("NaN in outcomes")
        experiments.drop(experiments.tail(1).index,inplace=True) # drop last n rows of experiments
        df = pd.DataFrame(data=outcomes)
        new = df[:-1]
        outcomes_as_loaded=outcomes
        outcomes=new.to_dict("series")
        
#%% Create a df with policies used
df_policies=pd.DataFrame()
df_policies["policy"]=experiments["policy"]
for i in model.levers.keys():
    df_policies[str(i)]=experiments[str(i)]
df_policies=df_policies.drop_duplicates()
df_policies.reset_index(drop=True, inplace=True)


#%%Visualize policies

#Plot table of policies
fig, ax = plt.subplots()
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=df_policies.values, colLabels=df_policies.columns, loc='center')
fig.tight_layout()
plt.show()
#%%
# Visualize each policy as a dot diagram
sns.set_theme(style="whitegrid")
g = sns.PairGrid(df_policies.sort_values("policy", ascending=False),
                 x_vars=df_policies.columns[1:], y_vars=["policy"],
                 palette="colorblind", height=10, aspect=.1)

g.map(sns.stripplot, size=5, orient="h", jitter=False, linewidth=1,
      edgecolor="w")
for ax, title in zip(g.axes.flat, df_policies.columns[1:]):
    # Set a different title for each axes
    ax.set(title=title)
    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
sns.despine(left=True, bottom=True)

#%%
#Parcoords plot, requires manual work 
parcoords=0
if parcoords == 1:
    df_policies["ICE CO2 reduction ambition level"]=df_policies["ICE CO2 reduction ambition level"].cat.as_ordered()
    df_policies["policy"]=df_policies["policy"].cat.as_ordered()
    a=df_policies["ICE CO2 reduction ambition level"].unique()
    b=[]
    for i in range(len(a)):
        b.append(a[i])
    #df_policies.at[0,'ICE ambition level']= b
    df_policies_parcoords=df_policies.drop(columns="ICE CO2 reduction ambition level")
    df_policies_parcoords=df_policies
    from ema_workbench.analysis import parcoords
    limits = parcoords.get_limits(df_policies_parcoords)
    paraxes = parcoords.ParallelAxes(limits)
    paraxes.plot(df_policies_parcoords)
    plt.legend(df_policies["policy"])

#%%
### Pairwise scatter plots on outcomes
fig,axes = pairs_scatter(experiments,outcomes, legend=True, group_by="policy")
fig.set_size_inches(15,15)
plt.show()

#%% mega plot
# df_full2=df_full.drop(columns=["ICE CO2 reduction ambition level",
#                                "Bus energy consumption","model","scenario",
#                                "Share FAME diesel","Share ethanol gasoline",
#                                "Transport efficient society trucks",
#                                "Transport efficient society light vehicles",
#                                "Additional energy efficiency trucks",
#                                "Additional energy efficiency light vehicles",
#                                "km-tax light vehicles",
#                                "km-tax trucks"]
#                       )
# fig,axes =  sns.pairplot(data=df_full2, hue="policy")
# fig.set_size_inches(15,15)
#%%

###FEATURE SCORING ON OUTCOMES

fs = feature_scoring.get_feature_scores_all(experiments, outcomes)
plt.figure()
fig=sns.heatmap(fs, cmap='viridis', annot=True,fmt=".2f")

### scenario discovery
#%%
#Define criterion for unwanted outcome
fail_criterion_CO2=-0.7
fail_criterion_bio=20
#Prepare data, x and y arrays
x = experiments
y1=outcomes['CO2 TTW change total']>fail_criterion_CO2
y2=outcomes['Energy bio total']>fail_criterion_bio
y3=[0]*len(x)
for j in range(len(y1)):
    if y1[j]==True or y2[j]==True:
        y3[j]=True
    else: 
        y3[j]=False
#choose criterion to use y1=first criterion only, y2=second criterion only,y3=either first or second criterion=fail
y=np.array(y3,dtype=bool)
x["Target not met"]=y
#Basic statistics 
n_fail = np.count_nonzero(y)
share_fail=n_fail/len(y)
share_success=1-share_fail
#Basic statistics per policy
df_policies=pd.DataFrame()
for i in x["policy"].unique():
    temp=x[x["policy"].str.contains(i)]
    print(i)
    df_policies["policy"].append(i)
    df_policies["n fail"]=np.count_nonzero(temp["Target not met"])
    df_policies["n success"]=len(temp)-np.count_nonzero(temp["Target met"])

#%%
import statistics
sns.set_palette("bright")
plt.figure()
sns.scatterplot(x='CO2 TTW change total', y='Energy bio total', 
              data=df_full, hue="policy", alpha=0.5)
plt.ylabel("Energy bio total [TWh")
plt.xlabel("Change in CO2 cmpr.2010, TTW")
plt.show()

sns.scatterplot(x='CO2 TTW change total', y='Energy bio total', 
              data=df_full, hue="Car el share", alpha=0.5)
plt.show()

sns.scatterplot(x='Car el share', y= 'CO2 TTW change total', 
              data=df_full, hue="policy", alpha=0.5)
plt.show()

sns.scatterplot(x='Car el share', y='Energy bio total', 
              data=df_full, hue="policy", alpha=0.5)
plt.show()

sns.scatterplot(x='CO2 TTW change light vehicles', y='Driving cost light vehicles', 
              data=df_full, hue="policy", alpha=0.5)
plt.show()

sns.scatterplot(x='CO2 TTW change trucks', y='Driving cost trucks', 
              data=df_full, hue="policy", alpha=0.5)
plt.show()

#%%
#Plot hist/KDE on criterions
sns.displot(x='CO2 TTW change total', data=df_full, hue="policy",kde=True)
plt.axvspan(fail_criterion_CO2, max(outcomes['CO2 TTW change total']), 
            facecolor='red', alpha=0.05,edgecolor='None')
#plt.axvline(statistics.mean(outcomes['CO2 TTW change total']),color="black", 
#            ls="--")
plt.axvline(fail_criterion_CO2,color="red", 
            ls="--")
#plt.legend (["B","C1","C2","C3","C4",
         #   "D1","D2","D3"])

sns.displot(x='Energy bio total', data=df_full, hue="policy", kde=True)
plt.axvspan(fail_criterion_bio, max(outcomes['Energy bio total']), 
            facecolor='red', alpha=0.05,edgecolor='None')
plt.axvline(statistics.mean(outcomes['Energy bio total']),color="red",ls="--")
#%%
g=sns.displot(x='CO2 TTW change total', y='Energy bio total', 
              data=df_full, hue="policy", alpha=0.8)
ylim=g.ax.get_ylim()
xlim=g.ax.get_xlim()
plt.axvspan(fail_criterion_CO2, xlim[1],facecolor='red', alpha=0.1, 
            edgecolor='none')
plt.axvspan(xlim[0],fail_criterion_CO2,fail_criterion_bio/ylim[1],
            facecolor='red', alpha=.1,edgecolor='none')

#%%


# Feature scoring on scenario disocvery data (what ucnertainties drive fail/success of outcomes)
fs_discovery, alg = feature_scoring.get_ex_feature_scores(x, y, mode=feature_scoring.RuleInductionType.CLASSIFICATION)
fs_discovery.sort_values(ascending=False, by=1)
plt.figure()
fig=sns.heatmap(fs_discovery, cmap='viridis', annot=True)
#%%
#Scatter pair plot for all uncertainties and whether or not target is met
# x_copy2 = experiments.copy()
# x_copy2= x_copy2.drop('scenario', axis=1)
# x_copy2["Target not met"]=y
# g= sns.PairGrid(x_copy2, hue="Target not met")
# g.map_diag(sns.kdeplot, shade=True)
# g.map_offdiag(plt.scatter,edgecolor="white",alpha=0.5)
# g.add_legend()
# g.fig.set_size_inches(20,20)
#%%
#Dimensional stacking
from ema_workbench.analysis import dimensional_stacking
dimensional_stacking.create_pivot_plot(x,y, 2, nbins=5)
plt.show()
#%%
#Regional sensitivty analysis, # model is the same across experiments
from ema_workbench.analysis import regional_sa
sns.set_style('white')
x_copy = experiments.copy()
x_copy = x_copy.drop('model', axis=1)
#x_copy = x_copy.drop('policy', axis=1)
fig = regional_sa.plot_cdfs(x_copy,y,ccdf=False)
sns.despine()
plt.show()

#%%%#Perform PRIM analysis

#Set up PRIM
from ema_workbench.analysis import prim
prim_alg = prim.Prim(x, y, threshold=0.5)
#%%
#Find 1st box
box1 = prim_alg.find_box()
#%%
#Visualizations of Box1
box1.show_tradeoff()

for i in range(0,len(box1.peeling_trajectory.T.columns)):
    s=box1.peeling_trajectory.T[i].id
    if (i%2)==0:
        plt.text(box1.peeling_trajectory.T[i].coverage+.02,box1.peeling_trajectory.T[i].density-.03 ,s,fontsize=10)
    else:
        plt.text(box1.peeling_trajectory.T[i].coverage-.03,box1.peeling_trajectory.T[i].density+.03 ,s,fontsize=10)
plt.show()

#Choose point for inspection
i1=round((len(box1.peeling_trajectory.T.columns)-1))
#or choose box manually
i1=17
box1.inspect(i1)
box1.inspect(i1, style='graph')
plt.show()
box1.show_ppt()
ax=box1.show_pairs_scatter(i1)
plt.show()
#%%
#Find 2nd box
box2 = prim_alg.find_box()
#Visualizations of Box2
box2.show_tradeoff()
for i in range(0,len(box2.peeling_trajectory.T.columns)):
    s=box2.peeling_trajectory.T[i].id
    plt.text(box2.peeling_trajectory.T[i].coverage,box2.peeling_trajectory.T[i].density,s)
#Choose the last point
i2=len(box2.peeling_trajectory.T.columns)-1
box2.inspect(i2)
box2.inspect(i2,style='graph')
plt.show()
box2.show_pairs_scatter(i2)
plt.show()

coverage_2boxes=box1.coverage+box2.coverage

#%%
# #%%
# # ------- PART 1: Define a function that do a plot for one line of the dataset!
 
# def make_spider( row, title, color):
#     from math import pi
#     # number of variable
#     categories=list(df_policies)[1:]
#     N = len(categories)

#     # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
#     angles = [n / float(N) * 2 * pi for n in range(N)]
#     angles += angles[:1]

#     # Initialise the spider plot
#     ax = plt.subplot(3,3,row+1, polar=True, )

#     # If you want the first axis to be on top:
#     ax.set_theta_offset(pi / 2)
#     ax.set_theta_direction(-1)

#     # Draw one axe per variable + add labels labels yet
#     plt.xticks(angles[:-1], categories, color='grey', size=8)

#     # Draw ylabels
#     ax.set_rlabel_position(0)
#     plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
#     plt.ylim(0,5)

#     # Ind1
#     values=df_policies.loc[row].drop('policy').values.flatten().tolist()
#     values += values[:1]
#     ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
#     ax.fill(angles, values, color=color, alpha=0.4)

#     # Add a title
#     plt.title(title, size=11, color=color, y=1.1)

    
# # ------- PART 2: Apply the function to all individuals
# # initialize the figure
# my_dpi=96
# plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
 
# # Create a color palette:
# my_palette = plt.cm.get_cmap("Set2", len(df_policies.index))
 
# # Loop to plot
# for row in range(0, len(df_policies.index)):
#     print(row)
#     make_spider( row=row, title='policy '+str(df_policies['policy'][row]), color=my_palette(row))