# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:25:15 2022

@author: aengholm
"""


load_results=1
if load_results==1:
    date="2023-02-27"
    nfe=10000
    t1='./output_data/'+str(nfe)+"_nfe_"+"directed_search_"+date+".p"
    import pickle
    results,convergence=pickle.load( open(t1, "rb" ))
    t2='./output_data/'+str(nfe)+"_nfe_"+"directed_search_"+date+"model_.p"
    model=pickle.load( open(t2, "rb" ) )

#%% Create a df with policies used
import pandas as pd
df_policies=pd.DataFrame()
policies=list(range(len(results)))
df_policies["policy"]=policies
for i in model.levers.keys():
    df_policies[str(i)]=results[str(i)]
df_policies=df_policies.drop_duplicates()
df_policies.reset_index(drop=True, inplace=True)
#%% visualize pareto frontier

import seaborn as sns
import matplotlib as plt
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="whitegrid")
g=sns.scatterplot(data=results,x="Driving cost light vehicles relative reference",y="Energy total",hue="Driving cost trucks relative reference")

g.invert_yaxis()
g.invert_xaxis()

import plotly.express as px
from plotly.offline import plot
fig=px.scatter_3d(data_frame=results,x="Driving cost light vehicles relative reference",y="Energy total",z="Driving cost trucks relative reference")
fig.show()
plot(fig)
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
from ema_workbench.analysis import parcoords

data = pd.DataFrame.from_dict(results) #results.loc[:, [o.name for o in model.outcomes]]

# get the minimum and maximum values as present in the dataframe
limits = parcoords.get_limits(data)

# we know that the lowest possible value for all objectives is 0
limits.loc[0, ["utility", "inertia", "reliability", "max_P"]] = 0
# inertia and reliability are defined on unit interval, so their theoretical maximum is 1
limits.loc[1, ["inertia", "reliability"]] = 1

paraxes = parcoords.ParallelAxes(limits)
paraxes.plot(data)
#paraxes.invert_axis("max_P")
plt.show()
#plot distributions of policies and robust results

#sns.displot(data=robust_results)
#sns.pairplot(data=robust_results)


