# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:20:38 2021
This is a script for performing open exploration for Trafikverkets scenario tool
This script is designed for a fixed set of 9 policies (as used in Trafikverkets analysis)
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

#%% Create color palette for separating policies
# Create color palette  for seaborn
colors_policy=["#12c400",  # B startegies = green
               "#0011B3", "#0018FF" ,"#2236FF","#5767FF", # C strategiess=blues
               "#C40012","#FF0018", "#FF6876", #D strategies = reds
               "#1E171A"] # Reference strategy = dark grey
policy_palette = sns.set_palette(sns.color_palette(colors_policy))
sns.set_palette(sns.color_palette(colors_policy))

#hue_marker=[".",".",".",".",".",".",".",".","."]
from ema_workbench import (RealParameter, CategoricalParameter, 
                           ScalarOutcome, ema_logging,
                           perform_experiments, Constant)

from ema_workbench.connectors.excel import ExcelModel
model = ExcelModel("scenarioModel", wd="./models",
                   model_file='Master.xlsx')
model.default_sheet = "EMA"

#Specification of levers
model.levers = [CategoricalParameter("ICE CO2 reduction ambition level",
                                     ["1. Beslutad politik","2 Mer ambitios politik"]
                                     ,variable_name="C62"),
                CategoricalParameter("Bus energy consumption",
                                     ["Beslutad politik","Level 1","Level 2"]
                                     ,variable_name="C63"),
                RealParameter("Share HVO diesel",
                              0, 0.80,
                              variable_name="C64"),
                RealParameter("Share FAME diesel",
                              0, 0.07,
                              variable_name="C65"),
                RealParameter("Share HVO gasoline", 
                              0, 0.7,
                              variable_name="C66"),
                RealParameter("Share ethanol gasoline", 
                              0, 0.1,
                              variable_name="C67"),
                RealParameter("km-tax light vehicles",
                              0,2
                              ,variable_name="C68"),
                RealParameter("km-tax trucks",
                              0,3
                              ,variable_name="C69"),
                RealParameter("Change in fuel tax gasoline",
                              0,.12
                              ,variable_name="C70"),
                RealParameter("Change in fuel tax diesel",
                              0,.12
                              ,variable_name="C71"),
                RealParameter("Additional energy efficiency light vehicles",
                              0,.05
                              ,variable_name="C72"),
                RealParameter("Additional energy efficiency trucks",
                              0,.05
                              ,variable_name="C73"),
                RealParameter("Transport efficient society light vehicles",
                              0,.25
                              ,variable_name="C74"),
                RealParameter("Transport efficient society trucks",
                              0,.20
                              ,variable_name="C75"),
                ]
 

#%% Specify policies
from ema_workbench.em_framework import samplers
manual_policies=True #Use the pre-specified 9 policies?
n_policies=9
if manual_policies:
    n_policies=9
policies=samplers.sample_levers(model, n_policies, sampler=samplers.LHSSampler())
    #%% manual specification of policies   
if manual_policies: # overide the pre-sampled policies
    policy1=(0,     #Additional energy efficiency light vehicles [%]
             0,  #Additional energy efficiency trucks [%]
             1,  #Bus energy consumption [0,1,2]
             .02,     #Change in fuel tax diesel [%/y]
             .02,      #Change in fuel tax gasoline [%/y]  
             1,     #ICE CO2 reduction ambition level [0,1]
             .07,     #Share FAME diesel [%]
             .628,     #Share HVO diesel [%]
             .628,     #Share HVO gasoline [%]
             0.1,     #Share ethanol gasoline [%]
             0,     #Transport efficient society light vehicles [% reduction]
             0,     #Transport efficient society trucks [% reduction]
             0,     #km-tax light vehicles [SEK/km]
             0)     #km-tax trucks [SEK/km]
    policy2=(0,     #Additional energy efficiency light vehicles [%]
             0,  #Additional energy efficiency trucks [%]
             2,  #Bus energy consumption [0,1,2]
             .09,     #Change in fuel tax diesel [%/y]
             .09,      #Change in fuel tax gasoline [%/y]  
             1,     #ICE CO2 reduction ambition level [0,1]
             .07,     #Share FAME diesel [%]
             .52,     #Share HVO diesel [%]
             .52,     #Share HVO gasoline [%]
             0.1,     #Share ethanol gasoline [%]
             0,     #Transport efficient society light vehicles [% reduction]
             0,     #Transport efficient society trucks [% reduction]
             0,     #km-tax light vehicles [SEK/km]
             0)     #km-tax trucks [SEK/km]
    policy3=(0,     #Additional energy efficiency light vehicles [%]
             0,  #Additional energy efficiency trucks [%]
             2,  #Bus energy consumption [0,1,2]
             .05,     #Change in fuel tax diesel [%/y]
             .05,      #Change in fuel tax gasoline [%/y]  
             1,     #ICE CO2 reduction ambition level [0,1]
             .07,     #Share FAME diesel [%]
             .54,     #Share HVO diesel [%]
             .54,     #Share HVO gasoline [%]
             0.1,     #Share ethanol gasoline [%]
             0,     #Transport efficient society light vehicles [% reduction]
             0,     #Transport efficient society trucks [% reduction]
             1,     #km-tax light vehicles [SEK/km]
             2)     #km-tax trucks [SEK/km]
    policy4=(0,     #Additional energy efficiency light vehicles [%]
             0,  #Additional energy efficiency trucks [%]
             2,  #Bus energy consumption [0,1,2]
             .097,     #Change in fuel tax diesel [%/y]
             .097,      #Change in fuel tax gasoline [%/y]  
             1,     #ICE CO2 reduction ambition level [0,1]
             .07,     #Share FAME diesel [%]
             .41,     #Share HVO diesel [%]
             .41,     #Share HVO gasoline [%]
             0.1,     #Share ethanol gasoline [%]
             0,     #Transport efficient society light vehicles [% reduction]
             0,     #Transport efficient society trucks [% reduction]
             1,     #km-tax light vehicles [SEK/km]
             2)     #km-tax trucks [SEK/km]
    policy5=(0,     #Additional energy efficiency light vehicles [%]
             0,  #Additional energy efficiency trucks [%]
             2,  #Bus energy consumption [0,1,2]
             .12,     #Change in fuel tax diesel [%/y]
             .12,      #Change in fuel tax gasoline [%/y]  
             1,     #ICE CO2 reduction ambition level [0,1]
             .07,     #Share FAME diesel [%]
             .34,     #Share HVO diesel [%]
             .34,     #Share HVO gasoline [%]
             0.1,     #Share ethanol gasoline [%]
             0,     #Transport efficient society light vehicles [% reduction]
             0,     #Transport efficient society trucks [% reduction]
             0,     #km-tax light vehicles [SEK/km]
             0)     #km-tax trucks [SEK/km]
    policy6=(0.05,     #Additional energy efficiency light vehicles [%]
             0.05,  #Additional energy efficiency trucks [%]
             2,  #Bus energy consumption [0,1,2]
             .031,     #Change in fuel tax diesel [%/y]
             .031,      #Change in fuel tax gasoline [%/y]  
             1,     #ICE CO2 reduction ambition level [0,1]
             .07,     #Share FAME diesel [%]
             .58,     #Share HVO diesel [%]
             .55,     #Share HVO gasoline [%]
             0.1,     #Share ethanol gasoline [%]
             .10,     #Transport efficient society light vehicles [%]
             .050,     #Transport efficient society trucks [%]
             .50,     #km-tax light vehicles [SEK/km]
             1)     #km-tax trucks [SEK/km]
    policy7=(.05,     #Additional energy efficiency light vehicles [%]
             .05,  #Additional energy efficiency trucks [%]
             2,  #Bus energy consumption [0,1,2]
             .092,     #Change in fuel tax diesel [%/y]
             .092,      #Change in fuel tax gasoline [%/y]  
             1,     #ICE CO2 reduction ambition level [0,1]
             .07,     #Share FAME diesel [%]
             .425,     #Share HVO diesel [%]
             .425,     #Share HVO gasoline [%]
             0.1,     #Share ethanol gasoline [%]
             .10,     #Transport efficient society light vehicles [% reduction]
             .050,     #Transport efficient society trucks [% reduction]
             .50,     #km-tax light vehicles [SEK/km]
             1)     #km-tax trucks [SEK/km]
    policy8=(.05,     #Additional energy efficiency light vehicles [%]
             .05,  #Additional energy efficiency trucks [%]
             2,  #Bus energy consumption [0,1,2]
             .063,     #Change in fuel tax diesel [%/y]
             .063,      #Change in fuel tax gasoline [%/y]  
             1,     #ICE CO2 reduction ambition level [0,1]
             .07,     #Share FAME diesel [%]
             .438,     #Share HVO diesel [%]
             .438,     #Share HVO gasoline [%]
             0.1,     #Share ethanol gasoline [%]
             .26,     #Transport efficient society light vehicles [% reduction]
             .17,     #Transport efficient society trucks [% reduction]
             .50,     #km-tax light vehicles [SEK/km]
             1)     #km-tax trucks [SEK/km]
    policy9=(0,     #Additional energy efficiency light vehicles [%] ### SET VALUES FOR REFERENCE SCENARIO
             0,  #Additional energy efficiency trucks [%]
             0,  #Bus energy consumption [0,1,2]
             .02,     #Change in fuel tax diesel [%/y]
             .02,      #Change in fuel tax gasoline [%/y]  
             0,     #ICE CO2 reduction ambition level [0,1]
             .07,     #Share FAME diesel [%]
             .25,     #Share HVO diesel [%]
             0,     #Share HVO gasoline [%]
             0.075,     #Share ethanol gasoline [%]
             0,     #Transport efficient society light vehicles [% reduction]
             0,     #Transport efficient society trucks [% reduction]
             0,     #km-tax light vehicles [SEK/km]
             0)     #km-tax trucks [SEK/km]

all_policies=[policy1,policy2,policy3,policy4,policy5,policy6,policy7,policy8, policy9]
policies.designs=all_policies
policy_names=["B  Bio fuels",
              "C1 High fuel tax, biofuels <20TWh",
              "C2 Fuel and km-tax, biofuels <20TWh",
              "C3 High fuel and km-tax, biofuels <13TWh",
              "C4 High fuel tax, biofuels <13TWh",
              "D1 Transport efficiency, fuel and km-tax, biofuels <20TWh",
              "D2 High transport efficiency, high fuel and km-tax, biofuels <13TWh",
              "D3 High transport efficiency, high fuel and km-tax, biofuels <13TWh",
              "Reference - planned policies"]

#%% Create a df with policies
df_policies=pd.DataFrame(all_policies,columns=policies.params)
df_policies["policy"]=policy_names
#%%
#df_policies["policy"]=policy_names
# count=0
# for i in policies.params:
#     df_policies[str(i)]=policies.designs[str(i)]
# df_policies=df_policies.drop_duplicates()
# df_policies.reset_index(drop=True, inplace=True)




# # #%%Visualize policies
# #Plot table of policies
# fixed_policies=True 
# if fixed_policies: 
#     policy_vis=False
#     if policy_vis:
#         fig, ax = plt.subplots()
#         # hide axes
#         fig.patch.set_visible(False)
#         ax.axis('off')
#         ax.axis('tight')
#         table=ax.table(cellText=df_policies.values, colLabels=df_policies.columns, loc='center')
#         table.auto_set_font_size(False)
#         table.set_fontsize(6)
#         fig.tight_layout()
#         plt.show()
#         # Visualize each policy as a dot diagram
    
#         sns.set_theme(style="whitegrid")
#         sns.set(font_scale=2)
#         g = sns.PairGrid(df_policies.sort_values("policy", ascending=False),
#                          x_vars=df_policies.columns[1:], y_vars=["policy"],
#                          palette="colorblind", height=15, aspect=.15)
        
#         g.map(sns.stripplot, size=8, orient="h", jitter=False, linewidth=1,
#               edgecolor="w")
#         for ax, title in zip(g.axes.flat, df_policies.columns[1:]):
#             # Set a different title for each axes
#             ax.set(title=title)
#             # Make the grid horizontal instead of vertical
#             ax.xaxis.grid(False)
#             ax.yaxis.grid(True)
#             ax.tick_params(axis='x', rotation=90)
#         sns.despine(left=True, bottom=True)
#         sns.set(font_scale=1)
#%%Parcoords plot, requires manual work 

df_policies_parcoords=df_policies
# Rearrange columns in the desired order

column_order = ['ICE CO2 reduction ambition level', 'Bus energy consumption', 
                'Share FAME diesel','Share HVO diesel', 'Share HVO gasoline', 'Share ethanol gasoline',  
                'Change in fuel tax diesel', 'Change in fuel tax gasoline'
                , 'km-tax light vehicles','km-tax trucks',
                'Additional energy efficiency light vehicles', 'Additional energy efficiency trucks',
                'Transport efficient society light vehicles','Transport efficient society trucks',
                'policy']  # Adjust the order as needed
df_policies_parcoords = df_policies_parcoords[column_order]

from ema_workbench.analysis import parcoords

# Get limits and create a parallel axes plot
limits = parcoords.get_limits(df_policies_parcoords.drop(columns=["policy"]))
limits.loc[0]=0 #Set all lower bounds to 0
paraxes = parcoords.ParallelAxes(limits)

# Plot the data with thicker lines
paraxes.plot(df_policies_parcoords, linewidth=4, alpha=0.7)

# Add legend and show the plot

# Add legend below the plot
plt.legend(df_policies["policy"], loc='center left', bbox_to_anchor=(1.0, 0.5),
           fancybox=False, shadow=False, ncol=1,fontsize=16)
plt.show()




# #%% Radar chart with plotly express
# import pandas as pd
# import plotly.express as px
# from plotly.offline import plot
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# df = df_policies.drop(columns=["policy"])
# levers = df.columns

# # Create a grid of polar plots
# n_rows = 2
# n_cols = (len(levers) + 1) // 2
# fig = make_subplots(rows=n_rows, cols=n_cols, specs=[[{'type': 'polar'}] * n_cols] * n_rows)

# # Add traces for each row (policy) with a different color
# colors = px.colors.qualitative.Plotly
# for index, row in df.iterrows():
#     for i, lever in enumerate(levers):
#         r_values = [0, row[lever], 0]
#         theta_values = [0, i, i + 1]

#         fig.add_trace(go.Scatterpolar(
#             r=r_values,
#             theta=theta_values,
#             mode='lines',
#             name=f'Policy {index}' if i == 0 else None,
#             legendgroup=f'Policy {index}',
#             showlegend=True if i == 0 else False,
#             fill='toself',
#             line=dict(color=colors[index % len(colors)]),
#         ), row=i // n_cols + 1, col=i % n_cols + 1)

# # Update layout
# fig.update_layout(
#     title='Radar Chart for Policies',
#     showlegend=True
# )

# # Update radial and angular axis properties for each subplot
# for i, lever in enumerate(levers):
#     max_value = df[lever].max() * 1.1
#     fig.update_polars(
#         radialaxis=dict(range=[0, max_value], visible=True),
#         angularaxis=dict(tickvals=[i], ticktext=[lever]),
#         row=i // n_cols + 1, col=i % n_cols + 1
#     )

# # Plot the radar chart
# plot(fig)



