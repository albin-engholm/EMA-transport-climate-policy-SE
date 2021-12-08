# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:20:38 2021

@author: aengholm
"""

import numpy as np
import pandas as pd
from ema_workbench import (RealParameter, TimeSeriesOutcome, ScalarOutcome, ema_logging,
                           perform_experiments)

import matplotlib.pyplot as plt
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

from ema_workbench.analysis.pairs_plotting import (pairs_lines, pairs_scatter,
                                               pairs_density)
from ema_workbench.analysis import feature_scoring
import seaborn as sns
sns.set(rc={"figure.dpi":300})

#Should previously saved result data be loaded? If not, data from workspace is used
load_results=1
if load_results==1:
    from ema_workbench import load_results
    results = load_results('3000_scenarios_2021-12-08.tar.gz')
    experiments=results[0]
    outcomes=results[1]

### SCENARIO EXPLORATION###
sns.set_palette("deep")


### Pairwise scatter plots on outcomes
fig,axes = pairs_scatter(experiments,outcomes, legend=True, group_by="policy")
fig.set_size_inches(15,15)
plt.show()


# plt.xticks(rotation=45) #rotate x-axis labels by 45 degrees.
# plt.yticks(rotation=45) #rotate y-axis labels by 90 degrees.

### Show hist/KDE of 

# plt.axvline(-0.7, color='r', linestyle='dashed', linewidth=1)
# plt.axhline(20, color='r', linestyle='dashed', linewidth=1)

###FEATURE SCORING ON OUTCOMES

fs = feature_scoring.get_feature_scores_all(experiments, outcomes)
plt.figure()
fig=sns.heatmap(fs, cmap='viridis', annot=True,fmt=".2f")



### scenario discovery

#Define criterion for unwanted outcome
fail_criterion_CO2=-0.7
fail_criterion_bio=20
#Prepare data, x and y arrays
x = experiments
y1=outcomes['CO2 change tot']>fail_criterion_CO2
y2=outcomes['Energy bio']>fail_criterion_bio
y3=[0]*len(x)
for j in range(len(y1)):
    if y1[j]==True or y2[j]==True:
        y3[j]=True
    else: 
        y3[j]=False
#choose criterion to use y1=first criterion only, y2=second criterion only,y3=either first or second criterion=fail
y=np.array(y3,dtype=bool)


#Basic statistics 
n_fail = np.count_nonzero(y)
share_fail=n_fail/len(y)
share_success=1-share_fail

import statistics
#Plot hist/KDE on criterions
sns.displot(x='CO2 change tot', data=outcomes, kde=True)
plt.axvspan(fail_criterion_CO2, max(outcomes['CO2 change tot']), facecolor='red', alpha=0.2,edgecolor='None')
plt.axvline(statistics.mean(outcomes['CO2 change tot']),color="red")

sns.displot(x='Energy bio', data=outcomes, kde=True)
plt.axvspan(fail_criterion_bio, max(outcomes['Energy bio']), facecolor='red', alpha=0.2,edgecolor='None')
plt.axvline(statistics.mean(outcomes['Energy bio']),color="red")

g=sns.displot(x='CO2 change tot', y='Energy bio', data=outcomes)
ylim=g.ax.get_ylim()
xlim=g.ax.get_xlim()
plt.axvspan(fail_criterion_CO2, xlim[1],facecolor='red', alpha=0.2, edgecolor='none')
plt.axvspan(xlim[0],fail_criterion_CO2,fail_criterion_bio/ylim[1],facecolor='red', alpha=0.2,edgecolor='none')




# Feature scoring on scenario disocvery data (what ucnertainties drive fail/success of outcomes)
fs_discovery, alg = feature_scoring.get_ex_feature_scores(x, y, mode=feature_scoring.RuleInductionType.CLASSIFICATION)
fs_discovery.sort_values(ascending=False, by=1)
plt.figure()
fig=sns.heatmap(fs_discovery, cmap='viridis', annot=True)

#Scatter pair plot for all uncertainties and whether or not target is met
x_copy2 = experiments.copy()
x_copy2= x_copy2.drop('scenario', axis=1)
x_copy2["Target not met"]=y
g= sns.PairGrid(x_copy2, hue="Target not met")
g.map_diag(sns.kdeplot, shade=True)
g.map_offdiag(plt.scatter,edgecolor="white",alpha=0.5)
g.add_legend()
g.fig.set_size_inches(20,20)

#Dimensional stacking
from ema_workbench.analysis import dimensional_stacking
dimensional_stacking.create_pivot_plot(x,y, 2, nbins=5)
plt.show()

#Regional sensitivty analysis, # model is the same across experiments
from ema_workbench.analysis import regional_sa
from numpy.lib import recfunctions as rf
sns.set_style('white')
x_copy = experiments.copy()
x_copy = x_copy.drop('model', axis=1)
x_copy = x_copy.drop('policy', axis=1)
fig = regional_sa.plot_cdfs(x_copy,y,ccdf=False)
sns.despine()
plt.show()

#Perform PRIM analysis
from ema_workbench.analysis import prim
prim_alg = prim.Prim(x, y, threshold=0.5)
#Find 1st box
box1 = prim_alg.find_box()
#%%%
#Visualizations of Box1
box1.show_tradeoff()

for i in range(0,len(box1.peeling_trajectory.T.columns)):
    s=box1.peeling_trajectory.T[i].id
    if (i%2)==0:
        plt.text(box1.peeling_trajectory.T[i].coverage+.02,box1.peeling_trajectory.T[i].density+.03 ,s,fontsize=10)
    else:
        plt.text(box1.peeling_trajectory.T[i].coverage-.03,box1.peeling_trajectory.T[i].density-.02 ,s,fontsize=10)
plt.show()
#Choose the last point
i1=len(box1.peeling_trajectory.T.columns)-1
box1.inspect(i1)
box1.inspect(i1, style='graph')
plt.show()
ax=box1.show_pairs_scatter(i1)

plt.show()

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