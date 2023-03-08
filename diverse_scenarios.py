# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:00:57 2023

@author: aengholm
"""
#Load a scenario database
import pickle
import math
scenarios=pickle.load( open('./output_data/'+"scenario_df.p", "rb" ))

#Normalize all scenario variables
scenarios_norm_mean=(scenarios-scenarios.mean())/scenarios.std()

scenarios_norm=(scenarios-scenarios.min())/(scenarios.max()-scenarios.min())

#Create a matrix of all euclidian distances
import pandas as pd
distance_matrix=pd.DataFrame()

for i,row_i in scenarios.iterrows():
    for j,row_j in scenarios.iterrows():
        if i!=j:
            dist_i_j=math.dist(list(row_i),list(row_j))
            distance_matrix.loc[i,j]=dist_i_j

#%% works reasonably for up to a handful of scenarios
n_scenarios=3
import itertools
all_possible_scenario_sets=itertools.combinations(range(len(scenarios_norm)),n_scenarios)
all_possible_scenario_sets_list=list(all_possible_scenario_sets)
set_scores=[]
for scenario_set in all_possible_scenario_sets_list:
    # All possible pairs in List
    # Using list comprehension + enumerate()
    res = [(a, b) for idx, a in enumerate(scenario_set) for b in scenario_set[idx + 1:]]
     
    # printing result 
    #print("All possible pairs : " + str(res))
    set_distances=[]
    for i in res:
        set_distances.append(distance_matrix.loc[i])
    min_d_set=min(set_distances)
    mean_d_set=sum(set_distances)/len(set_distances)
    a=0.75
    set_score=a*min_d_set+(1-a)*mean_d_set
    set_scores.append(set_score)

max_score = max(set_scores)
max_set_i = set_scores.index(max_score)
max_set=all_possible_scenario_sets_list[max_set_i]
#%%
#Create a df with max_set
scenario_set_diverse_df=pd.DataFrame(columns=scenarios.columns,index=[])
for i in max_set:
    scenario=scenarios.iloc[i]
    scenario_set_diverse_df_1=pd.DataFrame(scenario).transpose()
    scenario_set_diverse_df=pd.concat([scenario_set_diverse_df,scenario_set_diverse_df_1])
scenario_set_diverse_df.index=range(len(scenario_set_diverse_df))
pickle.dump(scenario_set_diverse_df,open("./output_data/"+"diverse_scenarios_"+str(n_scenarios)+".p","wb"))
