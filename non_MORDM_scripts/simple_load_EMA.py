# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:13:01 2023
A script with generic functionality to load data generated from EMA runs. Separated into two parts, 1 to load experiment/results/convergence/scenario files, 2 to load model file
@author: aengholm
"""

import pandas as pd
df_full=pd.DataFrame()

#load results, note that format may matter depending on analysis
filename1="./output_data/60000_nfe_directed_search_worst_best_case_2023-05-16.p"
# =str(nfe)+'_nfe_directed_search_sequential_'+str(date.today())+'_'+str(n_scenarios)+'_scenarios'
import pickle
results,convergence=pickle.load( open(filename1, "rb" ))
#%%
#Load model
t2='./output_data/'+policy_type+str(nfe)+"_nfe_"+"directed_search_sequential_"+date+"_"+str(n_scenarios)+"_scenarios"+"model_.p"
model=pickle.load( open(t2, "rb" ) )


scenario_count=0
for results in results_list:
    results["Scenario"]=scenario_count
    results["Policy type"]=policy_type
    df_full=pd.concat([df_full,results],ignore_index=True)
    scenario_count=scenario_count+1