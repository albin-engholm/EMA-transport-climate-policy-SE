# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:25:15 2022

@author: aengholm
"""


load_results=1
if load_results==1:
    date="2022-03-30"
    t1='./output_data/'+"robust_search"+date+"robust_results_.p"
    import pickle
    robust_results=pickle.load( open(t1, "rb" ) )


#plot distributions of policies and robust results
import seaborn as sns
#sns.displot(data=robust_results)
#sns.pairplot(data=robust_results)
scaled_df=robust_results