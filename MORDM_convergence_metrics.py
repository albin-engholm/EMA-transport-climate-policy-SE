# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:43:27 2023
A script for computing and visualizing MOEA convergence metrics for the latest saved archive (specified by filename)
@author: aengholm
"""

from ema_workbench.em_framework.optimization import (HypervolumeMetric,
                                                    EpsilonProgress,
                                                    ArchiveLogger,epsilon_nondominated) 
import pickle
import pandas as pd
import matplotlib.pyplot as plt
#%% Load data

#policy_types=["All levers", "No transport efficient society"]
#policy_types=["No transport efficient society"]#,
policy_types=["All levers"]
date='2023-12-06'#Specify date the MORDM MOEA was completed
nfe=100 #Specify the number of nfes used for the MORDM MOEA
                
metrics=True
if metrics:
    all_archives = []
    
    for policy_type in policy_types:
        archives = ArchiveLogger.load_archives(f"./archives/{str(nfe)}_{policy_type}_{date}.tar.gz") #load archive
        model_filename = './output_data/'+policy_type + str(nfe)+"_nfe_"+"directed_search_MORDM_"+date+"model_.p"
        model = pickle.load(open(model_filename, "rb")) #Load model
        results_filename = './output_data/'+policy_type+str(nfe)+"_nfe_"+"directed_search_MORDM_"+date+".p"
        results_list, convergence_list, scenarios,epsilons = pickle.load(open(results_filename, "rb"))
        results_final=results_list[0]
        updated_archives = {}
        
        # Check and drop the 'unnamed' column in each dataframe if it exists
        for key, dataframe in archives.items():
            if "Unnamed: 0" in dataframe.columns:
                # inplace=True: modify the DataFrame in place (do not create a new object)
                dataframe=dataframe.drop(columns=['Unnamed: 0'])
    
            # Add the modified dataframe to the new dictionary
            updated_archives[key] = dataframe
        all_archives.append(updated_archives)
#%% Prepare for convergence calculations        
    from ema_workbench import (
        HypervolumeMetric,
        GenerationalDistanceMetric,
        EpsilonIndicatorMetric,
        InvertedGenerationalDistanceMetric,
        SpacingMetric,
    )
    from ema_workbench.em_framework.optimization import to_problem
    # Create a new list of outcomes excluding INFO (0)  kind
    new_outcomes = [o for o in model.outcomes if o.kind!=0]
    problem_model = model
    problem_model.outcomes=new_outcomes
    problem = to_problem(problem_model, searchover="levers")
    
    results_epsilon=[results_final]
    #reference_set = epsilon_nondominated((results_epsilon), epsilons, problem) #Not required since results are already non-dominated solutions. This is just required if solutions merged from multiple random seeds are used
    reference_set=results_epsilon[0]
    hv = HypervolumeMetric(reference_set, problem)
    gd = GenerationalDistanceMetric(reference_set, problem, d=1)
    ei = EpsilonIndicatorMetric(reference_set, problem)
    ig = InvertedGenerationalDistanceMetric(reference_set, problem, d=1)
    sm = SpacingMetric(problem)
    #%%Compute and visualize metrics
    metrics_by_seed = []
    n_archives=len
    
    for archives in all_archives:
        metrics = []
        counter=0
        for nfe, archive in archives.items():
            #Remove archive columns with INFO outcomes
            for outcome in model.outcomes:
                if (outcome.kind == 0) & (outcome in archive.columns) : #0 means INFO
                    archive = archive.drop(columns=[outcome.name])
            #if counter <n_archives:
            print("Generation: " + str(counter)+", nfe: "+str(nfe)+", # solutions: "+str(len(archive)))
            scores = {
                
                "generational_distance": gd.calculate(archive),
                "hypervolume": hv.calculate(archive),
                "epsilon_indicator": ei.calculate(archive),
                "inverted_gd": ig.calculate(archive),
                "spacing": sm.calculate(archive),
                "nfe": int(nfe),
            }
            metrics.append(scores)
            counter=counter+1
            # else:
            #     break
        metrics = pd.DataFrame.from_dict(metrics)
    
        # sort metrics by number of function evaluations
        metrics.sort_values(by="nfe", inplace=True)
        metrics_by_seed.append(metrics)
    #%% visualize metrics    
    import seaborn as sns
    sns.set_style("white")
    fig, axes = plt.subplots(nrows=6, figsize=(8, 12), sharex=True)
    
    ax1,ax2, ax3, ax4, ax5, ax6 = axes
    
    for metrics, convergence in zip(metrics_by_seed, convergence_list):
        ax1.plot(metrics.nfe, metrics.hypervolume)
        ax1.set_ylabel("hypervolume")
    
        ax2.plot(convergence.nfe, convergence.epsilon_progress)
        ax2.set_ylabel("$\epsilon$ progress")
    
        ax3.plot(metrics.nfe, metrics.generational_distance)
        ax3.set_ylabel("generational distance")
    
        ax4.plot(metrics.nfe, metrics.epsilon_indicator)
        ax4.set_ylabel("epsilon indicator")
    
        ax5.plot(metrics.nfe, metrics.inverted_gd)
        ax5.set_ylabel("inverted generational\ndistance")
    
        ax6.plot(metrics.nfe, metrics.spacing)
        ax6.set_ylabel("spacing")
    
    ax6.set_xlabel("nfe")
    
    sns.despine(fig)
    
    plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, figsize=(8,4))
ax1.plot(convergence.nfe, convergence.epsilon_progress)
ax1.set_ylabel('$\epsilon$-progress')
# ax2.plot(convergence.nfe, convergence.hypervolume)
# ax2.set_ylabel('hypervolume')

ax1.set_xlabel('number of function evaluations')
# ax2.set_xlabel('number of function evaluations')
