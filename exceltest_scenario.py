# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:45:55 2021

@author: aengholm
This is a script for setting up and running an EMA analysis using a static 
excel model with parametric uncertainties(designed for the TRV scenario model)
Current version has not implement policies for the specific model
Results are saved and can be loaded for analysis in separate script - 
e.g. scenario_exploration_excel_static.py'
""" 
import numpy as np
import pandas as pd
from ema_workbench import (RealParameter, TimeSeriesOutcome, ScalarOutcome, ema_logging,
                           perform_experiments)

from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)

    #Set model
    model = ExcelModel("scenarioModel", wd="./models",
                       model_file='trv_scenario2.xlsx')
    model.default_sheet = "Indata"
    
    #Set parametric uncetainties
    model.uncertainties = [RealParameter("D79",.05,0.7)
                          ,RealParameter("F61",-0.3,0.30)
                          ,RealParameter("F28",0.1,0.7)
                          ,RealParameter("F38",-0.3,0.3)
                          ]
    
    #Specification of levers
    model.levers = [RealParameter("F10", 0, 0.7),
                    RealParameter("F12", 0.25, 0.80),
                    RealParameter("F15", 0,1),
                    RealParameter("F16", 0,1)
                    ]

    # specification of the outcomes
    model.outcomes = [ScalarOutcome("D66"),
                      ScalarOutcome("D67"),
                      ScalarOutcome("D68"),
                      ScalarOutcome("D69"),
                      ScalarOutcome("D70"),
                      ScalarOutcome("D71"),
                      ScalarOutcome("D72"),
                      ScalarOutcome("D73"),
                      #ScalarOutcome("D74"),
                      #ScalarOutcome("D75")
                      ]  
    
    # Add policies
    # policies = [Policy('Low ambition',
    #                    model_file=r'FLUvensimV1basecase.vpm'),
    #             Policy('High ambition',
    #                    model_file=r'FLUvensimV1static.vpm'),
    #                      ]
    from ema_workbench.em_framework import samplers
    n_policies=5
    policies=samplers.sample_levers(model, n_policies, sampler=samplers.LHSSampler())       
    #%%
    #select number of scenarios (per policy)
    nr_scenarios=1000
    
    #Run model - for open exploration
    import time
    tic=time.perf_counter()
    run_with_policies=1
    use_multi=1
    n_p=3
    if run_with_policies == 1:
        if use_multi==1:
            with MultiprocessingEvaluator(msis=model,n_processes=n_p) as evaluator:
                experiments, outcomes = perform_experiments(model, 
                                                            nr_scenarios, 
                                                            reporting_frequency=100, 
                                                            evaluator=evaluator,policies=policies)            
        else:
            experiments, outcomes = perform_experiments(model, nr_scenarios,policies=policies)
    else:
        ### Run model using multiprocessing
        if use_multi == 1:
            with MultiprocessingEvaluator(msis=model,n_processes=n_p) as evaluator:
                experiments, outcomes = perform_experiments(model, 
                                                            nr_scenarios, 
                                                            reporting_frequency=100, 
                                                            evaluator=evaluator)
        else:
            experiments, outcomes = perform_experiments(model, nr_scenarios, 
                                                        reporting_frequency=100)
            
    toc=time.perf_counter()
    print("Runtime [s]= " +str(toc-tic))
    print("Runtime [h]= " +str(round((toc-tic)/3600,1)))
    print("Runtime per experiment [s]= " +str((toc-tic)/len(experiments)))
    
    #%%
    #Run model for wost case discovery
    run_worstcase=0
    if run_worstcase ==1:
        
        model_worstcase=model
        model_worstcase.outcomes.clear()
        # specification of the outcomes forworst case. Set desirable direction
        model_worstcase.outcomes = [ScalarOutcome("D66", ScalarOutcome.MINIMIZE),
                                    ScalarOutcome("D67",ScalarOutcome.MINIMIZE),
                                    ScalarOutcome("D79",ScalarOutcome.MINIMIZE)
                                    ]  
        
        # change outcomes so direction is undesirable
        minimize = ScalarOutcome.MINIMIZE
        maximize = ScalarOutcome.MAXIMIZE
        for outcome in model_worstcase.outcomes:
            if outcome.kind == minimize:
                outcome.kind = maximize
            else:
                outcome.kind = minimize
        ## Run worstcase scenario
        from ema_workbench.em_framework.optimization import (HyperVolume,
                                                     EpsilonProgress)

        convergence_metrics = [HyperVolume(minimum=[0,0,0,0], maximum=[1,1.01,1.01,1.01]),
                       EpsilonProgress()]
        with MultiprocessingEvaluator(model_worstcase) as evaluator:
            results_worstcase, convergence = evaluator.optimize(nfe=200, searchover='uncertainties',
                                         epsilons=[0.1,]*len(model_worstcase.outcomes),
                                         convergence=convergence_metrics)
        #%%
        from ema_workbench.analysis import parcoords
        import matplotlib.pyplot as plt


        fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, figsize=(8,4))
        ax1.plot(convergence.nfe, convergence.epsilon_progress)
        ax1.set_ylabel('$\epsilon$-progress')
        ax2.plot(convergence.nfe, convergence.hypervolume)
        ax2.set_ylabel('hypervolume')
        
        ax1.set_xlabel('number of function evaluations')
        ax2.set_xlabel('number of function evaluations')
        plt.show()
        data = results_worstcase.loc[:, [o.name for o in model_worstcase.outcomes]]
        data = data.iloc[: , :-1]
        limits = parcoords.get_limits(data)
        limits.loc[0, ['D66', 'D67', 'D79']] = 0
        limits.loc[1, ['D66', 'D67', 'D79']] = 1
        
        paraxes = parcoords.ParallelAxes(limits)
        paraxes.plot(data)
        #paraxes.invert_axis('max_P')
        plt.show()

     
     #%%

    #Rename outcome variables
    outcomes["CO2 change truck"] = outcomes.pop("D66")
    outcomes["CO2 change tot"] = outcomes.pop("D67")
    outcomes["VKT trucks"] = outcomes.pop("D68")
    outcomes["VKT tot"] = outcomes.pop("D69")
    outcomes["Energy tot"] = outcomes.pop("D70")
    outcomes["Energy bio"] = outcomes.pop("D71")
    outcomes["Energy fossile"] = outcomes.pop("D72")
    outcomes["Energy el"] = outcomes.pop("D73")
    #outcomes["Total cost trucks"] = outcomes.pop("D74")
    #outcomes["Total costs cars"] = outcomes.pop("D75")
    #Rename input variables
    # Uncertainties
  
    experiments["Heavy truck el. share"] = experiments.pop("D79")
    experiments["Truck demand change"] = experiments.pop("F61")
    experiments["Car demand change"] = experiments.pop("F38")
    experiments["Car el share"] = experiments.pop("F28")
    
    #levers
    if run_with_policies==1:
        experiments["share HVO gas."] = experiments.pop("F10")
        experiments["share HVO diesel"] = experiments.pop("F12")
        experiments["km-tax light veh"] = experiments.pop("F15")
        experiments["km-tax trucks"] = experiments.pop("F16")
    
         # Save results?
    save_results=1
    if save_results==1:
        from datetime import date    
        from ema_workbench import save_results
        filename=str(nr_scenarios)+'_scenarios_'+str(date.today())+'.tar.gz'
        save_results([experiments,outcomes], filename)
        #df.to_hdf('data.h5', key='df', mode='w')    


