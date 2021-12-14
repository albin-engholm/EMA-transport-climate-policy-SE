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

from ema_workbench import (RealParameter, CategoricalParameter, 
                           ScalarOutcome, ema_logging,
                           perform_experiments)

from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)

    #Set model
    model = ExcelModel("scenarioModel", wd="./models",
                       model_file='trv_scenario2.xlsx')
    model.default_sheet = "Indata"
    
    
     #%% Specify inputsde
    #Set parametric uncetainties
    model.uncertainties = [RealParameter("Heavy truck el share",
                                         .05,0.7
                                         ,variable_name="D79")
                          ,RealParameter("Truck demand change",
                                         -0.3,0.30,
                                         variable_name="F61")
                          ,RealParameter("Car el share",
                                         0.1,0.7,
                                         variable_name="F28")
                          ,RealParameter("Car demand change",
                                         -0.3,0.3
                                         ,variable_name="F38")
                          ]
    #Specification of levers
    model.levers = [RealParameter("Share HVO gasoline", 
                                  0, 0.7,
                                  variable_name="F10"),
                    RealParameter("Share HVO diesel",
                                  0.25, 0.80,
                                  variable_name="F12"),
                    RealParameter("km-tax light vehicles",
                                  0,3
                                  ,variable_name="F15"),
                    RealParameter("km-tax trucks",
                                  0,3
                                  ,variable_name="F16"),
                    CategoricalParameter("ICE CO2 reduction ambition level",
                                         ["1 Beslutad politik","2 Mer ambitios politik"]
                                         ,variable_name="F17")
                    ]
    # specification of the outcomes
    model.outcomes = [ScalarOutcome("CO2 TTW change trucks",
                                    variable_name="D66"),
                      ScalarOutcome("CO2 TTW change total", 
                                    variable_name="D67"),
                      ScalarOutcome("VKT trucks", 
                                    variable_name="D68"),
                      ScalarOutcome("Energy tot", 
                                    variable_name="D69"),
                      ScalarOutcome("Energy total", 
                                    variable_name="D70"),
                      ScalarOutcome("Energy bio total", 
                                    variable_name="D71"),
                      ScalarOutcome("Energy fossile total", 
                                    variable_name="D72"),
                      ScalarOutcome("Energy el total",
                                    variable_name="D73")
                      #ScalarOutcome("Total cost trucks",
                                    #variable_name="D74"),
                      #ScalarOutcome("Total cost cars",
                                  #  variable_name="D75")
                      ]  
    
    # Specify policies
   # from ema_workbench.em_framework import samplers
    n_policies=100
    #policies=samplers.sample_levers(model, n_policies, sampler=samplers.LHSSampler())       
    #%%
    #select number of scenarios (per policy)
    nr_scenarios=1000
    
    #Run model - for open exploration
    
    #Simulation settings
    run_with_policies=1
    use_multi=1
    n_p=4
    
    #Run
    import time
    tic=time.perf_counter()
    if run_with_policies == 1:
        if use_multi==1:
            with MultiprocessingEvaluator(msis=model,n_processes=n_p) as evaluator:
                experiments, outcomes = perform_experiments(model, 
                                                            nr_scenarios, 
                                                            reporting_frequency=100, 
                                                            evaluator=evaluator,
                                                            policies=n_policies)            
        else:
            experiments, outcomes = perform_experiments(model, nr_scenarios,
                                                        policies=n_policies,
                                                        reporting_frequency=100)
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
         # Save results?
    save_results=1
    if save_results==1:
        from datetime import date    
        from ema_workbench import save_results
        if run_with_policies==1:
            filename=str(nr_scenarios)+'_scenarios_'+str(n_policies)+"_policies_"+str(date.today())
        else:
            filename=str(nr_scenarios)+'_scenarios_'+str(date.today())
        filename1=filename+'.tar.gz'
        save_results([experiments,outcomes], "./output_data/"+filename1)
        import pickle
        filename2=filename+'model_'+".p"
        pickle.dump(model,open("./output_data/"+filename2,"wb"))
