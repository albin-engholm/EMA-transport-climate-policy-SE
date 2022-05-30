# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:05:08 2022

@author: aengholm
This is a script for setting up and running robust search over policy levers for a static 
excel model with parametric uncertainties. It is designed for the TRV scenario model.
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
    
    
     #%% Specify inputs
    #Set parametric uncetainties
    model.uncertainties = [RealParameter("Heavy truck el share",
                                         .05,0.7
                                         ,variable_name="D79")
                          ,RealParameter("Truck demand change",
                                         -0.2,0.20,
                                         variable_name="F61")
                          ,RealParameter("Car el share",
                                         0.1,0.8,
                                         variable_name="F28")
                          ,RealParameter("Car demand change",
                                         -0.2,0.2
                                         ,variable_name="F38")
                          ,RealParameter("Fossile price adjustment",
                                         0.8,2,
                                         variable_name="Modell - Drivmedelpriser!G5")
                          ,RealParameter("Biofuel price adjustment",
                                         0.8,3,
                                         variable_name="Modell - Drivmedelpriser!G6")
                          ,RealParameter("Electricity price adjustment",
                                         .5,3
                                         ,variable_name="Modell - Drivmedelpriser!G4")
                          # ,RealParameter("Truck demand elasticity",
                          #                -.5,-.1
                          #                ,variable_name="D78")
                          # ,RealParameter("Car demand elasticity",
                          #                -.5,-.1
                          #                ,variable_name="D80")
                          ]
    #Specification of levers
    model.levers = [CategoricalParameter("ICE CO2 reduction ambition level",
                                         ["1 Beslutad politik","2 Mer ambitios politik"]
                                         ,variable_name="F17"),
                    CategoricalParameter("Bus energy consumption",
                                         ["Beslutad politik","Level 1","Level 2"]
                                         ,variable_name="F23"),
                    RealParameter("Share HVO diesel",
                                  0, 0.80,
                                  variable_name="F12"),
                    RealParameter("Share FAME diesel",
                                  0, 0.07,
                                  variable_name="F11"),
                    RealParameter("Share HVO gasoline", 
                                  0, 0.7,
                                  variable_name="F10"),
                    RealParameter("Share ethanol gasoline", 
                                  0, 0.1,
                                  variable_name="F11"),
                    RealParameter("km-tax light vehicles",
                                  0,3
                                  ,variable_name="F15"),
                    RealParameter("km-tax trucks",
                                  0,3
                                  ,variable_name="F16"),
                    RealParameter("Change in fuel tax gasoline",
                                  0,.5
                                  ,variable_name="F13"),
                    RealParameter("Change in fuel tax diesel",
                                  0,.5
                                  ,variable_name="F14"),
                    RealParameter("Additional energy efficiency light vehicles",
                                  0,.2
                                  ,variable_name="F19"),
                    RealParameter("Additional energy efficiency trucks",
                                  0,.2
                                  ,variable_name="F20"),
                    RealParameter("Transport efficient society light vehicles",
                                  0,.3
                                  ,variable_name="F21"),
                    RealParameter("Transport efficient society trucks",
                                  0,.3
                                  ,variable_name="F22")
                    ]
    # specification of the outcomes
    model.outcomes = [ScalarOutcome("CO2 TTW change trucks",
                                    variable_name="D66"),
                      ScalarOutcome("CO2 TTW change light vehicles",
                                    variable_name="D74"),
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
                                    variable_name="D73"),
                      ScalarOutcome("VKT light vehicles",
                                    variable_name="Resultat!F30"),
                      ScalarOutcome("VKT total",
                                    variable_name="Resultat!F34"),
                      ScalarOutcome("Driving cost light vehicles",
                                    variable_name="Resultat!F23"),
                      ScalarOutcome("Driving cost trucks",
                                    variable_name="Resultat!F26")
                      #ScalarOutcome("Total cost trucks",
                                    #variable_name="D74"),
                      #ScalarOutcome("Total cost cars",
                                  #  variable_name="D75")
                      ]  

    #%% Create robustness metrics
    import functools
    import numpy as np
    percentile10 = functools.partial(np.percentile, q=10)
    percentile90 = functools.partial(np.percentile, q=90)
    
    MAXIMIZE = ScalarOutcome.MAXIMIZE
    MINIMIZE = ScalarOutcome.MINIMIZE
    INFO =  ScalarOutcome.INFO
    robustnes_functions = [ScalarOutcome('90th percentile driving costs trucks', kind=MINIMIZE,
                                  variable_name='Driving cost trucks', function=percentile90),
                            ScalarOutcome('90th percentile light vehicles', kind=MINIMIZE,
                                  variable_name='Driving cost light vehicles', function=percentile90),
                            ScalarOutcome('maxCO2', kind=MINIMIZE,
                                  variable_name='CO2 TTW change total', 
                                  function= lambda x : max(x)),
                            ScalarOutcome('maxBio', kind=MINIMIZE,
                                  variable_name='Energy bio total', 
                                    function= lambda x : max(x))]
        #%% Specify constraints
    from ema_workbench import Constraint
    def constraint_CO2 (x):
        CO2_target=-.7
        array=np.array(x)
        diff=array-CO2_target
        #diff = [i - CO2_target for i in x]
        x_max=np.amax(diff)
        return max (0,x_max)

    def constraint_bio (x):
        bio_target=15
        array=np.array(x)
        diff=array-bio_target
        x_max=np.amax(diff)
        return max (0,x_max)

    constraints = [Constraint("max CO2", outcome_names="maxCO2",
                              function=constraint_CO2),
                            Constraint("max bio", outcome_names="maxBio",
                          function=constraint_bio)]
    #%%
    #select number of scenarios (per policy)
    from ema_workbench.em_framework import sample_uncertainties

    n_scenarios = 100
    scenarios = sample_uncertainties(model, n_scenarios)
    #Run model - for open exploration
    
    #Simulation settings
    n_p=4
    
    #Run
    import time
    tic=time.perf_counter()
    from ema_workbench import ema_logging
    import matplotlib.pyplot as plt
    from ema_workbench.em_framework.optimization import (HyperVolume,
                                                         EpsilonProgress)
    
    
    ema_logging.log_to_stderr(ema_logging.INFO)
    convergence_metrics = [HyperVolume(minimum=[0,0,0,0], maximum=[10,10,10,10]),
                           EpsilonProgress()]
    
        
    nfe = int(500)
    with MultiprocessingEvaluator(msis=model,n_processes=n_p) as evaluator:
        robust_results,convergence = evaluator.robust_optimize(robustnes_functions,                     
                                                               scenarios,
                                nfe=nfe, epsilons=[0.1,]*len(robustnes_functions),
                                convergence=convergence_metrics,searchover='levers',
                              constraints=constraints)
                
    toc=time.perf_counter()
    print("Runtime [s]= " +str(toc-tic))
    print("Runtime [h]= " +str(round((toc-tic)/3600,1)))
    

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, figsize=(8,4))
    ax1.plot(convergence.nfe, convergence.epsilon_progress)
    ax1.set_ylabel('$\epsilon$-progress')
    ax2.plot(convergence.nfe, convergence.hypervolume)
    ax2.set_ylabel('hypervolume')
    
    ax1.set_xlabel('number of function evaluations')
    ax2.set_xlabel('number of function evaluations')
    plt.show()
    


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

        convergence_metrics = [HyperVolume(minimum=[0,0,0,0], 
                                           maximum=[1,1.01,1.01,1.01]),
                       EpsilonProgress()]
        with MultiprocessingEvaluator(model_worstcase) as evaluator:
            results_worstcase, convergence = evaluator.optimize(nfe=200, 
                                                                searchover='uncertainties',
                                         epsilons=[0.1,]*len(model_worstcase.outcomes),
                                         convergence=convergence_metrics)

#%%
          # Save results?
    save_results=1
    if save_results==1:
        from datetime import date    
        from ema_workbench import save_results
        filename="robust_search"+str(date.today())
        filename1=filename+'.tar.gz'
        # save_results(robust_results, "./output_data/"+filename1)
        import pickle
        filename2=filename+'robust_results_'+".p"
        pickle.dump(robust_results,open("./output_data/"+filename2,"wb"))
