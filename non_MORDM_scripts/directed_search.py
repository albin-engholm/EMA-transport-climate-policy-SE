# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:11:08 2022

@author: aengholm
This is a script for setting up and running directed search over policy levers for a static 
excel model with parametric uncertaintie. It is designed for the TRV scenario model.
Results are saved and can be loaded for analysis in separate script - 
e.g. scenario_exploration_excel_static.py'
"""



from ema_workbench import (RealParameter, CategoricalParameter, 
                           ScalarOutcome, ema_logging,
                           perform_experiments, Constant)

from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)

    #Set model
    model = ExcelModel("scenarioModel", wd="./models",
                       model_file='trv_scenario_AV.xlsx')
    model.default_sheet = "EMA"

   #%% Specify inputs

   #Set parametric uncetainties
    model.uncertainties = [RealParameter("Heavy truck el share",
                                         .02,0.5
                                         ,variable_name="C6")
                          ,RealParameter("Truck demand change",
                                         -0.2,0.20,
                                         variable_name="C4")
                          ,RealParameter("Car el share",
                                         0.18,0.7,
                                         variable_name="C5")
                          ,RealParameter("Car demand change",
                                         -0.2,0.2
                                         ,variable_name="C3")
                          ,RealParameter("Fossile price adjustment",
                                         0.8,1.5,
                                         variable_name="C7")
                          ,RealParameter("Biofuel price adjustment",
                                         0.8,2.5,
                                         variable_name="C8")
                          ,RealParameter("Electricity price adjustment",
                                         .5,2
                                         ,variable_name="C9")
                          # ,RealParameter("Truck demand elasticity",
                          #                -.5,-.1
                          #                ,variable_name="D78")
                          # ,RealParameter("Car demand elasticity",
                          #                -.5,-.1
                          #                ,variable_name="D80")
                          ]
    #Select whether if electrification should be treated as an external uncertainty (True)
    External_electrification_rate=False
    if External_electrification_rate==False:         
        model.constants = [Constant("C10","No")] #Set external electrification parameter in excel to no
        model.uncertainties._data.pop("Car el share") #Remove electrification uncertainties
        model.uncertainties._data.pop("Heavy truck el share")
    else:
        model.constants = [Constant("C10", "Yes")]
    
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
                    # RealParameter("Additional energy efficiency light vehicles",
                    #               0,.05
                    #               ,variable_name="C72"),
                    # RealParameter("Additional energy efficiency trucks",
                    #               0,.05
                    #               ,variable_name="C73"),
                    # RealParameter("Transport efficient society light vehicles",
                    #               0,.25
                    #               ,variable_name="C74"),
                    # RealParameter("Transport efficient society trucks",
                    #               0,.20
                    #               ,variable_name="C75"),
                    ]
    # specification of the outcomes
    model.outcomes = [
                      ScalarOutcome("CO2 TTW change light vehicles", ScalarOutcome.INFO,
                                                      variable_name="C32"),
                      
                      ScalarOutcome("CO2 TTW change trucks",ScalarOutcome.INFO,
                                                      variable_name="C33"),
                      
                      ScalarOutcome("CO2 TTW change total", ScalarOutcome.INFO,
                                    variable_name="C34"),
                      
                      
                      ScalarOutcome("VKT light vehicles",ScalarOutcome.INFO,
                                    variable_name="C35"),
                      
                      ScalarOutcome("VKT trucks",ScalarOutcome.INFO,
                                    variable_name="C36"),
                      
                      ScalarOutcome("VKT total",ScalarOutcome.INFO,
                                    variable_name="C37"),
                      
    
                      ScalarOutcome("Energy bio total", ScalarOutcome.INFO,
                                    variable_name="C38"),
                      
                      ScalarOutcome("Energy fossile total", ScalarOutcome.INFO,
                                     variable_name="C39"),
                      
                      ScalarOutcome("Energy el total",ScalarOutcome.INFO,
                                    variable_name="C40"),
                      
                      ScalarOutcome("Energy total", ScalarOutcome.MINIMIZE,
                                    variable_name="C41"),
    
    
    
                      ScalarOutcome("Electrified VKT share light vehicles",ScalarOutcome.INFO,
                                    variable_name="C42"),
                      ScalarOutcome("Electrified VKT share trucks",ScalarOutcome.INFO,
                                    variable_name="C43"),
                      ScalarOutcome("Electric share of total energy",ScalarOutcome.INFO,
                                    variable_name="C44"),
                      
                      ScalarOutcome("Driving cost light vehicles relative reference",ScalarOutcome.MINIMIZE,
                                    variable_name="C45"),
                      ScalarOutcome("Driving cost trucks relative reference",ScalarOutcome.MINIMIZE,
                                    variable_name="C46"),
                      ScalarOutcome("Fossile fuel price relative reference trucks",ScalarOutcome.INFO,
                                    variable_name="C52"),
                      ScalarOutcome("Fossile fuel price relative reference light vehicles",ScalarOutcome.INFO,
                                    variable_name="C53"),
                      
                      ScalarOutcome("Delta CS light vehicles",ScalarOutcome.INFO,
                                                      variable_name="C47"),
                      ScalarOutcome("Delta CS trucks",ScalarOutcome.INFO,
                                                      variable_name="C48"),
                      ScalarOutcome("Delta CS total", ScalarOutcome.INFO,
                                    variable_name="C49"),                      
                      ScalarOutcome("Delta tax income total",ScalarOutcome.INFO,
                                                      variable_name="C50"),                    
                      ScalarOutcome("Delta CS tax",ScalarOutcome.INFO,
                                                      variable_name="C51")
    
                      ]  
   #%% set reference scenario
    #reference = Scenario('reference', b=0.4, q=2, mean=0.02, stdev=0.01)

    #%% Specify constraints
    from ema_workbench import Constraint
    CO2_target=-.9
    bio_target=15
    # constraints = [Constraint("max CO2", outcome_names="CO2 TTW change total",
    #                           function=lambda x : max(0, x-CO2_target)),
    #                 Constraint("max bio", outcome_names="Energy bio total",
    #                                           function=lambda y : max(0, y-bio_target)),
    #                 Constraint("positive CO2",outcome_names="CO2 TTW change total",
    #                            function=lambda x : min(0,x+1))
    #                            ]
    
    # constraints = [Constraint("max CO2", outcome_names="CO2 TTW change total",
    #                           function=lambda x : max(0, x-CO2_target)),
    #                 Constraint("max bio", outcome_names="Energy bio total",
    #                                           function=lambda y : max(0, y-bio_target))]
    
    constraints = [Constraint("max CO2", outcome_names="CO2 TTW change total",
                              function=lambda x : max(0, x-CO2_target))]
    #constraints=[]
#%%
    #Simulation settings
    n_p=3
    
    #Run
    import time
    tic=time.perf_counter()
    from ema_workbench import MultiprocessingEvaluator, ema_logging
    import matplotlib.pyplot as plt
    from ema_workbench.em_framework.optimization import (HyperVolume,
                                                         EpsilonProgress)

    
    ema_logging.log_to_stderr(ema_logging.INFO)
    convergence_metrics = [HyperVolume(minimum=[0]*len(model.outcomes), maximum=[1]*len(model.outcomes)),
                           EpsilonProgress()]
    nfe=10000
    with MultiprocessingEvaluator(msis=model,n_processes=n_p) as evaluator:
        results, convergence = evaluator.optimize(nfe=nfe, searchover='levers',
                                      epsilons=[0.01,]*len(model.outcomes),
                                      convergence=convergence_metrics,
                                      constraints=constraints
                                      )
    # results = []
    # with MultiprocessingEvaluator(msis=model,n_processes=n_p) as evaluator:
    # # we run 5 seperate optimizations
    #     for _ in range(1):
    #         result = evaluator.optimize(
    #             nfe=100, searchover="levers", epsilons=[0.05] * len(model.outcomes),
    #             constraints=constraints,convergence=convergence_metrics,
    #         )
    #         results.append(result)
        

    toc=time.perf_counter()
    print("Runtime [s]= " +str(toc-tic))
    print("Runtime [h]= " +str(round((toc-tic)/3600,1)))
    #%%

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

        convergence_metrics = [HyperVolume(minimum=[0,0,0,0], maximum=[1,1.01,1.01,1.01]),
                       EpsilonProgress()]
        with MultiprocessingEvaluator(model_worstcase) as evaluator:
            results_worstcase, convergence = evaluator.optimize(nfe=50000, searchover='uncertainties',
                                         epsilons=[0.1,]*len(model_worstcase.outcomes),
                                         convergence=convergence_metrics)

#%%
          # Save results?
    save_results=1
    if save_results==1:
        from datetime import date    
        from ema_workbench import save_results
        filename=str(nfe)+'_nfe_directed_search_'+str(date.today())
        filename1=filename+'.p'
        
        import pickle
        pickle.dump([results,convergence],open("./output_data/"+filename1,"wb"))
        filename2=filename+'model_'+".p"
        pickle.dump(model,open("./output_data/"+filename2,"wb"))
