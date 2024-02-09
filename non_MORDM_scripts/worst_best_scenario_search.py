# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:52:02 2023

@author: aengholm
"""


from ema_workbench import (RealParameter, CategoricalParameter, 
                           ScalarOutcome, ema_logging,
                           perform_experiments, Constant)

from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
from ema_workbench.em_framework import samplers
if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)

    #Set model
    model = ExcelModel("scenarioModel", wd="./models",
                       model_file='Master.xlsx')
    model.default_sheet = "EMA"
#%% specify simulation parameters
    sample_new_scenarios=False
    load_diverse_scenarios=False
    n_scenarios=0#Numbers of scenarios to generate
    sampler=samplers.FullFactorialSampler()
    n_p=8 #set # of parallel threads
    nfe=74000 #
   #%% Specify inputs
    model.uncertainties = [RealParameter("Heavy truck el share", 
                                         .10,0.60
                                         ,variable_name="C6")
                          ,RealParameter("Truck demand change", 
                                         -0.3,0.2,
                                         variable_name="C4")
                          ,RealParameter("Car el share", 
                                         0.35,0.9,
                                         variable_name="C5")
                          ,RealParameter("Car demand change", 
                                         -0.3,0.2
                                         ,variable_name="C3")
                          ,RealParameter("Fossile price adjustment", 
                                         0.4,1.3,
                                         variable_name="C7")
                          ,RealParameter("Biofuel price adjustment",
                                         0.8,2.2,
                                         variable_name="C8")
                          ,RealParameter("Electricity price adjustment",  
                                         .5,1.5
                                         ,variable_name="C9")
                           ,RealParameter("AV penetration trucks",
                                           0,0.55
                                           ,variable_name="C11")
                           ,RealParameter("AV truck change in energy use",
                                           -.2,-.1
                                           ,variable_name="C12")
                           ,RealParameter("AV truck change in non-energy cost",
                                           -.55,-.2
                                           ,variable_name="C14")
                           ,RealParameter("AV truck change in utilization",
                                           0.2,0.5
                                           ,variable_name="C13")
                           ,RealParameter("SAV penetration",
                                           0,.40
                                           ,variable_name="C15")
                           ,RealParameter("SAV VKT multiplier",
                                           .5,2
                                           ,variable_name="C16")
                           ,RealParameter("SAV change in energy use",
                                         0,0.25,
                                         variable_name="C81")
                           ,RealParameter("SAV change in non-energy cost",
                                         -.1,1,
                                         variable_name="C82")
                          ]
    #Select whether if electrification should be treated as an external uncertainty (True)
    External_electrification_rate=True
    if External_electrification_rate==False:         
        model.constants = [Constant("C10","No")] #Set external electrification parameter in excel to no
        model.uncertainties._data.pop("Car el share") #Remove electrification uncertainties
        model.uncertainties._data.pop("Heavy truck el share")
    else:
        model.constants = [Constant("C10", "Yes")]

    # specification of the outcomes
    model.outcomes = [
                      ScalarOutcome("CO2 TTW change light vehicles", ScalarOutcome.INFO,
                                                      variable_name="C32"),
                      
                      ScalarOutcome("CO2 TTW change trucks",ScalarOutcome.INFO, 
                                                      variable_name="C33"),
                      
                      ScalarOutcome("CO2 TTW change total", ScalarOutcome.MINIMIZE,
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
                                    variable_name="C40"),#min
                      
                      ScalarOutcome("Energy total", ScalarOutcome.INFO, 
                                    variable_name="C41"),
    
                             ScalarOutcome("Electrified VKT share light vehicles",ScalarOutcome.INFO, 
                                    variable_name="C42"),
                      ScalarOutcome("Electrified VKT share trucks",ScalarOutcome.INFO, 
                                    variable_name="C43"),
                      ScalarOutcome("Electric share of total energy",ScalarOutcome.INFO, 
                                    variable_name="C44"),
                      
                      ScalarOutcome("Driving cost light vehicles",ScalarOutcome.INFO, 
                                    variable_name="C54"),
                      ScalarOutcome("Driving cost trucks",ScalarOutcome.INFO, 
                                    variable_name="C55"), 
                      ScalarOutcome("Fossile fuel price relative reference trucks",ScalarOutcome.INFO, 
                                    variable_name="C52"),
                      ScalarOutcome("Fossile fuel price relative reference light vehicles",ScalarOutcome.INFO, 
                                    variable_name="C53"),
                      
                      ScalarOutcome("Delta CS light vehicles",ScalarOutcome.INFO, 
                                                      variable_name="C47"),
                      ScalarOutcome("Delta CS trucks",ScalarOutcome.INFO, 
                                                      variable_name="C48"),#
                      ScalarOutcome("Delta CS total", ScalarOutcome.INFO, #
                                    variable_name="C49"),                      
                      ScalarOutcome("Delta tax income total",ScalarOutcome.INFO, 
                                                      variable_name="C50"),                   
                      ScalarOutcome("Delta CS tax",ScalarOutcome.INFO, 
                                                      variable_name="C51")
                      ]
    model.levers = [
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
    
    #%Create a reference policy
    from ema_workbench import Policy
    reference_policy=Policy()
    reference_policy.data = {
    "Bus energy consumption": "Beslutad politik",       
    "Share HVO diesel": 0.25,
    "Share FAME diesel": 0.07,
    "Share HVO gasoline": 0,     
    "Share ethanol gasoline": 0.075,
    "km-tax light vehicles": 0,
    "km-tax trucks": 0,
    "Change in fuel tax gasoline": 0.02,
    "Change in fuel tax diesel": 0.02,
    "Additional energy efficiency light vehicles": 0.00,
    "Additional energy efficiency trucks": 0.00,
    "Transport efficient society light vehicles": 0,
    "Transport efficient society trucks": 0
    }

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
    constraints = [Constraint("max CO2", outcome_names="CO2 TTW change total",
                              function=lambda x : max(0, x-CO2_target)),
                    Constraint("max bio", outcome_names="Energy bio total",
                                              function=lambda y : max(0, y-bio_target))]
    
    constraints = [Constraint("max CO2", outcome_names="CO2 TTW change total",
                              function=lambda x : max(0, x-CO2_target))]
    constraints=[]
#%%#Simulation settings and RUN
    import time
    tic=time.perf_counter()
    from ema_workbench import MultiprocessingEvaluator, ema_logging
    import matplotlib.pyplot as plt
    from ema_workbench.em_framework.optimization import (HyperVolume,
                                                        EpsilonProgress)
    ema_logging.log_to_stderr(ema_logging.INFO)
    # convergence_metrics = [HyperVolume(minimum=[0,0,0,0,0,0,0,0], maximum=[1,1,1,1,1,1,1,1]),
    #                        EpsilonProgress()]

    convergence_metrics = [
                           EpsilonProgress()]
    with MultiprocessingEvaluator(msis=model,n_processes=n_p) as evaluator:
        results, convergence = evaluator.optimize(nfe=nfe, searchover='uncertainties',
                                      epsilons=[0.01]*1,
                                      convergence=convergence_metrics,
                                      constraints=constraints,reference=reference_policy
                                      )

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, figsize=(8,4))
    ax1.plot(convergence.nfe, convergence.epsilon_progress)
    ax1.set_ylabel('$\epsilon$-progress')
    # ax2.plot(convergence.nfe, convergence.hypervolume)
    # ax2.set_ylabel('hypervolume')
    
    ax1.set_xlabel('number of function evaluations')
    # ax2.set_xlabel('number of function evaluations')

    toc=time.perf_counter()
    print("Runtime [s]= " +str(toc-tic))
    print("Runtime [h]= " +str(round((toc-tic)/3600,1)))
        
    #%% Save results?
    save_results=1
    if save_results==1:
        from datetime import date    
        from ema_workbench import save_results
        filename=str(nfe)+'_nfe_directed_search_worst_best_case_'+str(date.today())
        filename1=filename+'.p'
        import pickle
        pickle.dump([results,convergence],open("./output_data/"+filename1,"wb"))
        filename2=filename+'model_'+".p"
        pickle.dump(model,open("./output_data/"+filename2,"wb"))
