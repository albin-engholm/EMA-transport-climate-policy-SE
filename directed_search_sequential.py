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
    nfe=100 #
   #%% Specify inputs
    model.uncertainties = [RealParameter("Heavy truck el share", 
                                         .10,0.55
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
                                           0,.45
                                           ,variable_name="C15")
                           ,RealParameter("SAV VKT multiplier",
                                           .5,2
                                           ,variable_name="C16")
                           ,RealParameter("SAV change in energy use",
                                         0,0.25,
                                         variable_name="Indata!G75")
                           ,RealParameter("SAV change in non-energy cost",
                                         -.1,1,
                                         variable_name="Indata!G76")
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
                      
    
                      ScalarOutcome("Energy bio total", ScalarOutcome.MINIMIZE, 
                                    variable_name="C38"),
                      
                      ScalarOutcome("Energy fossile total", ScalarOutcome.INFO, 
                                     variable_name="C39"),
                      
                      ScalarOutcome("Energy el total",ScalarOutcome.MINIMIZE,
                                    variable_name="C40"),#min
                      
                      ScalarOutcome("Energy total", ScalarOutcome.INFO, 
                                    variable_name="C41"),
    
       
                      ScalarOutcome("Electrified VKT share light vehicles",ScalarOutcome.INFO, 
                                    variable_name="C42"),
                      ScalarOutcome("Electrified VKT share trucks",ScalarOutcome.INFO, 
                                    variable_name="C43"),
                      ScalarOutcome("Electric share of total energy",ScalarOutcome.INFO, 
                                    variable_name="C44"),
                      
                      ScalarOutcome("Driving cost light vehicles",ScalarOutcome.MINIMIZE, 
                                    variable_name="C54"),
                      ScalarOutcome("Driving cost trucks",ScalarOutcome.MINIMIZE, 
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
   #%% Create scenarios to sample over
    from ema_workbench import Scenario
    import pandas as pd
    if sample_new_scenarios:
        def Extract(lst,i):
            return [item[i] for item in lst]
        
        #Set sampling paramters
        n_uncertainties=len(model.uncertainties.keys())
        
        scenarios=samplers.sample_uncertainties(model, n_scenarios, sampler=sampler)
        scenarios_dict=dict.fromkeys(scenarios.params)
        #create a dict with all scenario parameters based on scnearios
        count=0
        for i in scenarios_dict.keys():
            scenarios_dict[str(i)]=Extract(scenarios.designs,count)
            count=count+1
         
        #list of scenario dicts
        scenario_list=[]    
        #create a scenario-dict for a single scenario   
        scenario_dict=dict.fromkeys(scenarios.params)
        for j in range(len(scenarios.designs)):
            scenario_dict=dict.fromkeys(scenarios.params)
            for key in scenario_dict.keys():  
                scenario_dict[str(key)]=scenarios_dict[str(key)][j]
            scenario_list.append(scenario_dict)
        df_scenarios=pd.DataFrame.from_dict(scenarios_dict)
    elif load_diverse_scenarios:
        import pickle
        df_scenarios=pickle.load( open("./output_data/diverse_scenarios_3.p", "rb" ) )
        scenario_list=[]    
        for i,row in df_scenarios.iterrows():
            scenario_list.append(row.to_dict())
        
    #%% Definition of scenarios
    if n_scenarios==0:
        scenario_list=[]
    worst_case={
        "AV penetration trucks": model.uncertainties["AV penetration trucks"].upper_bound,
        "AV truck change in energy use": model.uncertainties["AV truck change in energy use"].upper_bound,
        "AV truck change in non-energy cost": model.uncertainties["AV truck change in non-energy cost"].lower_bound,
        "AV truck change in utilization": model.uncertainties["AV truck change in utilization"].upper_bound,
        "Biofuel price adjustment":model.uncertainties["Biofuel price adjustment"].upper_bound,
        "Car demand change":model.uncertainties["Car demand change"].upper_bound,
        "Car el share":model.uncertainties["Car el share"].lower_bound,
        "Electricity price adjustment":model.uncertainties["Electricity price adjustment"].upper_bound,
        "Fossile price adjustment":model.uncertainties["Fossile price adjustment"].lower_bound,
        "Heavy truck el share": model.uncertainties["Heavy truck el share"].lower_bound,
        "SAV change in energy use": model.uncertainties["SAV change in energy use"].lower_bound,
        "SAV change in non-energy cost": model.uncertainties["SAV change in non-energy cost"].lower_bound,
        "SAV penetration": model.uncertainties["SAV penetration"].upper_bound,
        "SAV VKT multiplier": model.uncertainties["SAV VKT multiplier"].upper_bound,
        "Truck demand change": model.uncertainties["Truck demand change"].upper_bound,
        }   
    best_case={
        "AV penetration trucks": 0,
        "AV truck change in energy use": 0,
        "AV truck change in non-energy cost": 0,
        "AV truck change in utilization": 0,
        "Biofuel price adjustment":model.uncertainties["Biofuel price adjustment"].lower_bound,
        "Car demand change":model.uncertainties["Car demand change"].lower_bound,
        "Car el share":model.uncertainties["Car el share"].upper_bound,
        "Electricity price adjustment":model.uncertainties["Electricity price adjustment"].lower_bound,
        "Fossile price adjustment":model.uncertainties["Fossile price adjustment"].upper_bound,
        "Heavy truck el share": model.uncertainties["Heavy truck el share"].upper_bound,
        "SAV change in energy use": model.uncertainties["SAV change in energy use"].upper_bound,
        "SAV change in non-energy cost": model.uncertainties["SAV change in non-energy cost"].upper_bound,
        "SAV penetration": model.uncertainties["SAV penetration"].lower_bound,
        "SAV VKT multiplier": model.uncertainties["SAV VKT multiplier"].lower_bound,
        "Truck demand change": model.uncertainties["Truck demand change"].lower_bound,
        }  
    mid_case={
        "AV penetration trucks":0,
        "AV truck change in energy use":0,
        "AV truck change in non-energy cost":0,
        "AV truck change in utilization":0,
        "Biofuel price adjustment":1,
        "Car demand change":0,
        "Car el share":.38,
        "Electricity price adjustment":1,
        "Fossile price adjustment":1,
        "Heavy truck el share": .1,
        "SAV change in energy use":0,
        "SAV change in non-energy cost":0,
        "SAV penetration":0,
        "SAV VKT multiplier": 0,
        "Truck demand change": 0
        }
    #scenario_list.append(best_case)
    #scenario_list.append(worst_case)
    scenario_list.append(mid_case)
    df_scenarios=pd.DataFrame(scenario_list)
    n_scenarios=len(scenario_list)
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
    scenario_count=0
    policy_types=["B","C","D"]
    for policy_type in policy_types:
        results_list=[]
        convergence_list=[]
        print("Estimated total runs: ",len(policy_types*nfe*len(df_scenarios)))
        scenario_count=0
        print ("Policy type ",policy_type)
        if policy_type=="B":
    #Specification of levers
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
                                          variable_name="C67")
                    ]            
        if policy_type=="C":
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
                                          ,variable_name="C71")
                    ]
        if policy_type=="D":
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
        for scenario in scenario_list:
            print("Scenario: ",scenario_count)
            reference = Scenario()
            reference.data=scenario
            convergence_metrics = [
                                   EpsilonProgress()]
            with MultiprocessingEvaluator(msis=model,n_processes=n_p) as evaluator:
                results, convergence = evaluator.optimize(nfe=nfe, searchover='levers',
                                              epsilons=[0.01]*5,
                                              convergence=convergence_metrics,
                                              constraints=constraints,reference=reference
                                              )
            results_list.append(results)
            convergence_list.append(convergence)
            scenario_count=scenario_count+1
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
            filename=str(nfe)+'_nfe_directed_search_sequential_'+str(date.today())+'_'+str(n_scenarios)+'_scenarios'
            filename1=policy_type+filename+'.p'
            import pickle
            pickle.dump([results_list,convergence_list,df_scenarios],open("./output_data/"+filename1,"wb"))
            filename2=policy_type+filename+'model_'+".p"
            pickle.dump(model,open("./output_data/"+filename2,"wb"))
