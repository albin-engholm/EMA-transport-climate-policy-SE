# -*- coding: utf-8 -*-
"""
Created on Thu Jun 1 12:45:55 2022

@author: aengholm
This is a script for setting up and running an EMA analysis using a static 
excel model with parametric uncertainties(designed for the TRV scenario model)
Results are saved and can be loaded for analysis in separate script - 
e.g. scenario_exploration_excel_static.py'

This is created specifically to run for _2040_
""" 

from ema_workbench import (RealParameter, CategoricalParameter, 
                           ScalarOutcome, ema_logging,
                           perform_experiments, Constant,
                           BooleanParameter,IntegerParameter)

from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)

    #Set model
    model = ExcelModel("scenarioModel", wd="./models",
                       model_file='trv_scenario2.xlsx')
    model.default_sheet = "EMA"

#%% read inputs from Excel sheet
    import pandas as pd
    
    df_inputs = pd.read_excel (r'C:\Users\aengholm\MUST code repository\climate_scenarios_EMA\input_data\input_test.xlsx')
    df_inputs=df_inputs[df_inputs["Use"]==1] #Remove unused inputs
    u=[]
    for index,row in df_inputs[df_inputs["Input type"]=="U"].iterrows(): #Extract uncerrtainties
        if row["Use"]==1:
            if row["Parameter type"]=="RealParameter":
                u.append(RealParameter(row["Name"],row["LB"],row["UB"],variable_name=row["Cell"]))
            if row["Parameter type"] == "CategoricalParameter":
                u.append(CategoricalParameter(row["Name"],row["LB"],row["UB"],variable_name=row["Cell"]))
            if row["Parameter type"] == "IntegerParameter":
                u.append(IntegerParameter(row["Name"],int(row["LB"]),int(row["UB"]),variable_name=row["Cell"]))
                
    # for index,row in df_inputs[df_inputs["Input type"]=="L"].iterrows(): #Extract Levers
    #     if row["Use"]==1:
    #         if row["Parameter type"]=="RealParameter":
    #             u.append(RealParameter(row["Name"],row["LB"],row["UB"],variable_name=row["Cell"]))
    #         if row["Parameter type"] == "CategoricalParameter":
    #             u.append(CategoricalParameter(row["Name"],row["LB"],row["UB"],variable_name=row["Cell"]))
    #         if row["Parameter type"] == "IntegerParameter":
    #             u.append(IntegerParameter(row["Name"],int(row["LB"]),int(row["UB"]),variable_name=row["Cell"]))
        
        
   # print (df_uncertainties)
    model.uncertainties=u
    #%% Specify inputs

    #Set parametric uncetainties
    # model.uncertainties = [RealParameter("Heavy truck el share",
    #                                      .02,0.5
    #                                      ,variable_name="C6")
    #                       ,RealParameter("Truck demand change",
    #                                      -0.2,0.20,
    #                                      variable_name="C4")
    #                       ,RealParameter("Car el share",
    #                                      0.18,0.7,
    #                                      variable_name="C5")
    #                       ,RealParameter("Car demand change",
    #                                      -0.2,0.2
    #                                      ,variable_name="C3")
    #                       ,RealParameter("Fossile price adjustment",
    #                                      0.8,1.5,
    #                                      variable_name="C7")
    #                       ,RealParameter("Biofuel price adjustment",
    #                                      0.8,2.5,
    #                                      variable_name="C8")
    #                       ,RealParameter("Electricity price adjustment",
    #                                      .5,2
    #                                      ,variable_name="C9")
    #                       # ,RealParameter("Truck demand elasticity",
    #                       #                -.5,-.1
    #                       #                ,variable_name="D78")
    #                       # ,RealParameter("Car demand elasticity",
    #                       #                -.5,-.1
    #                       #                ,variable_name="D80")
    #                       ]
    #Select whether if electrification should be treated as an extenral uncertainty (True)
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
    # specification of the outcomes
    model.outcomes = [
                      ScalarOutcome("CO2 TTW change light vehicles",
                                                      variable_name="C32"),
                      ScalarOutcome("CO2 TTW change trucks",
                                                      variable_name="C33"),
                      ScalarOutcome("CO2 TTW change total", 
                                    variable_name="C34"),
                      
                      ScalarOutcome("VKT light vehicles",
                                    variable_name="C35"),
                      ScalarOutcome("VKT trucks", 
                                    variable_name="C36"),
                      ScalarOutcome("VKT total",
                                    variable_name="C37"),
                      

                      ScalarOutcome("Energy bio total", 
                                    variable_name="C38"),
                      ScalarOutcome("Energy fossile total", 
                                     variable_name="C39"),
                      ScalarOutcome("Energy el total",
                                    variable_name="C40"),
                      ScalarOutcome("Energy total", 
                                    variable_name="C41"),



                      ScalarOutcome("Electrified VKT share light vehicles",
                                    variable_name="C42"),
                      ScalarOutcome("Electrified VKT share trucks",
                                    variable_name="C43"),
                      ScalarOutcome("Electric share of total energy",
                                    variable_name="C44"),
                      
                      ScalarOutcome("Driving cost light vehicles relative reference",
                                    variable_name="C45"),
                      ScalarOutcome("Driving cost trucks relative reference",
                                    variable_name="C46"),
                      ScalarOutcome("Fossile fuel price relateive reference trucks",
                                    variable_name="C52"),
                      ScalarOutcome("Fossile fuel price relateive reference light vehicles",
                                    variable_name="C53"),
                      
                      ScalarOutcome("Delta CS light vehicles",
                                                      variable_name="C47"),
                      ScalarOutcome("Delta CS trucks",
                                                      variable_name="C48"),
                      ScalarOutcome("Delta CS total", 
                                    variable_name="C49"),                      
                      ScalarOutcome("Delta tax income total",
                                                      variable_name="C50"),                    
                      ScalarOutcome("Delta CS tax",
                                                      variable_name="C51"),

                      ]  
    
    #%% Specify policies
    from ema_workbench.em_framework import samplers
    n_policies=20
    manual_policies=True #Use the pre-specified 9 policies
    if manual_policies:
        n_policies=9
    policies=samplers.sample_levers(model, n_policies, sampler=samplers.LHSSampler())
    
    #%% manual specification of policies   
    if manual_policies: # overide the pre-sampled policies
        policy1=(0,     #Additional energy efficiency light vehicles [%]
                 0,  #Additional energy efficiency trucks [%]
                 1,  #Bus energy consumption [0,1,2]
                 .02,     #Change in fuel tax diesel [%/y]
                 .02,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .628,     #Share HVO diesel [%]
                 .628,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 0,     #Transport efficient society light vehicles [% reduction]
                 0,     #Transport efficient society trucks [% reduction]
                 0,     #km-tax light vehicles [SEK/km]
                 0)     #km-tax trucks [SEK/km]
        policy2=(0,     #Additional energy efficiency light vehicles [%]
                 0,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .09,     #Change in fuel tax diesel [%/y]
                 .09,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .52,     #Share HVO diesel [%]
                 .52,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 0,     #Transport efficient society light vehicles [% reduction]
                 0,     #Transport efficient society trucks [% reduction]
                 0,     #km-tax light vehicles [SEK/km]
                 0)     #km-tax trucks [SEK/km]
        policy3=(0,     #Additional energy efficiency light vehicles [%]
                 0,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .05,     #Change in fuel tax diesel [%/y]
                 .05,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .54,     #Share HVO diesel [%]
                 .54,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 0,     #Transport efficient society light vehicles [% reduction]
                 0,     #Transport efficient society trucks [% reduction]
                 1,     #km-tax light vehicles [SEK/km]
                 2)     #km-tax trucks [SEK/km]
        policy4=(0,     #Additional energy efficiency light vehicles [%]
                 0,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .097,     #Change in fuel tax diesel [%/y]
                 .097,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .41,     #Share HVO diesel [%]
                 .41,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 0,     #Transport efficient society light vehicles [% reduction]
                 0,     #Transport efficient society trucks [% reduction]
                 1,     #km-tax light vehicles [SEK/km]
                 2)     #km-tax trucks [SEK/km]
        policy5=(0,     #Additional energy efficiency light vehicles [%]
                 0,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .12,     #Change in fuel tax diesel [%/y]
                 .12,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .34,     #Share HVO diesel [%]
                 .34,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 0,     #Transport efficient society light vehicles [% reduction]
                 0,     #Transport efficient society trucks [% reduction]
                 0,     #km-tax light vehicles [SEK/km]
                 0)     #km-tax trucks [SEK/km]
        policy6=(0.05,     #Additional energy efficiency light vehicles [%]
                 0.05,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .031,     #Change in fuel tax diesel [%/y]
                 .031,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .58,     #Share HVO diesel [%]
                 .55,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 .10,     #Transport efficient society light vehicles [%]
                 .050,     #Transport efficient society trucks [%]
                 .50,     #km-tax light vehicles [SEK/km]
                 1)     #km-tax trucks [SEK/km]
        policy7=(.05,     #Additional energy efficiency light vehicles [%]
                 .05,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .092,     #Change in fuel tax diesel [%/y]
                 .092,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .425,     #Share HVO diesel [%]
                 .425,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 .10,     #Transport efficient society light vehicles [% reduction]
                 .050,     #Transport efficient society trucks [% reduction]
                 .50,     #km-tax light vehicles [SEK/km]
                 1)     #km-tax trucks [SEK/km]
        policy8=(.05,     #Additional energy efficiency light vehicles [%]
                 .05,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .63,     #Change in fuel tax diesel [%/y]
                 .063,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .438,     #Share HVO diesel [%]
                 .438,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 .26,     #Transport efficient society light vehicles [% reduction]
                 .17,     #Transport efficient society trucks [% reduction]
                 .50,     #km-tax light vehicles [SEK/km]
                 1)     #km-tax trucks [SEK/km]
        policy9=(0,     #Additional energy efficiency light vehicles [%] ### SET VALUES FOR REFERENCE SCENARIO
                 0,  #Additional energy efficiency trucks [%]
                 0,  #Bus energy consumption [0,1,2]
                 .02,     #Change in fuel tax diesel [%/y]
                 .02,      #Change in fuel tax gasoline [%/y]  
                 0,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .25,     #Share HVO diesel [%]
                 0,     #Share HVO gasoline [%]
                 0.075,     #Share ethanol gasoline [%]
                 0,     #Transport efficient society light vehicles [% reduction]
                 0,     #Transport efficient society trucks [% reduction]
                 0,     #km-tax light vehicles [SEK/km]
                 0)     #km-tax trucks [SEK/km]

        all_policies=[policy1,policy2,policy3,policy4,policy5,policy6,policy7,policy8, policy9]
        policies.designs=all_policies
        policy_names=["B  Bio fuels",
                      "C1 High fuel tax, biofuels <20TWh",
                      "C2 Fuel and km-tax, biofuels <20TWh",
                      "C3 High fuel and km-tax, biofuels <13TWh",
                      "C4 High fuel tax, biofuels <13TWh",
                      "D1 Transport efficiency, fuel and km-tax, biofuels <20TWh",
                      "D2 High transport efficiency, high fuel and km-tax, biofuels <13TWh",
                      "D3 High transport efficiency, high fuel and km-tax, biofuels <13TWh",
                      "Reference - planned policies"]
    #%%    #Run model - for open exploration
    #Simulation settings
    nr_scenarios=50  #select number of scenarios (per policy)
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
                                                            policies=policies)            
        else:
            experiments, outcomes = perform_experiments(model, nr_scenarios,
                                                        policies=policies,
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
    #%% rename policies if manual policies are used 
    if manual_policies:
        j=0
        for i in experiments["policy"].unique():
            experiments.loc[experiments['policy'] == i, 'policy'] = policy_names[j]
            j=j+1
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
