# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:45:55 2021

@author: aengholm
This is a script for setting up and running an EMA analysis using a static 
excel model with parametric uncertainties(designed for the TRV scenario model)
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
    model.default_sheet = "EMA"

    #%% Specify inputs
    #Set parametric uncetainties
    model.uncertainties = [RealParameter("Heavy truck el share",
                                         .02,0.5
                                         ,variable_name="B6")
                          ,RealParameter("Truck demand change",
                                         -0.2,0.20,
                                         variable_name="B4")
                          ,RealParameter("Car el share",
                                         0.18,0.7,
                                         variable_name="B5")
                          ,RealParameter("Car demand change",
                                         -0.2,0.2
                                         ,variable_name="B3")
                          ,RealParameter("Fossile price adjustment",
                                         0.8,1.5,
                                         variable_name="B7")
                          ,RealParameter("Biofuel price adjustment",
                                         0.8,2.5,
                                         variable_name="B8")
                          ,RealParameter("Electricity price adjustment",
                                         .5,2
                                         ,variable_name="B9")
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
                                         ,variable_name="Indata!F17"),
                    CategoricalParameter("Bus energy consumption",
                                         ["Beslutad politik","Level 1","Level 2"]
                                         ,variable_name="Indata!F23"),
                    RealParameter("Share HVO diesel",
                                  0, 0.80,
                                  variable_name="Indata!F12"),
                    RealParameter("Share FAME diesel",
                                  0, 0.07,
                                  variable_name="Indata!F11"),
                    RealParameter("Share HVO gasoline", 
                                  0, 0.7,
                                  variable_name="Indata!F10"),
                    RealParameter("Share ethanol gasoline", 
                                  0, 0.1,
                                  variable_name="Indata!F11"),
                    RealParameter("km-tax light vehicles",
                                  0,2
                                  ,variable_name="Indata!F15"),
                    RealParameter("km-tax trucks",
                                  0,3
                                  ,variable_name="Indata!F16"),
                    RealParameter("Change in fuel tax gasoline",
                                  0,.20
                                  ,variable_name="Indata!F13"),
                    RealParameter("Change in fuel tax diesel",
                                  0,.20
                                  ,variable_name="Indata!F14"),
                    RealParameter("Additional energy efficiency light vehicles",
                                  0,.1
                                  ,variable_name="Indata!F19"),
                    RealParameter("Additional energy efficiency trucks",
                                  0,.1
                                  ,variable_name="Indata!F20"),
                    RealParameter("Transport efficient society light vehicles",
                                  0,.20
                                  ,variable_name="Indata!F21"),
                    RealParameter("Transport efficient society trucks",
                                  0,.15
                                  ,variable_name="Indata!F22"),

                    ]
    # specification of the outcomes
    model.outcomes = [
                      ScalarOutcome("CO2 TTW change light vehicles",
                                                      variable_name="B32"),
                      ScalarOutcome("CO2 TTW change trucks",
                                                      variable_name="B33"),
                      ScalarOutcome("CO2 TTW change total", 
                                    variable_name="B34"),
                      
                      ScalarOutcome("VKT light vehicles",
                                    variable_name="B35"),
                      ScalarOutcome("VKT trucks", 
                                    variable_name="B36"),
                      ScalarOutcome("VKT total",
                                    variable_name="B37"),
                      

                      ScalarOutcome("Energy bio total", 
                                    variable_name="B38"),
                      ScalarOutcome("Energy fossile total", 
                                     variable_name="B39"),
                      ScalarOutcome("Energy el total",
                                    variable_name="B40"),
                      ScalarOutcome("Energy total", 
                                    variable_name="B41"),



                      ScalarOutcome("Electrified VKT share light vehicles",
                                    variable_name="B42"),
                      ScalarOutcome("Electrified VKT share trucks",
                                    variable_name="B43"),
                      ScalarOutcome("Electric share of total energy",
                                    variable_name="B44"),
                      
                      ScalarOutcome("Driving cost light vehicles relative reference",
                                    variable_name="B45"),
                      ScalarOutcome("Driving cost trucks relative reference",
                                    variable_name="B46"),
                      
                      ScalarOutcome("Delta CS light vehicles",
                                                      variable_name="B47"),
                      ScalarOutcome("Delta CS trucks",
                                                      variable_name="B48"),
                      ScalarOutcome("Delta CS total", 
                                    variable_name="B49"),
                      
                      ScalarOutcome("Delta tax income total",
                                                      variable_name="B50"),
                      
                      ScalarOutcome("Delta CS tax",
                                                      variable_name="B51"),

                      ]  
    
    #%% Specify policies
    from ema_workbench.em_framework import samplers
    manual_policies=True #Use the pre-specified 9 policies?
    n_policies=200
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
                 .490,     #Share HVO diesel [%]
                 .490,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 0,     #Transport efficient society light vehicles [% reduction]
                 0,     #Transport efficient society trucks [% reduction]
                 0,     #km-tax light vehicles [SEK/km]
                 0)     #km-tax trucks [SEK/km]
        policy2=(0,     #Additional energy efficiency light vehicles [%]
                 0,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .15,     #Change in fuel tax diesel [%/y]
                 .15,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .390,     #Share HVO diesel [%]
                 .390,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 0,     #Transport efficient society light vehicles [% reduction]
                 0,     #Transport efficient society trucks [% reduction]
                 0,     #km-tax light vehicles [SEK/km]
                 0)     #km-tax trucks [SEK/km]
        policy3=(0,     #Additional energy efficiency light vehicles [%]
                 0,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .08,     #Change in fuel tax diesel [%/y]
                 .08,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .390,     #Share HVO diesel [%]
                 .390,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 0,     #Transport efficient society light vehicles [% reduction]
                 0,     #Transport efficient society trucks [% reduction]
                 1,     #km-tax light vehicles [SEK/km]
                 2)     #km-tax trucks [SEK/km]
        policy4=(0,     #Additional energy efficiency light vehicles [%]
                 0,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .16,     #Change in fuel tax diesel [%/y]
                 .16,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .275,     #Share HVO diesel [%]
                 .275,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 0,     #Transport efficient society light vehicles [% reduction]
                 0,     #Transport efficient society trucks [% reduction]
                 1,     #km-tax light vehicles [SEK/km]
                 2)     #km-tax trucks [SEK/km]
        policy5=(0,     #Additional energy efficiency light vehicles [%]
                 0,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .204,     #Change in fuel tax diesel [%/y]
                 .204,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .275,     #Share HVO diesel [%]
                 .275,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 0,     #Transport efficient society light vehicles [% reduction]
                 0,     #Transport efficient society trucks [% reduction]
                 0,     #km-tax light vehicles [SEK/km]
                 0)     #km-tax trucks [SEK/km]
        policy6=(0.05,     #Additional energy efficiency light vehicles [%]
                 0.05,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .04,     #Change in fuel tax diesel [%/y]
                 .04,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .386,     #Share HVO diesel [%]
                 .386,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 .10,     #Transport efficient society light vehicles [%]
                 .050,     #Transport efficient society trucks [%]
                 .50,     #km-tax light vehicles [SEK/km]
                 1)     #km-tax trucks [SEK/km]
        policy7=(.05,     #Additional energy efficiency light vehicles [%]
                 .05,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .15,     #Change in fuel tax diesel [%/y]
                 .15,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .270,     #Share HVO diesel [%]
                 .270,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 .10,     #Transport efficient society light vehicles [% reduction]
                 .050,     #Transport efficient society trucks [% reduction]
                 .50,     #km-tax light vehicles [SEK/km]
                 1)     #km-tax trucks [SEK/km]
        policy8=(.05,     #Additional energy efficiency light vehicles [%]
                 .05,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .125,     #Change in fuel tax diesel [%/y]
                 .125,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .2550,     #Share HVO diesel [%]
                 .2550,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 .18,     #Transport efficient society light vehicles [% reduction]
                 .120,     #Transport efficient society trucks [% reduction]
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
    nr_scenarios=5000  #select number of scenarios (per policy)
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
