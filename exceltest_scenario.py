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
                          ,RealParameter("Fossile price adjustment",
                                         0.75,1.5,
                                         variable_name="Modell - Drivmedelpriser!G5")
                          ,RealParameter("Biofuel price adjustment",
                                         0.75,1.5,
                                         variable_name="Modell - Drivmedelpriser!G6")
                          ,RealParameter("Electricity price adjustment",
                                         .5,2
                                         ,variable_name="Modell - Drivmedelpriser!G4")
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
                                  0,3
                                  ,variable_name="F13"),
                    RealParameter("Change in fuel tax diesel",
                                  0,3
                                  ,variable_name="F14"),
                    RealParameter("Additional energy efficiency light vehicles",
                                  0,.2
                                  ,variable_name="F19"),
                    RealParameter("Additional energy efficiency trucks",
                                  0,.2
                                  ,variable_name="F20"),
                    RealParameter("Transport efficient society light vehicles",
                                  0,.2
                                  ,variable_name="F21"),
                    RealParameter("Transport efficient society trucks",
                                  0,.2
                                  ,variable_name="F22"),

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
    
    #%% Specify policies
    from ema_workbench.em_framework import samplers
    n_policies=8
    policies=samplers.sample_levers(model, n_policies, sampler=samplers.LHSSampler())   
    #%% manual specification of policies
    # overide the pre-sampled policies
    manual_policies=1
    if manual_policies==1:
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
        policy6=(-0.05,     #Additional energy efficiency light vehicles [%]
                 -0.05,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .04,     #Change in fuel tax diesel [%/y]
                 .04,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .386,     #Share HVO diesel [%]
                 .386,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 -.10,     #Transport efficient society light vehicles [%]
                 -.050,     #Transport efficient society trucks [%]
                 .50,     #km-tax light vehicles [SEK/km]
                 1)     #km-tax trucks [SEK/km]
        policy7=(-.05,     #Additional energy efficiency light vehicles [%]
                 -.05,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .15,     #Change in fuel tax diesel [%/y]
                 .15,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .270,     #Share HVO diesel [%]
                 .270,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 -.10,     #Transport efficient society light vehicles [% reduction]
                 -.050,     #Transport efficient society trucks [% reduction]
                 .50,     #km-tax light vehicles [SEK/km]
                 1)     #km-tax trucks [SEK/km]
        policy8=(-.05,     #Additional energy efficiency light vehicles [%]
                 -.05,  #Additional energy efficiency trucks [%]
                 2,  #Bus energy consumption [0,1,2]
                 .125,     #Change in fuel tax diesel [%/y]
                 .125,      #Change in fuel tax gasoline [%/y]  
                 1,     #ICE CO2 reduction ambition level [0,1]
                 .07,     #Share FAME diesel [%]
                 .2550,     #Share HVO diesel [%]
                 .2550,     #Share HVO gasoline [%]
                 0.1,     #Share ethanol gasoline [%]
                 -.18,     #Transport efficient society light vehicles [% reduction]
                 -.120,     #Transport efficient society trucks [% reduction]
                 .50,     #km-tax light vehicles [SEK/km]
                 1)     #km-tax trucks [SEK/km]

        all_policies=[policy1,policy2,policy3,policy4,policy5,policy6,policy7,policy8]
        policies.designs=all_policies
        policy_names=["B  Bio fuels",
                      "C1 High fuel tax, biofuels <20TWh",
                      "C2 Fuel and km-tax, biofuels <20TWh",
                      "C3 High fuel and km-tax, biofuels <13TWh",
                      "C4 High fuel tax, biofuels <13TWh",
                      "D1 Transport efficiency, fuel and km-tax, biofuels <20TWh",
                      "D2 High transport efficiency, high fuel and km-tax, biofuels <13TWh",
                      "D3 High transport efficiency, high fuel and km-tax, biofuels <13TWh"]
    #%%
    #select number of scenarios (per policy)
    nr_scenarios=500
    
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
    if manual_policies ==1:
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
