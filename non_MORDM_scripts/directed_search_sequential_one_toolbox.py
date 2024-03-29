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
    nfe=50000 #
   #%% Specify inputs
    model.uncertainties = [
                           RealParameter("X1_car_demand", 
                                           -0.3,0.2
                                           ,variable_name="C3")
                          ,RealParameter("X2_truck_demand", 
                                         -0.3,0.2,
                                         variable_name="C4")
                          ,RealParameter("X3_fossil_fuel_price", 
                                         0.4,1.3,
                                         variable_name="C7")                    
                          ,RealParameter("X4_bio_fuel_price",
                                         0.8,2.2,
                                         variable_name="C8")
                          ,RealParameter("X5_electricity_price",  
                                         .5,1.5
                                         ,variable_name="C9")
                          ,RealParameter("X6_car_electrification_rate", 
                                         0.35,0.9,
                                         variable_name="C5")
                          ,RealParameter("X7_truck_electrification_rate", 
                                         .10,0.60
                                         ,variable_name="C6")
                           ,RealParameter("X8_SAV_market_share",
                                           0,.45
                                           ,variable_name="C15")
                           ,RealParameter("X9_SAV_driving_cost",
                                         -.1,1,
                                         variable_name="Indata!G76")
                           ,RealParameter("X10_SAV_energy_efficiency",
                                         0,0.25,
                                         variable_name="Indata!G75")
                           
                           ,RealParameter("X11_VKT_per_SAV",
                                           .5,2
                                           ,variable_name="C16")

                           ,RealParameter("X12_driverless_truck_market_share",
                                           0,0.60
                                           ,variable_name="C11")
                           ,RealParameter("X13_driverless_truck_driving_costs",
                                           -0.50,-0.20
                                           ,variable_name="C14")
                           ,RealParameter("X14_driverless_truck_energy_efficiency",
                                           -0.2,-0.1
                                           ,variable_name="C12")

                           ,RealParameter("X15_VKT_per_driverless_truck",
                                           0.2,0.5
                                           ,variable_name="C13")

                          ]
    #Select whether if electrification should be treated as an external uncertainty (True)
    External_electrification_rate=True
    if External_electrification_rate==False:         
        model.constants = [Constant("C10","No")] #Set external electrification parameter in excel to no
        model.uncertainties._data.pop("X6_car_electrification_rate") #Remove electrification uncertainties
        model.uncertainties._data.pop("X7_truck_electrification_rate")
    else:
        model.constants = [Constant("C10", "Yes")]

    # specification of the outcomes
    # model.outcomes = [
    #                   ScalarOutcome("CO2 TTW change light vehicles", ScalarOutcome.INFO,
    #                                                   variable_name="C32"),
                      
    #                   ScalarOutcome("CO2 TTW change trucks",ScalarOutcome.INFO, 
    #                                                   variable_name="C33"),
                      
    #                   ScalarOutcome("CO2 TTW change total", ScalarOutcome.MINIMIZE,
    #                                 variable_name="C34"),
                      
                      
    #                   ScalarOutcome("VKT light vehicles",ScalarOutcome.INFO,
    #                                 variable_name="C35"),
                      
    #                   ScalarOutcome("VKT trucks",ScalarOutcome.INFO,
    #                                 variable_name="C36"),
                      
    #                   ScalarOutcome("VKT total",ScalarOutcome.INFO,
    #                                 variable_name="C37"),
                      
    
    #                   ScalarOutcome("Energy bio total", ScalarOutcome.MINIMIZE, 
    #                                 variable_name="C38"),
                      
    #                   ScalarOutcome("Energy fossile total", ScalarOutcome.INFO, 
    #                                   variable_name="C39"),
                      
    #                   ScalarOutcome("Energy el total",ScalarOutcome.MINIMIZE,
    #                                 variable_name="C40"),#min
                      
    #                   ScalarOutcome("Energy total", ScalarOutcome.INFO, 
    #                                 variable_name="C41"),
    
       
    #                   ScalarOutcome("Electrified VKT share light vehicles",ScalarOutcome.INFO, 
    #                                 variable_name="C42"),
    #                   ScalarOutcome("Electrified VKT share trucks",ScalarOutcome.INFO, 
    #                                 variable_name="C43"),
    #                   ScalarOutcome("Electric share of total energy",ScalarOutcome.INFO, 
    #                                 variable_name="C44"),
                      
    #                   ScalarOutcome("Driving cost light vehicles",ScalarOutcome.MINIMIZE, 
    #                                 variable_name="C54"),
    #                   ScalarOutcome("Driving cost trucks",ScalarOutcome.MINIMIZE, 
    #                                 variable_name="C55"), 
    #                   ScalarOutcome("Fossile fuel price relative reference trucks",ScalarOutcome.INFO, 
    #                                 variable_name="C52"),
    #                   ScalarOutcome("Fossile fuel price relative reference light vehicles",ScalarOutcome.INFO, 
    #                                 variable_name="C53"),
                      
    #                   ScalarOutcome("Delta CS light vehicles",ScalarOutcome.INFO, 
    #                                                   variable_name="C47"),
    #                   ScalarOutcome("Delta CS trucks",ScalarOutcome.INFO, 
    #                                                   variable_name="C48"),#
    #                   ScalarOutcome("Delta CS total", ScalarOutcome.INFO, #
    #                                 variable_name="C49"),                      
    #                   ScalarOutcome("Delta tax income total",ScalarOutcome.INFO, 
    #                                                   variable_name="C50"),                   
    #                   ScalarOutcome("Delta CS tax",ScalarOutcome.INFO, 
    #                                                   variable_name="C51")
    #                   ]
    
    model.outcomes = [
                      ScalarOutcome("M1_CO2_TTW_change_total", ScalarOutcome.MINIMIZE,
                                    variable_name="C34"),

                      ScalarOutcome("M2_driving_cost_car",ScalarOutcome.MINIMIZE, 
                                    variable_name="C54"),
                      
                      ScalarOutcome("M3_driving_cost_truck",ScalarOutcome.MINIMIZE, 
                                    variable_name="C55"), 
                      
                      ScalarOutcome("M4_energy_use_bio", ScalarOutcome.MINIMIZE, 
                                    variable_name="C38"),

                      ScalarOutcome("M5_energy_use_electricity",ScalarOutcome.MINIMIZE,
                                    variable_name="C40"),#min
                      ]
   #%% Create scenarios to sample over
    from ema_workbench import Scenario
    import pandas as pd
    if sample_new_scenarios:
        def Extract(lst,i):
            return [item[i] for item in lst]
        
        #Set sampling paramters
        n_uncertainties=len(model.uncertainties.keys())
        sampler=samplers.LHSSampler()
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
    worst_case = {
        "X1_car_demand": model.uncertainties["X1_car_demand"].upper_bound,
        "X2_truck_demand": model.uncertainties["X2_truck_demand"].upper_bound,
        "X3_fossil_fuel_price": model.uncertainties["X3_fossil_fuel_price"].lower_bound,
        "X4_bio_fuel_price": model.uncertainties["X4_bio_fuel_price"].lower_bound,
        "X5_electricity_price": model.uncertainties["X5_electricity_price"].lower_bound,
        "X6_car_electrification_rate": model.uncertainties["X6_car_electrification_rate"].lower_bound,
        "X7_truck_electrification_rate": model.uncertainties["X7_truck_electrification_rate"].lower_bound,
        "X8_SAV_market_share": model.uncertainties["X8_SAV_market_share"].upper_bound,
        "X9_SAV_driving_cost": model.uncertainties["X9_SAV_driving_cost"].lower_bound,
        "X10_SAV_energy_efficiency": model.uncertainties["X10_SAV_energy_efficiency"].lower_bound,
        "X11_VKT_per_SAV": model.uncertainties["X11_VKT_per_SAV"].upper_bound,
        "X12_driverless_truck_market_share": model.uncertainties["X12_driverless_truck_market_share"].upper_bound,
        "X13_driverless_truck_driving_costs": model.uncertainties["X13_driverless_truck_driving_costs"].lower_bound,
        "X14_driverless_truck_energy_efficiency": model.uncertainties["X14_driverless_truck_energy_efficiency"].upper_bound,
        "X15_VKT_per_driverless_truck": model.uncertainties["X15_VKT_per_driverless_truck"].upper_bound
    }

    
    
    best_case = {
        "X1_car_demand": model.uncertainties["X1_car_demand"].lower_bound,
        "X2_truck_demand": model.uncertainties["X2_truck_demand"].lower_bound,
        "X3_fossil_fuel_price": model.uncertainties["X3_fossil_fuel_price"].upper_bound,
        "X4_bio_fuel_price": model.uncertainties["X4_bio_fuel_price"].upper_bound,
        "X5_electricity_price": model.uncertainties["X5_electricity_price"].upper_bound,
        "X6_car_electrification_rate": model.uncertainties["X6_car_electrification_rate"].upper_bound,
        "X7_truck_electrification_rate": model.uncertainties["X7_truck_electrification_rate"].upper_bound,
        "X8_SAV_market_share": model.uncertainties["X8_SAV_market_share"].upper_bound,
        "X9_SAV_driving_cost": model.uncertainties["X9_SAV_driving_cost"].upper_bound,
        "X10_SAV_energy_efficiency": model.uncertainties["X10_SAV_energy_efficiency"].upper_bound,
        "X11_VKT_per_SAV": model.uncertainties["X11_VKT_per_SAV"].lower_bound,
        "X12_driverless_truck_market_share": model.uncertainties["X12_driverless_truck_market_share"].upper_bound,
        "X13_driverless_truck_driving_costs": model.uncertainties["X13_driverless_truck_driving_costs"].upper_bound,
        "X14_driverless_truck_energy_efficiency": model.uncertainties["X14_driverless_truck_energy_efficiency"].lower_bound,
        "X15_VKT_per_driverless_truck": model.uncertainties["X15_VKT_per_driverless_truck"].upper_bound
    }

    mid_case = {
        "X1_car_demand": 0,
        "X2_truck_demand": 0,
        "X3_fossil_fuel_price": 1,
        "X4_bio_fuel_price": 1,
        "X5_electricity_price": 1,
        "X6_car_electrification_rate": .38,
        "X7_truck_electrification_rate": .1,
        "X8_SAV_market_share": 0,
        "X9_SAV_driving_cost": 0,
        "X10_SAV_energy_efficiency": 0,
        "X11_VKT_per_SAV": 0,
        "X12_driverless_truck_market_share": 0,
        "X13_driverless_truck_driving_costs": 0,
        "X14_driverless_truck_energy_efficiency": 0,
        "X15_VKT_per_driverless_truck": 0
    }


    scenario_list.append(best_case)
    scenario_list.append(worst_case)
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
    from ema_workbench.em_framework.optimization import (HypervolumeMetric,
                                                        EpsilonProgress,
                                                        ArchiveLogger,epsilon_nondominated)
    ema_logging.log_to_stderr(ema_logging.INFO)

    scenario_count=0
    results_list=[]
    convergence_list=[]
    print("Estimated total runs: ",nfe*len(df_scenarios))
    scenario_count=0
    model.levers = [

                    RealParameter("L1_HVO_share_diesel",
                                  0, 0.80,
                                  variable_name="C64"),
                    RealParameter("L2_FAME_share_diesel",
                                  0, 0.07,
                                  variable_name="C65"),
                    RealParameter("L3_HVO_share_gasoline", 
                                  0, 0.7,
                                  variable_name="C66"),
                    RealParameter("L4_ethanol_share_gasoline", 
                                  0, 0.1,
                                  variable_name="C67"),
                    RealParameter("L5_additional_car_energy_efficiency",
                                  0,.05
                                  ,variable_name="C72"),
                    RealParameter("L6_additional_truck_energy_efficiency",
                                  0,.05
                                  ,variable_name="C73"),
                    CategoricalParameter("L7_bus_energy_efficiency",
                                         ["Beslutad politik","Level 1","Level 2"]
                                         ,variable_name="C63"),
                    RealParameter("L8_fuel_tax_increase_gasoline",
                                  0,.12
                                  ,variable_name="C70"),
                    RealParameter("L9_fuel_tax_increase_diesel",
                                  0,.12
                                  ,variable_name="C71"),
                    RealParameter("L10_km_tax_cars",
                                  0,2
                                  ,variable_name="C68"),
                    RealParameter("L11_km_tax_trucks",
                                  0,3
                                  ,variable_name="C69"),

                    RealParameter("L12_transport_efficient_planning_cars",
                                  0,.25
                                  ,variable_name="C74"),
                    RealParameter("L13_transport_efficient_planning_trucks",
                                  0,.25
                                  ,variable_name="C75"),
            ]
    for scenario in scenario_list:
        print("Scenario: ",scenario_count)
        reference = Scenario()
        reference.data=scenario
        convergence_metrics = [
                               EpsilonProgress()]
        convergence_metrics = [
            ArchiveLogger(
                "./archives",
                [l.name for l in model.levers],
                [o.name for o in model.outcomes],
                base_filename=f"{scenario_count}.tar.gz",
            ),
            EpsilonProgress(),
        ]
        epsilons=[0.01]*5
        with MultiprocessingEvaluator(msis=model,n_processes=n_p) as evaluator:
            results, convergence = evaluator.optimize(nfe=nfe,
                                          searchover='levers',
                                          epsilons=epsilons,
                                          convergence=convergence_metrics,
                                          constraints=constraints,
                                          reference=reference
                                          )
        results_list.append(results)
        convergence_list.append(convergence)
        scenario_count=scenario_count+1
        
    # Save results?
        save_results=1
        if save_results==1:
            from datetime import date    
            from ema_workbench import save_results
            filename=str(nfe)+'_nfe_directed_search_sequential_'+str(date.today())+'_'+str(n_scenarios)+'_scenarios'
            filename1=filename+'.p'
            import pickle
            pickle.dump([results_list,convergence_list,df_scenarios],open("./output_data/"+filename1,"wb"))
            filename2=filename+'model_'+".p"
            pickle.dump(model,open("./output_data/"+filename2,"wb"))
       #%% Compute and visualize convergence metrics
    all_archives = []
    
    for i in range(scenario_count):
        archives = ArchiveLogger.load_archives(f"./archives/{i}.tar.gz")
        all_archives.append(archives)    

    from ema_workbench import (
        HypervolumeMetric,
        GenerationalDistanceMetric,
        EpsilonIndicatorMetric,
        InvertedGenerationalDistanceMetric,
        SpacingMetric,
    )
    from ema_workbench.em_framework.optimization import to_problem
    
    problem = to_problem(model, searchover="levers")
    


# Create a new DataFrame from the modified rows
    results_epsilon=[results]
    reference_set = epsilon_nondominated((results_epsilon), epsilons, problem)

    # hv = HypervolumeMetric(reference_set, problem)
    gd = GenerationalDistanceMetric(reference_set, problem, d=1)
    ei = EpsilonIndicatorMetric(reference_set, problem)
    ig = InvertedGenerationalDistanceMetric(reference_set, problem, d=1)
    sm = SpacingMetric(problem)
    
    
    metrics_by_seed = []
    for archives in all_archives:
        metrics = []
        for nfe, archive in archives.items():
            scores = {
                "generational_distance": gd.calculate(archive),
                # "hypervolume": hv.calculate(archive),
                "epsilon_indicator": ei.calculate(archive),
                "inverted_gd": ig.calculate(archive),
                "spacing": sm.calculate(archive),
                "nfe": int(nfe),
            }
            metrics.append(scores)
        metrics = pd.DataFrame.from_dict(metrics)
    
        # sort metrics by number of function evaluations
        metrics.sort_values(by="nfe", inplace=True)
        metrics_by_seed.append(metrics)
    import seaborn as sns
    sns.set_style("white")
    fig, axes = plt.subplots(nrows=6, figsize=(8, 12), sharex=True)
    
    ax1, ax2, ax3, ax4, ax5, ax6 = axes
    
    for metrics, convergence in zip(metrics_by_seed, convergence_list):
        # ax1.plot(metrics.nfe, metrics.hypervolume)
        # ax1.set_ylabel("hypervolume")
    
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
    
    #%% Old visualization
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


