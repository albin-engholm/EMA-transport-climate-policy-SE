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

from ema_workbench import ( 
                           ema_logging,
                           perform_experiments, 
                           Samplers)

from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)
    
#Load model
    import pickle
    model = pickle.load(open("./output_data/All levers100000_nfe_directed_search_MORDM_2023-07-07model_.p", "rb"))

    
#%% Specify the reference scenario
    mid_case = {
        "X1_car_demand": 0,
        "X2_truck_demand": 0,
        "X3_fossil_fuel_price": 1,
        "X4_bio_fuel_price": 1,
        "X5_electricity_price": 1,
        "X6_car_electrification_rate": .68,
        "X7_truck_electrification_rate": .3,
        "X8_SAV_market_share": 0,
        "X9_SAV_driving_cost": 0,
        "X10_SAV_energy_efficiency": 0,
        "X11_VKT_per_SAV": 0,
        "X12_driverless_truck_market_share": 0,
        "X13_driverless_truck_driving_costs": 0,
        "X14_driverless_truck_energy_efficiency": 0,
        "X15_VKT_per_driverless_truck": 0,
        "R1_fuel_price_to_car_electrification" : 0.19,
        "R2_fuel_price_to_truck_electrification" : 0,
        "R3_fuel_price_to_car_fuel_consumption" : -0.05,
        "R4_car_driving_cost_to_car_ownership" : -0.1,
        "R5_car_driving_cost_to_car_VKT" : -0.2,
        "R6_truck_driving_cost_to_truck_VKT" : -0.9
    }

    from ema_workbench import Scenario
    
    # Create the reference scenario and add to scenario list

    reference = Scenario("Reference", **mid_case)
    #%% Run SOBOL sampling
    from SALib.analyze import sobol
    from ema_workbench.em_framework import samplers
    #Simulation settings
    n_sobol=1
    n_samples=1024*2 #select number of scenarios (per policy)
    n_p=8
    # The number of experiments = N*(p*2+2) where N is n_samples, p is len(levers)
    #Run
    import time
    tic=time.perf_counter()
    with MultiprocessingEvaluator(msis=model,n_processes=n_p) as evaluator:
        experiments, outcomes = perform_experiments(model,  
                                                    evaluator=evaluator,
                                                    policies=n_samples,
                                                    scenarios=reference,
                                                    lever_sampling=Samplers.SOBOL)            

            
    toc=time.perf_counter()
    print("Runtime [s]= " +str(toc-tic))
    print("Runtime [h]= " +str(round((toc-tic)/3600,1)))
    print("Runtime per experiment [s]= " +str((toc-tic)/len(experiments)))

#%% Save data
         # Save results?
    save_results=1
    if save_results==1:
        from datetime import date    
        from ema_workbench import save_results
        filename=str(n_samples)+"_SOBOL_"+str(date.today())
        filename1=filename+'.tar.gz'
        pickle.dump([experiments,outcomes],open("./output_data/"+filename+".p","wb"))
        pickle.dump(model,open("./output_data/"+filename+"model_"+".p","wb"))
        #save_results([experiments,outcomes], "./output_data/"+filename1)


