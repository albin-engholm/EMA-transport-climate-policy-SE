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

from ema_workbench.em_framework import samplers
from SALib.analyze import sobol
from ema_workbench import (Scenario, Policy)
from ema_workbench import (
    ema_logging,
    perform_experiments,
    Samplers)
from ema_workbench.em_framework import samplers
from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
from ema_workbench.em_framework import evaluators
if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)

# Load model
    import pickle
    model = pickle.load(
        open("./output_data/robustness_analysis_results/2100_scenarios_MORDM_OE_2024-01-05model_.p", "rb"))

# Load candidate policies
    # Load candidate policy dataframe
    date = "2023-12-28"
    nfe = 1000000
    filename = f"{date}_{nfe}candidate_policies.p"
    candidate_policy_data = pickle.load(open("./output_data/candidate_policies/"+filename, "rb"))
# %% Specify whether to search over levers or uncertainties
    search_over = "uncertainties"

# %% Specify the reference policy / scenario

    if search_over == "uncertainties":
        policy_name = "B"  # Policy to use as reference
        reference_policy = candidate_policy_data.loc["B"]

        # Create the reference scenario
        reference = Policy("Reference", **reference_policy.to_dict())

    elif search_over == "levers":
        reference_scenario = {
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
            "R1_fuel_price_to_car_electrification": 0.19,
            "R2_fuel_price_to_truck_electrification": 0,
            "R3_fuel_price_to_car_fuel_consumption": -0.05,
            "R4_car_driving_cost_to_car_ownership": -0.1,
            "R5_car_driving_cost_to_car_VKT": -0.2,
            "R6_truck_driving_cost_to_truck_VKT": -1.14
        }
       # Create the reference scenario

        reference = Scenario("Reference", **reference_scenario)
# %% Speicify and run SOBOL sampling
    n_sobol = 1
    # The number of experiments = N*(p*2+2) where N is n_samples, p is number of decision variables
    n_samples = int(1024*2)  # select number of scenarios (per policy)
    n_p = 8
    # Run
    import time
    tic = time.perf_counter()

    if search_over == "uncertainties":
        with MultiprocessingEvaluator(msis=model, n_processes=n_p) as evaluator:
            experiments, outcomes = perform_experiments(model,
                                                        evaluator=evaluator,
                                                        scenarios=n_samples,
                                                        policies=reference,
                                                        uncertainty_sampling=Samplers.SOBOL)
    elif search_over == "levers":
        with MultiprocessingEvaluator(msis=model, n_processes=n_p) as evaluator:
            experiments, outcomes = perform_experiments(model,
                                                        evaluator=evaluator,
                                                        policies=n_samples,
                                                        scenarios=reference,
                                                        lever_sampling=Samplers.SOBOL)
    toc = time.perf_counter()
    print("Runtime [s]= " + str(toc-tic))
    print("Runtime [h]= " + str(round((toc-tic)/3600, 1)))
    print("Runtime per experiment [s]= " + str((toc-tic)/len(experiments)))

# %% Save data
    # Save results?
    save_results = 1
    if save_results == 1:
        from datetime import date
        from ema_workbench import save_results
        filename = str(n_samples)+"_SOBOL_"+str(search_over)+"_"+str(date.today())
        filename1 = filename+'.tar.gz'
        pickle.dump([experiments, outcomes], open("./output_data/sobol_results/"+filename+".p", "wb"))
        pickle.dump(model, open("./output_data/sobol_results/"+filename+"model_"+".p", "wb"))
        #save_results([experiments,outcomes], "./output_data/"+filename1)
