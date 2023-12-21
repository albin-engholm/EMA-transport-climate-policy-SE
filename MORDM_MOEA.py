# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:11:08 2022

@author: aengholm
This is a script for setting up and performing a MOEA for the TRV Scenario tool. This is the first step of MORDM.
Results are saved and can be used for further analysis in separate scripts. 
"""
# %% Imports
from ema_workbench import (RealParameter, CategoricalParameter,
                           ScalarOutcome, ema_logging,
                           perform_experiments, Constant,
                           Scenario, Constraint,
                           ema_logging, save_results)
from ema_workbench.em_framework import samplers
from ema_workbench.em_framework.optimization import (
    EpsilonProgress, ArchiveLogger)
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
from ema_workbench.connectors.excel import ExcelModel
from platypus import SBX, PM, GAOperator
import matplotlib.pyplot as plt
from datetime import date
import pandas as pd
import pickle
# %% Specify model and optimization parameters
if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)
    # Set model
    model = ExcelModel("scenarioModel", wd="./models",
                       model_file='Master.xlsx')
    model.default_sheet = "EMA"
    sample_scenarios = False  # Should reference scenario(s) be sampled?
    manual_scenarios = True  # Should reference scenario(s) be specified manually?
    if sample_scenarios:
        n_scenarios = 8  # Number of scenarios to sample if sampling is used
        sampler = samplers.LHSSampler()
    load_diverse_scenarios = False  # Should a pre-generated set of diverse scenarios be loaded and used as reference?

    n_p = 8  # set # of parallel threads
    nfe = 10000  # Set number of nfes  in optimization
    date = date.today()  # Date to use for storing files
    # What set of policies should the MOEA be run for?
    #policy_types=["All levers", "No transport efficient society"]
    #policy_types=["No transport efficient society"]
    policy_types = ["All levers"]

    # Optimization parameters
    # Set epsilons
    epsilons = [.75, 5, 0.25, 0.25]  # Epsilons for M2-M5

    # Create instances of the crossover and mutation operators
    crossover = SBX(probability=1, distribution_index=20)
    mutation = PM(probability=1, distribution_index=20)
    # %% #Specify inputs
    model.uncertainties = [
        RealParameter("X1_car_demand", -0.3, 0.2, variable_name="C3"),
        RealParameter("X2_truck_demand", -0.3, 0.2, variable_name="C4"),
        RealParameter("X3_fossil_fuel_price", 0.4, 1.3, variable_name="C7"),
        RealParameter("X4_bio_fuel_price", 0.8, 2.2, variable_name="C8"),
        RealParameter("X5_electricity_price", 0.5, 1.5, variable_name="C9"),
        RealParameter("X6_car_electrification_rate", 0.35, 0.9, variable_name="C5"),
        RealParameter("X7_truck_electrification_rate", 0.10, 0.60, variable_name="C6"),
        RealParameter("X8_SAV_market_share", 0, 0.45, variable_name="C15"),
        RealParameter("X9_SAV_driving_cost", -.1, 1, variable_name="Indata!G76"),
        RealParameter("X10_SAV_energy_efficiency", 0, 0.25, variable_name="Indata!G75"),
        RealParameter("X11_VKT_per_SAV", .5, 2, variable_name="C16"),
        RealParameter("X12_driverless_truck_market_share", 0, 0.60, variable_name="C11"),
        RealParameter("X13_driverless_truck_driving_costs", -0.50, -0.20, variable_name="C14"),
        RealParameter("X14_driverless_truck_energy_efficiency", -0.2, -0.1, variable_name="C12"),
        RealParameter("X15_VKT_per_driverless_truck", 0.2, 0.5, variable_name="C13")
    ]
    # Select whether if electrification should be treated as an external uncertainty (True by default)
    External_electrification_rate = True
    if External_electrification_rate is False:
        model.constants = [Constant("C10", "No")]  # Set external electrification parameter in excel to no
        # Remove electrification uncertainties
        model.uncertainties._data.pop("X6_car_electrification_rate")
        model.uncertainties._data.pop("X7_truck_electrification_rate")
    else:
        model.constants = [Constant("C10", "Yes")]  # Set external electrification parameter in excel to no
    # Set bus energy use to "Level 1" as defualt
    model.constants = [Constant("C63", "Level 1")]

    # Specify outcomes to use in optimization. Info is required when outcome is used as constraint
    model.outcomes = [
        ScalarOutcome("M1_CO2_TTW_total", ScalarOutcome.INFO,
                      variable_name="C58"),  # Min / info

        ScalarOutcome("M2_driving_cost_car", ScalarOutcome.MINIMIZE,
                      variable_name="C54"),

        ScalarOutcome("M3_driving_cost_truck", ScalarOutcome.MINIMIZE,
                      variable_name="C55"),

        ScalarOutcome("M4_energy_use_bio", ScalarOutcome.MINIMIZE,
                      variable_name="C38"),

        ScalarOutcome("M5_energy_use_electricity", ScalarOutcome.MINIMIZE,
                      variable_name="C40"),  # min,
    ]
    # %% Create reference scenario(s)
    scenario_list = []  # list of scenario dicts to store scenarios

    # If scenarios should be sampled, sample them
    if sample_scenarios is True:
        def Extract(lst, i):
            return [item[i] for item in lst]

        scenarios = samplers.sample_uncertainties(
            model, n_scenarios, sampler=sampler)
        scenarios_dict = dict.fromkeys(scenarios.params)
        # create a dict with all scenario parameters based on scenarios
        count = 0
        for i in scenarios_dict.keys():
            scenarios_dict[str(i)] = Extract(scenarios.designs, count)
            count = count+1

        # create a scenario-dict for a single scenario
        scenario_dict = dict.fromkeys(scenarios.params)
        for j in range(len(scenarios.designs)):
            scenario_dict = dict.fromkeys(scenarios.params)
            for key in scenario_dict.keys():
                scenario_dict[str(key)] = scenarios_dict[str(key)][j]
            scenario_list.append(scenario_dict)
        df_scenarios = pd.DataFrame.from_dict(scenarios_dict)

    # If pre-generated diverse scenarios should be used, load them
    elif load_diverse_scenarios is True:
        import pickle
        df_scenarios = pickle.load(
            open("./output_data/diverse_scenarios_3.p", "rb"))
        scenario_list = []
        for i, row in df_scenarios.iterrows():
            scenario_list.append(row.to_dict())

    # If manual specification of scenario(s), specify it / them
    elif manual_scenarios is True:
        reference_scenario = {  # This is the reference scenario
            "X1_car_demand": 0,
            "X2_truck_demand": 0,
            "X3_fossil_fuel_price": 1,
            "X4_bio_fuel_price": 1,
            "X5_electricity_price": 1,
            "X6_car_electrification_rate": .68,
            "X7_truck_electrification_rate": .30,
            "X8_SAV_market_share": 0,
            "X9_SAV_driving_cost": 0,
            "X10_SAV_energy_efficiency": 0,
            "X11_VKT_per_SAV": 0,
            "X12_driverless_truck_market_share": 0,
            "X13_driverless_truck_driving_costs": 0,
            "X14_driverless_truck_energy_efficiency": 0,
            "X15_VKT_per_driverless_truck": 0
        }
        scenario_list.append(reference_scenario)
    # Store a dataframe of scenarios and number of total scenarios
    df_scenarios = pd.DataFrame(scenario_list)
    n_scenarios = len(scenario_list)
    # %% Specify constraints
    CO2_target = 0.1*18.9  # Set CO2 target [m ton CO2 eq / year], 2040 target is 10% of 2010 emission levels
    bio_target = 15  # Set target for max use of biofuels

    # Different sets of constraints. Only CO2 target is used as default

    constraints = [Constraint("max CO2", outcome_names="M1_CO2_TTW_total",
                              function=lambda x: max(0, x-CO2_target))]

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

# %% Run MOEA for each policy type and scenario
    tic = date.time.perf_counter()
    # TODO add support for reading model object depending on what policy types should be used
    for policy_type in policy_types:
        results_list = []  # List to store different sets of results
        convergence_list = []  # List to store different sets of convergence metrics
        print("Estimated total model evaluations: ", len(
            policy_types*nfe*len(df_scenarios)))
        scenario_count = 0
        print("Running optimization for policy type ", policy_type)
        if policy_type == "All levers":
            # Specification of levers depending on policy type
            model.levers.clear()
            model.levers = [
                RealParameter("L1_bio_share_diesel",
                              0, 1,
                              variable_name="C76"),
                RealParameter("L2_bio_share_gasoline",
                              0, 1,
                              variable_name="C77"),
                RealParameter("L3_additional_car_energy_efficiency",
                              0.0, .05, variable_name="C72"),
                RealParameter("L4_additional_truck_energy_efficiency",
                              0.0, .05, variable_name="C73"),
                RealParameter("L5_fuel_tax_increase_gasoline",
                              0, .12, variable_name="C70"),
                RealParameter("L6_fuel_tax_increase_diesel",
                              0, .12, variable_name="C71"),
                RealParameter("L7_km_tax_cars",
                              0, 1, variable_name="C68"),
                RealParameter("L8_km_tax_trucks",
                              0, 2, variable_name="C69"),
                RealParameter("L9_transport_efficient_planning_cars",
                              .0, .26, variable_name="C74"),
                RealParameter("L10_transport_efficient_planning_trucks",
                              .0, .17, variable_name="C75")
            ]

        if policy_type == "No transport efficient society":
            model.levers.clear()
            model.levers = [
                RealParameter("L1_bio_share_diesel",
                              0, 1,
                              variable_name="C76"),
                RealParameter("L2_bio_share_gasoline",
                              0, 1,
                              variable_name="C77"),
                RealParameter("L3_additional_car_energy_efficiency",
                              0.0, .05, variable_name="C72"),
                RealParameter("L4_additional_truck_energy_efficiency",
                              0.0, .05, variable_name="C73"),
                RealParameter("L5_fuel_tax_increase_gasoline",
                              0, .12, variable_name="C70"),
                RealParameter("L6_fuel_tax_increase_diesel",
                              0, .12, variable_name="C71"),
                RealParameter("L7_km_tax_cars",
                              0, 1, variable_name="C68"),
                RealParameter("L8_km_tax_trucks",
                              0, 2, variable_name="C69")
            ]

        for scenario in scenario_list:
            print("Scenario: ", scenario_count)
            reference = Scenario()
            reference.data = scenario

            convergence_metrics = [
                ArchiveLogger(
                    "./archives",
                    [lever.name for lever in model.levers],
                    [outcome.name for outcome in model.outcomes if outcome.kind != 0],
                    base_filename=f"{str(nfe)}_{policy_type}_{str(date.today())}.tar.gz",
                ),
                EpsilonProgress(),
            ]

            # Create an instance of GAOperator with the operators
            variator_instance = GAOperator(crossover, mutation)

            with MultiprocessingEvaluator(msis=model, n_processes=n_p) as evaluator:
                results, convergence = evaluator.optimize(nfe=nfe, searchover='levers',
                                                          epsilons=epsilons,
                                                          convergence=convergence_metrics,
                                                          constraints=constraints,
                                                          reference=reference,
                                                          population_size=100,
                                                          variator=variator_instance
                                                          )

            scenario_count = scenario_count+1
            if policy_type == "No transport efficient society":
                model.levers = [RealParameter("L9_transport_efficient_planning_cars",
                                              0, .001, variable_name="C74"),
                                RealParameter("L10_transport_efficient_planning_trucks",
                                              0, .001, variable_name="C75")]
                results["L9_transport_efficient_planning_cars"] = 0
                results["L10_transport_efficient_planning_trucks"] = 0

            # plot epsilon progress
            fig, (ax1) = plt.subplots(ncols=1, sharex=True, figsize=(8, 4))
            ax1.plot(convergence.nfe, convergence.epsilon_progress)
            ax1.set_ylabel('$\epsilon$-progress')
            ax1.set_xlabel('number of function evaluations')
        results_list.append(results)
        convergence_list.append(convergence)

        toc = date.time.perf_counter()
        print("Runtime [s] = " + str(toc-tic))
        print("Runtime [h] = " + str(round((toc-tic)/3600, 1)))

        # Save results to file?
        save_files = True
        if save_files is True:

            filename = str(nfe)+'_nfe_directed_search_MORDM_'+str(date)
            filename1 = policy_type+filename+'.p'
            pickle.dump([results_list, convergence_list, df_scenarios, epsilons], open(
                "./output_data/"+filename1, "wb"))
            filename2 = policy_type+filename+'model_'+".p"
            pickle.dump(model, open("./output_data/"+filename2, "wb"))
