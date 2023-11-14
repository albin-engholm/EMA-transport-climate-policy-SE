# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:18:42 2023

@author: aengholm
"""
from ema_workbench import (RealParameter, CategoricalParameter,
                           ScalarOutcome, ema_logging,
                           perform_experiments, Constant,Policy)

from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)
    # %% Load candidate policies and model from previous optimization
    import pandas as pd
    df_full = pd.DataFrame()
    policy_types = ["All levers", "No transport efficient society"]
    #policy_types=["All levers"]#,"No transport efficient society"]
    load_results = 1
    if load_results == 1:
        
        
        count = 0
        for policy_type in policy_types:
            if count == 0:
                date = "2023-11-10"
                nfe = 500000
                t1 = './output_data/'+policy_type + \
                    str(nfe)+"_nfe_"+"directed_search_MORDM_"+date+".p"
               # =str(nfe)+'_nfe_directed_search_sequential_'+str(date.today())+'_'+str(n_scenarios)+'_scenarios'
                import pickle
                results_list, convergence, scenarios , epsilons= pickle.load(
                    open(t1, "rb"))
                t2 = './output_data/'+policy_type + \
                    str(nfe)+"_nfe_"+"directed_search_MORDM_"+date+"model_.p"
                model = pickle.load(open(t2, "rb"))
                scenario_count = 0
                for results in results_list:
                    results["Policy type"] = policy_type
                    df_full = pd.concat([df_full, results], ignore_index=True)
                    scenario_count = scenario_count+1

            if count == 1:
                date = "2023-11-14"
                nfe = 150000
                t1 = './output_data/'+policy_type + \
                    str(nfe)+"_nfe_"+"directed_search_MORDM_"+date+".p"
               # =str(nfe)+'_nfe_directed_search_sequential_'+str(date.today())+'_'+str(n_scenarios)+'_scenarios'
                import pickle
                results_list, convergence, scenarios, epsilons = pickle.load(
                    open(t1, "rb"))
                t2 = './output_data/'+policy_type + \
                    str(nfe)+"_nfe_"+"directed_search_MORDM_"+date+"model_.p"
                model = pickle.load(open(t2, "rb"))
                for results in results_list:
                    results["Policy type"] = policy_type
                    df_full = pd.concat([df_full, results], ignore_index=True)
            count = count+1
            
        #Load candidate policy dataframe
        date = "2023-11-14"
        nfe = 150000
        filename=date+"_"+str(nfe)+"candidate_policies"+".p"    
        candidate_policy_data = pickle.load(open("./output_data/"+filename, "rb"))

    # The model object already contains all information about levers and uncertainties, so no need to specify these things again

    

#%% include trv policies

    keys_trv = [
       'L3_additional_car_energy_efficiency',
       'L4_additional_truck_energy_efficiency',
       'L6_fuel_tax_increase_diesel',
       'L5_fuel_tax_increase_gasoline', 
       'L1_bio_share_diesel', 
       'L2_bio_share_gasoline',
       'L9_transport_efficient_planning_cars',
       'L10_transport_efficient_planning_trucks',
       'L7_km_tax_cars', 
       'L8_km_tax_trucks'
       ]

    policy1 = (0, 0,  .02, .02, .698, .728,  0, 0, 0, 0)
    policy2 = (0, 0,  .09, .09, .59, .62, 0, 0, 0, 0)
    policy3 = (0, 0,  .05, .05, .61, .64, 0, 0, 1, 2)
    policy4 = (0, 0,  .097, .097, .48, .51, 0, 0, 1, 2)
    policy5 = (0, 0,  .12, .12, .41, .44, 0, 0, 0, 0)
    policy6 = (0.05, 0.05,  .031, .031, .65, .65, .10, .050, .50, 1)
    policy7 = (.05, .05,  .092, .092, .495, .525, .10, .050, .50, 1)
    policy8 = (.05, .05,  .063, .063, .508, .538, .26, .17, .50, 1)
    policy9 = (0, 0,  .02, .02, .32, 0.075, 0, 0, 0, 0)
    
    # Create a list of all policies
    all_policies_trv = [policy1, policy2, policy3, policy4, policy5, policy6, policy7, policy8, policy9]
    
    # Create a list of policy names
    policy_names_trv = ['B', 'C1', 'C2', 'C3', 'C4', 'D1', 'D2', 'D3', 'Reference policy']
    
    # Now create the list of dictionaries
    policies_trv = [{name: dict(zip(keys_trv, policy))} for name, policy in zip(policy_names_trv, all_policies_trv)]
    import pandas as pd

    # Flatten the list of dictionaries into a single dictionary
    flattened_policies_trv = {name: dict(zip(keys_trv, policy)) for name, policy in zip(policy_names_trv, all_policies_trv)}
    
    # Convert the dictionary to a pandas DataFrame
    df_trv = pd.DataFrame.from_records(flattened_policies_trv).T
    df_trv["Policy type"] ="Trv"
    
    print(df_trv)
    
        
    candidate_policy_data = pd.concat([candidate_policy_data, df_trv], ignore_index=False)

    candidate_policy_data.loc[candidate_policy_data['L9_transport_efficient_planning_cars'] < 0.005, 'L9_transport_efficient_planning_cars'] = 0
    candidate_policy_data.loc[candidate_policy_data['L10_transport_efficient_planning_trucks'] < 0.005, 'L10_transport_efficient_planning_trucks'] = 0


    #%% Create a list of all policies
    candidate_policies = []
    for i, policy in candidate_policy_data.iterrows():
        candidate_policies.append(Policy(str(i), **policy.to_dict()))
        

        # %% Add other interesting outcomes to calculate

    model.outcomes = [
        ScalarOutcome("CO2 TTW change light vehicles", ScalarOutcome.INFO,
                      variable_name="C32"),

        ScalarOutcome("CO2 TTW change trucks", ScalarOutcome.INFO,
                      variable_name="C33"),

        ScalarOutcome("VKT light vehicles", ScalarOutcome.INFO,
                      variable_name="C35"),

        ScalarOutcome("VKT trucks", ScalarOutcome.INFO,
                      variable_name="C36"),

        ScalarOutcome("VKT total", ScalarOutcome.INFO,
                      variable_name="C37"),

        ScalarOutcome("Energy fossile total", ScalarOutcome.INFO,
                      variable_name="C39"),

        ScalarOutcome("Energy total", ScalarOutcome.INFO,
                      variable_name="C41"),

        ScalarOutcome("Electrified VKT share light vehicles", ScalarOutcome.INFO,
                      variable_name="C42"),
        ScalarOutcome("Electrified VKT share trucks", ScalarOutcome.INFO,
                      variable_name="C43"),
        ScalarOutcome("Electric share of total energy", ScalarOutcome.INFO,
                      variable_name="C44"),


        ScalarOutcome("Fossile fuel price relative reference trucks", ScalarOutcome.INFO,
                      variable_name="C52"),
        ScalarOutcome("Fossile fuel price relative reference light vehicles", ScalarOutcome.INFO,
                      variable_name="C53"),

        ScalarOutcome("Delta CS light vehicles", ScalarOutcome.INFO,
                      variable_name="C47"),
        ScalarOutcome("Delta CS trucks", ScalarOutcome.INFO,
                      variable_name="C48"),
        ScalarOutcome("Delta CS total", ScalarOutcome.INFO,
                      variable_name="C49"),
        ScalarOutcome("Delta tax income total", ScalarOutcome.INFO,
                      variable_name="C50"),
        ScalarOutcome("Delta CS tax", ScalarOutcome.INFO,
                      variable_name="C51")
    ]

    # scenarios=[scenarios,reference]
    #%% Set up two runs for various uncertainty parameters #Run model - for open exploration
    n_p = 8
    # Run
    import time
    tic = time.perf_counter()
    results_list_OE=[]
    uncertainty_sets=["X","XP"]
    for uncertainty_set in uncertainty_sets:
        if uncertainty_set=="X":
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
                "X15_VKT_per_driverless_truck": 0
            }
        if uncertainty_set=="XP":

            # Add P parameters
            model.uncertainties = [
                RealParameter("R1_fuel_price_to_car_electrification",0.1,0.4,variable_name="C19"),
                RealParameter("R2_fuel_price_to_truck_electrification",0,0.5,variable_name="C22"),
                RealParameter("R3_fuel_price_to_car_fuel_consumption",-.15,-0.05,variable_name="C24"),
                RealParameter("R4_car_driving_cost_to_car_ownership",-0.2,-0.1,variable_name="C20"),
                RealParameter("R5_car_driving_cost_to_car_VKT",-0.4,-0.1,variable_name="C21"),
                RealParameter("R6_truck_driving_cost_to_truck_VKT",-1.2,-0.2,variable_name="C22")]
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

        #  Generate scenarios to  explore
        from ema_workbench.em_framework import samplers
        from ema_workbench import Scenario

        # Create the reference scenario and add to scenario list

        scenario_list = []
        
        reference = Scenario("Reference", **mid_case)
        scenario_list.append(reference)
        # Sample additional scenarios
        nr_scenarios = 500  # select number of scenarios (per policy)
        scenarios = samplers.sample_uncertainties(model, nr_scenarios, sampler=samplers.LHSSampler())
        # Create scenario objects of the sampled scenarios and add to scenarioset
        for i in range(nr_scenarios):
            s_dict = {k: v for k, v in zip(scenarios.params, scenarios.designs[i])}
            scenario_list.append(Scenario(str(i), **s_dict))
          
            
        with MultiprocessingEvaluator(msis=model, n_processes=n_p) as evaluator:
            experiments,outcomes = perform_experiments(model,
                                                        scenarios=scenario_list,
                                                        evaluator=evaluator,
                                                        policies=candidate_policies)
            
         #  Add policy type to experiments
        experiments_backup=experiments.copy()   
        experiments=experiments_backup
        count = 0
        policy_type_list=[]
        for policy in experiments["policy"]:
            #if policy not in policy_names_trv
            if policy not in policy_names_trv: 
                #print(policy)
                policy_type_list.append(candidate_policy_data.at[int(policy), "Policy type"])
            else:
                #print(policy)
                policy_type_list.append(candidate_policy_data.at[policy, "Policy type"])
            count = count+1
        experiments["Policy type"]=policy_type_list
           
        results=[outcomes,experiments]
        results_list_OE.append(results)
        # %%
    save_results = 1
    if save_results == 1:
        from datetime import date
        from ema_workbench import save_results

        filename = str(nr_scenarios)+'_scenarios_MORDM_OE_'+str(date.today())

        filename1 = filename+'.p'
        #save_results(results, "./output_data/"+filename1)
        pickle.dump([experiments, outcomes], open("./output_data/"+filename1, "wb"))
        import pickle
        filename2 = filename+'model_'+".p"
        pickle.dump(model, open("./output_data/"+filename2, "wb"))
        pickle.dump([results_list_OE],open("./output_data/"+"X_XP"+filename1,"wb"))
