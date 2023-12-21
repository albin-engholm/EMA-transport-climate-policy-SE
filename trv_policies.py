# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:46:59 2023

@author: aengholm
"""
import pandas as pd
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

# Flatten the list of dictionaries into a single dictionary
flattened_policies_trv = {name: dict(zip(keys_trv, policy))
                          for name, policy in zip(policy_names_trv, all_policies_trv)}

# Convert the dictionary to a pandas DataFrame
df_trv = pd.DataFrame.from_records(flattened_policies_trv).T
df_trv["Policy type"] = "Trv"
