# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:43:46 2023

@author: aengholm
"""

import re
from ema_workbench.analysis import feature_scoring
from ema_workbench.analysis.pairs_plotting import (pairs_lines, pairs_scatter,
                                                   pairs_density)
import numpy as np
import pandas as pd
from ema_workbench import (RealParameter, TimeSeriesOutcome, ScalarOutcome, ema_logging,
                           perform_experiments)

import SALib
import seaborn as sns
import pickle
from SALib.analyze import sobol
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
import matplotlib.pyplot as plt
SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['figure.dpi'] = 300

sns.set(rc={"figure.dpi": 300})

# Should previously saved result data be loaded? If not, data from workspace is used
load_results = 1
if load_results == 1:
    from ema_workbench import load_results

    ### USER INPUT ###
    t1 = "./output_data/sobol_results/2048_SOBOL_uncertainties_2024-01-05"
    results = pickle.load(open(t1+".p", "rb"))
    #results = load_results(t1+'.tar.gz')
    experiments = results[0]
    outcomes = results[1]
    model = pickle.load(open(t1+"model_.p", "rb"))
    #[outcome_df, uncertainties_df]=pickle.load( open( t1+"outcomes_uncertainties_.p", "rb" ) )
uncertainties_df = pd.DataFrame.from_dict(results[0])

# Create SALib problem
if "levers" in t1:
    problem = get_SALib_problem(model.levers)
elif "uncertainties" in t1:
    problem = get_SALib_problem(model.uncertainties)

problem_df = problem
# for i in range(len(problem["names"])) :
#     a=problem["names"][i]
#     b=uncertainties_df[a]
#     c=b.values.tolist()
#     problem["names"][i]=c[0][0]
#     print(a)
#     print (b)


outcome_names = list(model.outcomes.keys())
outcome_name = "M1_CO2_TTW_total"
Si = sobol.analyze(problem, outcomes[outcome_name],
                   calc_second_order=True, print_to_console=True)

scores_filtered = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
Si_df = pd.DataFrame(scores_filtered, index=problem['names'])


sns.set_style('white')
fig, ax = plt.subplots(1)

indices = Si_df[['S1', 'ST']]
err = Si_df[['S1_conf', 'ST_conf']]

indices.plot.bar(yerr=err.values.T, ax=ax)
fig.set_size_inches(16, 12)
fig.subplots_adjust(bottom=0.3)
plt.show()

# %% Create bar chart and heatmap of sobol indices
only_objectives_heatmap = True
if only_objectives_heatmap:
    all_outcomes = model.outcomes.keys()
    objective_names = [s for s in all_outcomes if re.match(r"M\d+", s)]

    outcome_names = []
    for outcome in model.outcomes:
        if outcome.name in objective_names:
            outcome_names.append(outcome.name)
else:
    outcome_names = list(model.outcomes.keys())
sobol_indices = {}

for outcome_name in outcome_names:
    Si = sobol.analyze(problem, outcomes[outcome_name], calc_second_order=True)
    sobol_indices[outcome_name] = Si['ST']

# Create a DataFrame from the Sobol indices dictionary
sobol_df = pd.DataFrame(sobol_indices, index=problem['names'])

# Sort the columns of the DataFrame using a custom sorting key
sobol_df = sobol_df.reindex(sorted(sobol_df.columns, key=lambda x: int(
    re.search(r'\d+', x).group()) if re.search(r'\d+', x) is not None else 0), axis=1)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(sobol_df.applymap(lambda x: 0 if x < 0.0001 else x), annot=True, cmap='viridis')
plt.title('Total Sobol Indices Heatmap')
plt.xlabel('Outcomes')
plt.ylabel('Parameters')
plt.show()


# %% Plot results for each outcome
threshold = 0.01
Si_df["ub_ST"] = Si_df["ST"]+Si_df["ST_conf"]
Si_df["ub_S1"] = Si_df["S1"]+Si_df["S1_conf"]

S1_below_t = Si_df["ub_S1"] < threshold
ST_below_t = Si_df["ub_ST"] < threshold

below_both = S1_below_t & ST_below_t
Si_df_screened = Si_df.drop(Si_df[below_both].index)
for outcome_name in outcome_names:
    fig, ax = plt.subplots(1)

    indices = Si_df_screened[['S1', 'ST']]
    err = Si_df_screened[['S1_conf', 'ST_conf']]

    indices.plot.bar(yerr=err.values.T, ax=ax)
    fig.set_size_inches(16, 12)
    fig.subplots_adjust(bottom=0.3)
    plt.show()

    Si_cumsum_df = Si_df.sort_values(by=['ST'], ascending=False)
    Si_cumsum_df = Si_cumsum_df.cumsum()
    plt.plot(Si_cumsum_df["ST"])

    plt.xticks(rotation=90)

    Si_sorted = Si_df["ST"].sort_values(ascending=False)

    plt.figure()
    plt.hist(outcomes[outcome_name], 100)
