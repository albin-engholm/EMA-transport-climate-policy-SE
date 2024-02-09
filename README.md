# EMA-transport-climate-policy-SE
The Swedish Transport Administration has developed the [Scenario Tool](https://bransch.trafikverket.se/tjanster/system-och-verktyg/Prognos--och-analysverktyg/scenarioverktyget-for-styrmedelsanalyser/), a model to support analyses of how different types of transport cliamte policy interventions can affect the Swedish transport system on an aggregated level.

EMA_transport_climate_policy_SE supports performing MORDM or other analyses based on Exploratory Modeling and Analysis using the Scenario Tool. 

The repository has been developed within the research project [MUST: Managing deep Uncertainty in planning for Sustainable Transport](https://www.itrl.kth.se/research/ongoingprojects/must-1.1146492) performed by KTH ITRL and VTI. The primary purpose of EMA_transport_climate_policy_SE is to support a study within the project and not for external use. However, the repo is public  to enable reproduction of an upcoming paper and to serve as inspiration for other projects or applications.

# Contents, maintenance, documentation and contributions
The repository includes a number of scripts to support MORDM and other EMA-related analyses using the Scenario Tool. The scripts use to perform the MORDM study, which is the core of the research project, have the prefix MORDM. 

Running the MORDM scripts to recreate the MORDM study involves the below steps. These scripts can also be used to redo the study with alternative problem formulations (e.g. optimization problem formulation in MORDM_MOEA.py, exposing the candidate policies to other uncertainties in MORDM_run_robustness_analysis, or using other robustness metrics or vulnerability analysis approaches in MORDM_analyze_robustness.py). To run these scripts, the user needs to ensure that the file paths for the input files (output from previous step(s)) are correct in each script. 

MORDM_MOEA.py --> (MORDM_convergence_metrics.py and MORDM_animation.py) --> MORDM_filter_candidate_policies.py --> MORDM_run_robustness_analysis.py --> MORDM_analyze_robustness.py

TODO: add a flowchart of scripts and output/input files for each step

Other scripts have been used for various trials and supporting analyses but are not part of the final study. These are not maintained and might not be runnable or work as intended.

The repository is not actively maintained to support external use. No additional documentation, relase notes, issue tracking or support is currently available and is not planned. The repository is not open for external contribution. 

There is active work ongoing with the repo to finalize the research study.

# Disclaimer: modified Scenario Tool is used in the repository
The repository uses a modified version of the Scenario Tool which has been developed within the MUST project. The Swedish Transport Administration, which is the original creator and owner of "Scenario Tool v1.0", has not participated in, approved, or in any other way been involved in the development of these changes and additions. Consequently, the Swedish Transport Administration does not assume any responsibility for the function, content, or any results that arise from the use of this modified version of the tool. Users of this version of the Scenario Tool should be aware that all changes and additions are exclusive to the MUST project and may differ from the Swedish Transport Administration's official version and recommendations.  

# Licenses
TODO
