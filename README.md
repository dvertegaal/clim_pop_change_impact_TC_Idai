# Code to attribute the historical effect of climate and population change on the affected population of tropical cyclone Idai
This work is part of the paper Vertegaal et al. (submitted) and Work Package 1 of the COMPASS project whose overarching objective is to characterise compound extremes in current and future climates. COMPASS (COMPound extremes Attribution of climate change: towardS an operational Service) aims to develop a harmonized, yet flexible, methodological framework for **climate and impact attribution** of various complex **extremes** that include compound, sequential and cascading hazard events. For more information and useful links about the project, have a look at the introduction on the [COMPASS Github repository](https://github.com/HORIZON-COMPASS)

![logoCOMPASS](https://github.com/user-attachments/assets/4c3b95d4-bfc0-4727-a1e8-ee6653a03b5e)


## Scripts and data to reproduce the figures and results from Vertegaal et al. (submitted)
### Data
For the resulting flood map from the models, the files are provided in the *data* folder that first has to be unzipped. 

For other data from open sources, links and intsructions are provided on how to obtain and process the data. The structure of the data folders is provided

For questions, you can contact the authors (main contact: doris.vertegaal@deltares.nl).


### Scripts
The scripts in the folder *scripts* can be used to reproduce the figures and numbers presented in Vertegaal et al. (submitted). On top of every script is stated which Pixi environment from the pixi.toml file can be used to run the scripts (see below). 

   
The figures and table are saved according to their figure or table number in the paper. See the Table below for which script is used for the production of which asset:

| Asset                   | Path                                                                   |
|-------------------------|------------------------------------------------------------------------|
| F01                     | *NA*                                                                   |
| F02                     | Socio-economic\scripts\change_affected_pop.py                          |
| F03                     | Socio-economic\scripts\change_affected_pop.py                          |
| F04                     | Socio-economic\scripts\change_affected_pop.py                          |
| F05                     | Socio-economic\scripts\analyse_synthpop_data_GLOPOP-SG.py              |
| Table 1                 | *NA*                                                                   |
|                                                                                                  |
| *Supplement*                                                                                     |
| FS1                     | Socio-economic\scripts\comparing_pop_data.py                           |
| FS2                     | Socio-economic\scripts\change_affected_pop.py                          |
| FS3                     | Socio-economic\scripts\change_affected_pop.py                          |
| FS4                     | Socio-economic\scripts\change_affected_pop.py                          |
| FS5                     | Socio-economic\scripts\change_affected_pop.py                          |
| FS6                     | Socio-economic\scripts\analyse_synthpop_data_GLOPOP-SG.py              |
| FS7                     | Socio-economic\scripts\analyse_synthpop_data_GLOPOP-SG.py              |
| Table S1                | *NA*                                                                   |
| Table S2                | Socio-economic\scripts\comparing_pop_data.py                           |
| Table S3                | Socio-economic\scripts\comparing_pop_data.py                           |
| Table S4                | Socio-economic\scripts\comparing_pop_data.py                           |
| Table S5                | Socio-economic\scripts\change_affected_pop.py                          | 
| Table S6                | Socio-economic\scripts\change_affected_pop.py                          | 
| Table S7                | Socio-economic\scripts\analyse_synthpop_data_GLOPOP-SG.py              | 


### Pixi istallation instructions
You will need [pixi](https://pixi.sh/latest/#installation) to install the required dependencies (defined in specific environments). 

To install dependencies: `pixi install`. This will install all the environments required to run all the workflows. To install only specific environments, mention it, for example : `pixi install -e compass-socio`. All the enviroments available are listed [environments] in the pixi.toml file


## Acknowledgements

![EU_logo](https://github.com/user-attachments/assets/e2fad699-697e-43fd-84be-032447d6dd21) The COMPASS project has received funding from the European Union’s HORIZON Research and Innovation Actions Programme under Grant Agreement No. 101135481

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or of the European Health and Digital Executive Agency (HADEA). Neither the European Union nor the granting authority HADEA can be held responsible for them.