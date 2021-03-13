# PEACOC: Patterns of Epileptiform Activity: a COntinuum-based Classification (and detection!)
Publication: [Inserted upon acceptance]

### 0 – Get prerequisites
* Install Python (both Python 2 and 3 work!)
* Install requirements: see PEACOC/good_to_know/installing_requirements.txt
* Download/clone this repository

### 1 – Copy template
Copy the file template__runparams1.yml from ~/PEACOC/templates in a folder where you would like to store your personalized templates and rename it (e.g. my_run_templates/template__runparamsExperiment1.yml)

### 2 – Manually adapt your template to your GENERAL preferences
Open your copied template file and use FIND-REPLACE to replace  ...
* PACKAGE_DIR → where you cloned/downloaded PEACOC to (e.g. /home/JonSnow/workspace)
* DATADIR → where you want your results saved
* FIGDIR → where you want the figures saved
* SOURCEDIR → directory in which you keep your .smr files

### 3 – Generate SPECIFIC parameter files from template
See PEACOC/examples/yml_generators.py

### 4 – Run PEACOC!
See: PEACOC/examples/commandline_runthrough.txt

### 5 – Harvest results, visualize, explore ...
See: PEACOC/examples/use_results.py

