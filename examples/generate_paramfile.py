from core.helpers import generate_ymlsetup

templpath = 'PATH_TO_YOUR_TEMPLATE_DIRECTORY/template__runparams.yml' #e.g for me PATH_TO_YOUR_TEMPLATE_DIRECTORY is something like /home/weltgeischt/EXAMPLES/Enya/yml_setup

# gererate one runparams file
my_id = 'EP11_1Hz2_HCi1'
my_smr = 'EP9_p2-1-EP11_1Hz2.smr'

generate_ymlsetup(my_id, my_smr, templpath)

# routine for several id-smr combinations
rundict = {'EPXX_YHzW_HCZ': 'EPXX_YHzW.smr', \
           'my_cute_id': 'sourcefile.smr'}
for my_id, my_smr in rundict.items():  generate_ymlsetup(my_id, my_smr, templpath,setuppath=templpath.replace('template',my_id))
