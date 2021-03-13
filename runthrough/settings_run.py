from __future__ import print_function

import sys
import yaml
import os

initpath = sys.argv[1] ##for command-line usage
#initpath = '/home/weltgeischt/EXAMPLES/Enya/yml_setup/EP5_doubletterec_py3.yml'
#initpath = "/home/weltgeischt/EAnalysis_EXAMPLES/Enya/yml_setup/EP63_05Hz1_HCi1__runparams.yml"
with open(initpath, 'r') as ymlfile: cfg = yaml.safe_load(ymlfile)
sys.path.append(cfg['settings']['code_path'])

filedict = cfg['config_files']
checkmake_envvar = lambda confkey, setkey: os.environ.update({confkey: filedict[setkey]}) if setkey in filedict else 0
for ckey, skey in zip(['displayConfig', 'analysisConfig'], ['display', 'analysis']): checkmake_envvar(ckey, skey)

import core.helpers as hf
import core.ed_detection as edd
import core.ea_management as eam
multiproc_on_preproc = cfg['preprocessing']['loading']['multiprocessing'] if 'multiprocessing' in cfg['preprocessing']['loading'] else False
mp_on = 1 if 'Multiprocessing' in cfg or multiproc_on_preproc else False
if mp_on: import core.multiproc as cmp

aRec = eam.Rec(init_ymlpath=initpath) #initializing

settings_run = True

################
# NOW IN THE run_LFPtoBursts
import core.ea_analysis as eana
subrun = cfg['analysis']


checkrun = lambda cobj: type(cobj).__name__ in subrun['run']#to check whether the respective analysis should be run
#setparams = lambda cobj: [cobj.setparam(item[0],item[1]) for item in list(subrun[type(cobj).__name__]['setparams'].items())] and None if type(subrun[type(cobj).__name__]['setparams'])==dict else None
setparams = lambda cobj,cdict: [cobj.setparam(item[0],item[1]) for item in list(cdict[type(cobj).__name__]['setparams'].items())] and None if type(cdict[type(cobj).__name__]['setparams'])==dict else None



aRec.create_paths()
