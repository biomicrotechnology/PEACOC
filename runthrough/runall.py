import os
import sys
import subprocess
import yaml


initpath = sys.argv[1] ##for command-line usage
#initpath = '/home/weltgeischt/EXAMPLES/yml_runs/run_smrToResample.yml'
with open(initpath, 'r') as ymlfile: cfg = yaml.safe_load(ymlfile)

rootdir = cfg['settings']['runfiles']
fnames = ['rawToResampled','polarityCheck','artifactCheck','LFPtoBursts']
filenames = [os.path.join(rootdir,fname+'.py') for fname in fnames]


for myscript in filenames: subprocess.call([sys.executable,myscript,sys.argv[1]])#execfile(myscript)
