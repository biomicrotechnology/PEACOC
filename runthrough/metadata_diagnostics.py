# call: python metadata_diagnostics.py   --format=odml -md /home/weltgeischt/EAdetection_tutorial/run_params/my_recording_params2.yml
import getopt
import sys
import os
import yaml

diagnosticsOn = False
metadataOn = False


#we need the hdf5 file --> from this we read the paraemters
options, remainder = getopt.getopt(sys.argv[1:], 'd:m', ['format=',
                                                         'diagnostics',
                                                         'metadataOn',
                                                         ])


for opt, arg in options:
    if opt in ('-d', '--diagnostics'):
        diagnosticsOn = True
    elif opt in ('-m', '--metadata'):
        metadataOn = True
    elif opt == '--format':
        format = arg

paramfile = sys.argv[-1]
print ('format',format)
print ('metadata',metadataOn)
print ('diagnostics',diagnosticsOn)
print ('paramfile',paramfile)

assert format in ['yaml','odml'], '--format must be = yaml or odml'

with open(paramfile, 'r') as ymlfile: cfg = yaml.safe_load(ymlfile)
sys.path.append(cfg['settings']['code_path'])

import os
import core.ea_management as eam
import core.helpers as hf
import core.metadata_writing as mw

import odml #pip3 install odml
# diagnosticsOn = True
# metadataOn = True
# format = 'odml'



#paramfile = '/home/weltgeischt/EAdetection_tutorial/run_params/my_recording_params.yml' #todo retrieve from command-line
aRec = eam.Rec(init_ymlpath=paramfile)
results = hf.open_hdf5(aRec.resultsfileH) #this opens the main results file

#path for saving
savepath = os.path.join(aRec._fullpath,'%s__'%aRec.id)
if metadataOn and diagnosticsOn:
	savepath += 'metadata_diagnostics'
elif metadataOn:
	savepath += 'metadata'
elif diagnosticsOn:
	savepath += 'diagnostics'
else:
	assert 0, 'set at least metadata or dignostics = True'


anagroups = ['EdDetection','SpikeSorting','BurstClassification']#,

template_path = eam.configpath.replace('configDisplay','meta_template')
with open(template_path, 'r') as ymlfile: templatedict = yaml.safe_load(ymlfile)
#todo make this open for windows --> os etc.


#retrieve the metadata into a dictionary
metadata = {}

if metadataOn:
	print ('retrieving metadata')
	metadata['general_info'] = mw.make_superdict(templatedict,aRec)
	for anagroup in anagroups:
		methdict = mw.generate_metadict(anagroup, templatedict, 'methods', aRec.resultsfileH)
		infodict = mw.generate_metadict(anagroup, templatedict, 'info', aRec.resultsfileH)
		metadata[anagroup] = {'methods':methdict,'info':infodict}

#diagnostics
if diagnosticsOn:
	print ('calculating diagnostics')
	diagnostics_dict = mw.calc_diagnostics(aRec,templatedict)
	metadata['Diagnostics'] = diagnostics_dict

if format == 'yaml':
	print ('writing yml')
	metadata = mw.delete_by_key(metadata,['dtype'])
	metadata = mw.string_value(metadata) #make values strings for yaml.dump to work nicely
	mw.dump_yml(metadata,savepath+'.yml')


if format == 'odml':
	print ('writing odml')
	import odml
	import getpass
	import time

	anagroups = ['EdDetection','SpikeSorting','BurstClassification'] if metadataOn else []
	S = mw.dict_to_odml(metadata,anagroups=anagroups)
	mydoc = odml.Document()
	mydoc.author = getpass.getuser()
	mydoc.date = time.strftime('%Y-%m-%d')
	mydoc.append(S.sect)

	#odml.save(mydoc, 'odmltest.odml')
	odml.tools.xmlparser.XMLWriter(mydoc).write_file(savepath+'.xml', local_style=True)


