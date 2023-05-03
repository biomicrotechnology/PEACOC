import core.helpers as hf
import numpy as np


#todo check for maskarray and make graveyard message


'''
def fill_prop(propname,indict,sect):
    if 'dtype' in indict:
        if indict['dtype'] == 'url':
            if not indict['value'][:7] == 'file://':
                indict['value'] = 'file://' + indict['value']

    if propname in sect.properties: sect.remove(sect.properties[propname])
    Prop = odml.Property(name=propname,**indict)
    sect.append(Prop)

def fill_section(odmldict,sect):
    for key,indict in odmldict.items():
        fill_prop(key,indict,sect)'''

class OSection(object):
	def __init__(self, name,type='analysis'):
		self.odml = __import__('odml')
		#self.type = type
		#self.name = name
		self.sect = self.odml.Section(name=name,type=type)

	def fill_section(self,odmldict):
		for key, indict in odmldict.items():
			self.fill_prop(key, indict)

	def fill_prop(self,propname, indict):
		checkmake_url(indict)
		if propname in self.sect.properties: self.sect.remove(self.sect.properties[propname])
		#print (propname,indict)
		if type(indict['value']) == np.ndarray: indict['value'] = list(indict['value'])
		Prop = self.odml.Property(name=propname, **indict)
		self.sect.append(Prop)

def checkmake_url(indict):
    if 'dtype' in indict:
        if indict['dtype'] == 'url':
            if type(indict['value']) == list:
                indict['value'] = [make_url(val) for val in indict['value']]
            else: indict['value'] = make_url(indict['value'])

def make_url(val):
    if val[:7] == 'file://': return val
    else: return 'file://' + val

def generate_metadict(anagroup,template_dict,subkey,hdf5results):
	groupkey = template_dict[anagroup]['groupkey']['value']
	if subkey =='info':	metadict = template_dict[subkey] #general info
	if subkey == 'methods': metadict = template_dict[anagroup]
	mydata = hf.open_hdf5(hdf5results,'/%s/%s'%(groupkey,subkey))
	for key,value in mydata.items():
		if key in metadict:	metadict[key]['value'] = value

	return metadict

def make_superdict(template_dict,aRec):
	metadict = template_dict['super']
	inputdict = {'parameters_run': aRec.init_ymlpath, \
				 'resultsfile': aRec.resultsfileH, \
				 'analysis_config': aRec.configpath_ana, \
				 'rawfile': aRec.rawfileH}
	for key, value in inputdict.items(): metadict[key]['value'] = value
	return metadict


def dict_to_odml(metadata,anagroups=['EdDetection','SpikeSorting','BurstClassification']):
	"""metadata is a dictionary"""
	S = OSection('EA_analysis')
	if len(anagroups)>0: S.fill_section(metadata['general_info'])
	if 'Diagnostics' in metadata:
		S1 = OSection('Diagnostics',type='analysis/results')
		S1.fill_section(metadata['Diagnostics'])
		S.sect.append(S1.sect)
	for anagroup in anagroups:
		S0 = OSection(anagroup)
		for subkey,mytype in zip(['methods','info'],['parameters','info']):
			#print (anagroup,subkey)
			S1 = OSection(anagroup+'_'+subkey,type='analysis/%s'%mytype)
			S1.fill_section(metadata[anagroup][subkey])
			S0.sect.append(S1.sect)
			#print (anagroup,subkey)
			del S1
		S.sect.append(S0.sect)
	return S

def delete_by_key(dict_del, lst_keys):
	for k in lst_keys:
		try:
			del dict_del[k]
		except KeyError:
			pass
	for v in dict_del.values():
		if isinstance(v, dict):
			delete_by_key(v, lst_keys)

	return dict_del

def string_value(mydict):
	for key,val in mydict.items():
		if isinstance(val,dict):
			mydict[key] = string_value(val)
	if 'value' in mydict:
		mydict['value'] = str(mydict['value'])
	return mydict


def prettify_yml(savepath_yml):
	with open(savepath_yml, 'r') as tempfile: myfile = tempfile.read()
	myfile = myfile.replace('value: \'','value: ')
	myfile = myfile.replace('\'\n', '\n')
	ymlgroups = ['EdDetection','SpikeSorting','BurstClassification','Diagnostics']
	for anagroup in ymlgroups:
		myfile = myfile.replace('\n'+anagroup,'\n\n'+anagroup)
	with open(savepath_yml, 'w') as outfile: outfile.write(myfile)

def dump_yml(metadata,savepath_yml):
	import yaml
	class NoAliasDumper(yaml.Dumper):
		def ignore_aliases(self, data):
			return True

	with open(savepath_yml, 'w') as myfile:
		yaml.dump(metadata,myfile,Dumper=NoAliasDumper, default_flow_style=False,sort_keys=False)#default_flow_style=False, sort_keys=False, allow_unicode = True, encoding = None,

	prettify_yml(savepath_yml)

def calc_diagnostics(aRec,template_dict):

	aRec.loadify_bursts()
	hl_bursts = [B for B in aRec.bursts if B.cname == 'high-load']
	N_hl = len(hl_bursts)  # number of high-load bursts
	rate_hl = N_hl / aRec.durAnalyzed
	dur_hl = np.sum([B.dur for B in hl_bursts])

	# burstiness
	get_intervals_spikes = lambda eaperiod: np.diff(eaperiod.spiketimes)
	isis = hf.apply_to_artfree(aRec, get_intervals_spikes)
	B = lambda vec: (np.std(vec) - np.mean(vec)) / (np.std(vec) + np.mean(vec))
	bness = B(isis)

	dur_bursts = np.sum([B.dur for B in aRec.bursts])
	dur_free = np.sum([Free.dur for Free in aRec.freesnips])

	diagnostics_dict = {}
	diagnostics_dict['rate(spikes)'] = len(aRec.spiketimes) / aRec.durAnalyzed
	diagnostics_dict['rate(high-load)'] = rate_hl * 60.
	diagnostics_dict['tfrac(high-load)'] = dur_hl / aRec.durAnalyzed
	diagnostics_dict['tfrac(bursts)'] = dur_bursts / aRec.durAnalyzed
	diagnostics_dict['tfrac(free)'] = dur_free / aRec.durAnalyzed
	diagnostics_dict['burstiness'] = bness
	diagnostics_dict['durAnalyzed'] = aRec.durAnalyzed / 60.
	diagnostics_dict['durArtifacts'] = np.sum([Art.dur for Art in aRec.artifacts]) / 60. if len(aRec.artifacts) > 0 else 0

	metadict = template_dict['Diagnostics']
	for key,value in diagnostics_dict.items():
		if key in metadict:	metadict[key]['value'] = value
	return metadict

