from __future__ import division
from __future__ import print_function

import logging
import logging.config
import yaml
import numpy as np
import os
import io
import sys
from glob import glob
from scipy.stats import scoreatpercentile as scorep
from matplotlib.pyplot import subplots,gcf
import matplotlib.patches as patches

from core.helpers import mergeblocks,string_decoder,saveto_hdf5,open_hdf5,read_burstdata
from core import artisfaction

logger = logging.getLogger(__name__)
logger.disabled = True

default_path = os.getcwd()

#loading configurations
# TODO overwrite this default display config path with the path stated in the ymldict when loading a recording.
#  Doing os. environ is really NOT nice. For the overwriting use Rec._get_default_config_path(self,basename='configDisplay.yml')

default_configpath = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace('core','config'),'configDisplay.yml')
this = sys.modules[__name__]
if 'displayConfig' in os.environ: configpath = os.environ['displayConfig']
else: configpath = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace('core','config'),'configDisplay.yml')
with io.open(configpath, 'r') as ymlfile: this.cfg = yaml.safe_load(ymlfile)
this.rank_names = {val:key for key,val in list(this.cfg['name_ranks'].items())}
this.hatchdict = this.cfg['hatchdict']

for key in list(this.hatchdict.keys()):
	if 'hatch' in this.hatchdict[key]:
		this.hatchdict[key]['hatch'] = ''.join([this.hatchdict[key]['hatch']]*this.cfg['nhatch'])

#small helper functions
cdict_translator = {'clustid':'rank','seizidx':'si'}
overlaps = lambda x,y: (x[0] <= y[1]) and (y[0] <= x[1])
overlaps2 = lambda x,y: (x[0] < y[1]) and (y[0] < x[1])






def prepare_burstlegend(eaperiod,loadleg=False):
	allranks =sorted([rank for rank in list(this.rank_names.keys()) if not rank==None])

	if loadleg:
		loaddict = this.cfg['load_style']
		legdict = {}
		for rank in allranks:
			clusname = this.rank_names[rank]
			if rank==0:
				nextrank=0
				legdict.update({nextrank: [loaddict[clusname]['cname'], loaddict[clusname]['color'], 'bold']})
			else:
				nextrank = np.int(np.max(list(legdict.keys())))+1
				if not legdict[nextrank-1][0] == loaddict[clusname]['cname']:
					legdict.update({nextrank: [loaddict[clusname]['cname'], loaddict[clusname]['color'], 'bold']})

	else:

		dcolors = eaperiod.som.colors if hasattr(eaperiod,'som') else eaperiod.parent.som.colors


		legdict = {-1:['LI1',this.cfg['si1_color'],'bold']}
		legdict.update({rank:[this.rank_names[rank],dcolors[rank],'bold'] for rank in allranks})
		if None in this.rank_names:legdict.update({len(allranks):['XS',this.cfg['xs_color'],'bold']})
	return legdict

def draw_legend(f,legdict,space=1/20.,y_level=0.99,x_start=0.01):

	r = f.canvas.get_renderer()

	for ii,key in enumerate(sorted(legdict.keys())[::-1]):
		mystr,mycol,fontw = legdict[key]
		if ii==0:xstart=x_start
		else: xstart =th.get_window_extent(renderer=r).xmax/f.get_window_extent().width+space
		th = f.text(xstart,y_level,mystr,color=mycol,va='top',ha='left',fontweight=fontw)

def draw_legend2(f,legdict,space=1/10.,x_level=0.01,y_start=0.99):

	r = f.canvas.get_renderer()

	for ii,key in enumerate(sorted(legdict.keys())[::-1]):
		mystr,mycol,fontw = legdict[key]
		if ii==0:ystart=y_start
		else: ystart =th.get_window_extent(renderer=r).ymax/f.get_window_extent().height-space
		th = f.text(x_level,ystart,mystr,color=mycol,va='top',ha='left',fontweight=fontw)

#check whether path exists, if not create recursively
def checkmakepaths(pathtofile):
	dpath_save = os.path.split(pathtofile)[0]#to get rid of file itself
	print(dpath_save)
	if not os.path.isdir(dpath_save):
		os.makedirs(dpath_save)


def lazy_property(fn):
	'''Decorator that makes a property lazy-evaluated.
	'''
	attr_name = '_lazy_' + fn.__name__

	@property
	def _lazy_property(self):

		if not hasattr(self, attr_name):
			#print 'Loading',self.id
			logger.info('Loading '+attr_name)
			setattr(self, attr_name, fn(self))
		return getattr(self, attr_name)
	return _lazy_property


def filter_objs(obj_list,checkfn_list):
	'''
	example:
		checkfn1 = lambda x: x.cname in ['XL','L']
		checkfn2 = lambda x: x.si<=0.8
		desired_list = eam.filter_objs(recObject.bursts,[checkfn1,checkfn2])
	'''
	filt_list = []
	for obj in obj_list:
		bools = np.array([checkfn(obj) for checkfn in checkfn_list])
		if bools.all():filt_list.append(obj)
	return filt_list


def timesort_objs(objlist,verbose=False):
	order = np.argsort([obj.start for obj in objlist])
	if verbose:  return list(np.array(objlist)[order]),order
	else: return list(np.array(objlist)[order])

def attrsort_objs(objlist,attr,verbose=False):
	order = np.argsort([getattr(obj,attr) for obj in objlist])
	if verbose:  return list(np.array(objlist)[order]),order
	else: return list(np.array(objlist)[order])


def neighbour_objs(objlist,myobj):
	idx = objlist.index(myobj)
	if idx == 0: bef = None
	else: bef = objlist[idx-1]
	if idx == len(objlist)-1: aft = None
	else: aft = objlist[idx+1]
	return [bef,aft]

class Period(object):
	'''Something that is extended in time (has a start, stop and duration)'''
	def __init__(self,start,stop):
		self.start = start
		self.stop = stop

	@property
	def dur(self):
		return self.stop-self.start

	@property
	def roi(self):
		return [self.start,self.stop]

class EAPeriod(Period):
	'''A period containing EA, it can have bursts and epiptiform discharges'''
	def __init__(self,start,stop,parentobj=None):
		self.start = start
		self.stop = stop
		self.type = 'EAPeriod'
		if parentobj: self.set_parentObj(parentobj)

	@property
	def id(self):
		if not hasattr(self,'_id'): self._id=None
		return self._id

	@id.setter
	def id(self,setid):
		self._id = setid


	def set_parentObj(self,PO):

		possible_attrs = ['_raw','_artifacts','_spiketimes','burststore','statestore'\
					 ,'som','sr','_sr']
		actual_attrs = [attr for attr in possible_attrs if hasattr(PO,attr)]
		#logger.info('Actual Attrs PO: %s'%(str(actual_attrs)))
		self.parent = PO
		for attr in actual_attrs:
			#logger.debug('Attr-set: %s'%(attr))
			val = getattr(PO,attr)
			setattr(self,attr,val)


	@property
	def raw_data(self):
		return self._raw[np.int(self.start*self.sr):np.int(self.stop*self.sr)]

	@property
	def isis(self):
		return np.diff(self.spiketimes)

	@property
	def artifacts(self):
		return [Period(art[0],art[1]) for art in self._artifacts if overlaps(self.roi,art)]

	@property
	def artifactTimes(self):
		artlist = [art.roi for art in self.artifacts]
		if np.size(artlist)>0: return np.vstack(artlist)
		else: return np.array([[],[]]).T

	@property
	def tdata(self):
		return np.linspace(self.start,self.stop,len(self.raw_data))

	@property
	def spiketimes(self):
		if hasattr(self,'_myspiketimes'): return self._myspiketimes
		else: return self._spiketimes[(self._spiketimes>=self.start) & (self._spiketimes<=self.stop)]

	@spiketimes.setter
	def spiketimes(self,times):
		self._myspiketimes = times

	@property
	def spikerate(self):
		if hasattr(self,'durAnalyzed'):
			return len(self.spiketimes)/self.durAnalyzed #in case you look at a Rec
		else:
			return len(self.spiketimes)/self.dur #for all other sub-events
	
	@property
	def singlets(self):
		smarg = this.cfg['singlet_marg']
		cond = np.where(np.diff(self.spiketimes)>=smarg)[0]
		spiketrain = self.spiketimes[[ii for ii in cond if ii in cond + 1]]
		# check whether first and last are also spikes
		if len(self.spiketimes) >= 2:
			if (self.spiketimes[1] - self.spiketimes[0]) >= smarg:
				spiketrain = np.r_[self.spiketimes[0], spiketrain]
			if (self.spiketimes[-1] - self.spiketimes[-2]) >= smarg:
				spiketrain = np.r_[spiketrain, self.spiketimes[-1]]
		return spiketrain


	@property
	def artfreeTimes(self):
		if np.size(self.artifactTimes)==0: return np.array([[self.start, self.stop]])

		return mergeblocks([self.artifactTimes.T],output='free',t_start=self.start,t_stop=self.stop).T

	@property
	def freetimes(self):
		#tstart is at minimum the start of the analysis window (offset)
		tstart = np.max([self.start,self.offset]) if hasattr(self,'offset') else np.float(self.start)
		free_margin = this.cfg['free_margin']
		freedist_min = this.cfg['freedist_min']
		freeT = np.array([[],[]])
		for cleansnip in self.artfreeTimes:
			snipspikes = self.spiketimes[(self.spiketimes>=cleansnip[0]) & (self.spiketimes<=cleansnip[1])]
			isis = np.diff(snipspikes)
			free_start = snipspikes[np.where(isis>freedist_min)[0]]+free_margin
			free_stop = snipspikes[np.where(isis>freedist_min)[0]+1]-free_margin
			freeTemp = np.vstack([free_start,free_stop])
			freeT = np.hstack([freeT,freeTemp])

		# including freetimes before the first (s1) and after the last spike (sn)
		s1,sn = np.min(self.spiketimes),np.max(self.spiketimes)
		freeT = np.hstack([np.vstack([tstart+free_margin,s1-free_margin]),freeT]) if s1-2*free_margin>tstart else freeT[:]
		freeT = np.hstack([freeT,np.vstack([sn+free_margin,self.stop-free_margin])]) if sn+2*free_margin<self.stop else freeT[:]
		return freeT.T


	@property
	def freesnips(self):
		return [Period(snip[0],snip[1]) for snip in self.freetimes]



	@property
	def bursts2(self):
		'''older version, delete if there are no problems with the new version'''
		if not hasattr(self,'_bursts'):
			logger.debug('Getting bursts from dict')
			bdict = self.burstdict
			rois = [key for key in  sorted(bdict.keys()) if not key=='params']
			allburstlist = [Burst(roi,*bdict[roi][0],parentobj=self) for roi in rois]

			params = bdict['params']
			for param in params[1:]:
				idx =params.index(param)
				attrname = cdict_translator[param] if param in cdict_translator else str(param)
				for B in allburstlist: setattr(B,attrname,bdict[B.id][idx])
			self._bursts = filter_objs(allburstlist,[lambda x: overlaps2(x.roi,self.roi)])

		return self._bursts

	@property
	def bursts(self):
		if not hasattr(self,'_bursts'):
			if not hasattr(self,'burststore'):
				if self.type == 'rec':self.get_burststore()
				else:
					self.parent.get_burststore()
					setattr(self,'burststore',self.parent.burststore)
			self._bursts = [burst for burst in self.burststore if overlaps2(burst.roi,self.roi)]

		return self._bursts

	@bursts.setter
	def bursts(self,burstlist):
		self._bursts = burstlist

	def get_tBurst(self,category):
		return np.sum([burst.dur for burst in self.bursts if burst.cname == category])

	def get_nCategory(self,category):
		if category == 'ES': return len(self.singlets)
		else:return np.sum([1 for burst in self.bursts if burst.cname == category])

	@property
	def states2(self):
		'''older version, delete if there are no problems with the new version'''
		if not hasattr(self,'_states'):
			logger.debug('Getting states from dict')
			rois = [key for key in  sorted(self.statedict.keys())]
			allstatelist = [State(roi,'void',0.,1.,parentobj=self) for roi in rois]
			for S in allstatelist:
				for attr,val in list(self.statedict[S.id].items()): setattr(S,attr,val)
			self._states = filter_objs(allstatelist,[lambda x: overlaps2(x.roi,self.roi)])
		return self._states

	@property
	def states(self):
		if not hasattr(self,'_states'):
			if not hasattr(self,'statestore'):
				if self.type == 'rec':self.get_statestore()
				else:
					self.parent.get_statestore()
					setattr(self,'statestore',self.parent.statestore)
			self._states = [state for state in self.statestore if overlaps2(state.roi,self.roi)]
		return self._states


	@states.setter
	def states(self,statelist):
		self._states = statelist


	def get_tStates(self,statetype):
		if type(statetype) == list:mystates = filter_objs(self.states,[lambda x: x.state in statetype])
		else: mystates = filter_objs(self.states,[lambda x: x.state == statetype])
		if len(mystates) == 0: return 0
		else: return np.sum([S.dur for S in mystates])

	@property
	def ll(self):
		return np.sum(np.abs(np.diff(self.raw_data)))

	def loadcolor_bursts(self):
		loaddict = this.cfg['load_style']
		for B in self.bursts:
			if B.cname in loaddict.keys(): B.color = loaddict[B.cname]['color']


	def loadname_bursts(self):
		loaddict = this.cfg['load_style']
		for B in self.bursts:
			if B.cname in loaddict.keys(): B.cname = loaddict[B.cname]['cname']

	def loadify_bursts(self):
		self.loadcolor_bursts()
		self.loadname_bursts()

	def plot(self,proplist = ['raw','artifacts','spikes','free','singlets','bursts','states'],\
			 ylab='',counterOn=False,showwhole=True,xlabOn=True, unit='s',legendOn=False,loadlegend=False,**kwargs):

		validprops = ['raw','artifacts','spikes','free','singlets','bursts','states']
		wrongprops = [prop for prop in proplist if not prop in validprops]
		if len(wrongprops)>0: logger.warning('Invalid props will not be plotted: %s'%(wrongprops))

		if unit=='s': ufac = 1.
		elif unit == 'min': ufac = 1./60.
		elif unit == 'h': ufac = 1/60./60.

		fs = kwargs['fs'] if 'fs' in kwargs else 12
		fs2 = kwargs['fs2'] if 'fs2' in kwargs else 9#for the markers and counts
		cumheight = scorep(self.raw_data,95) if 'raw' in proplist else 0
		fac = (self.raw_data.max()-self.raw_data.min())*0.3 if 'raw' in proplist else 1.

		ydict = {}
		for plotitem in [prop for prop in proplist if not prop=='raw']:
			myheight = this.cfg['relative_heights'][plotitem]*fac
			y0,y1 = np.float(cumheight),cumheight+myheight
			ydict[plotitem]  = [y0,y1]
			if not plotitem=='singlets': cumheight += y1

		cummin = np.min(self.raw_data) if 'raw' in proplist else np.min([val[0] for val in list(ydict.values())])
		cummax = np.max([val[1] for val in list(ydict.values())]) if not (len(proplist)==1 and proplist[0]=='raw') else np.max(self.raw_data)

		if 'ax' in kwargs: ax = kwargs['ax']
		else:
			f,ax = subplots(1,1,facecolor='w',figsize=(16,3))
			f.subplots_adjust(left=0.04,right=0.99,top=0.98,bottom=0.2)
		#ax.set_title(self.id)
		if 'raw' in proplist: ax.plot(self.tdata*ufac,self.raw_data,color='k',linewidth=0.5)

		if 'artifacts' in proplist:
			y0,yh = ydict['artifacts'][0],np.diff(ydict['artifacts'])[0]
			for art in self.artifacts:
				#ax.add_patch(patches.Rectangle((art.start*ufac,y0),art.dur*ufac,yh,color='khaki',alpha=0.4))
				ax.axvspan(art.start*ufac,art.stop*ufac,facecolor='khaki',alpha=0.4,edgecolor='khaki')
				#ax.hlines(self._yref*3.5,burst.start,burst.stop,color=burst.color,linewidth=3)
			if legendOn: gcf().text(0.99,0.99,'artifacts',fontweight='bold',color='khaki',fontsize=12,ha='right',va='top',bbox=dict(facecolor='w', edgecolor='none'))
		if 'spikes' in proplist: ax.vlines(self.spiketimes*ufac,ydict['spikes'][0],ydict['spikes'][1],color='r',linewidth=0.5)
		if 'free' in proplist:
			y0,yh = ydict['free'][0],np.diff(ydict['free'])[0]
			for free in self.freesnips:
				ax.add_patch(patches.Rectangle((free.start*ufac,y0),free.dur*ufac,yh,color='b'))


		if 'bursts' in proplist:
			y0,yh = ydict['bursts'][0],np.diff(ydict['bursts'])[0]
			if self.type== 'burst':
				ax.add_patch(patches.Rectangle((self.start*ufac,y0),self.dur*ufac,yh,color=self.color,linewidth=0))
			else:
				for burst in self.bursts:
					ax.add_patch(patches.Rectangle((burst.start*ufac,y0),burst.dur*ufac,yh,color=burst.color,linewidth=0))
					#ax.hlines(self._yref*3.5,burst.start,burst.stop,color=burst.color,linewidth=3)

			if legendOn: legdict = prepare_burstlegend(self,loadleg=loadlegend)

		if 'singlets' in proplist:
			y0,y1 = ydict['singlets']
			ax.vlines(self.singlets*ufac,y0,y1,color='k',linewidth=0.5,zorder=30)
			if 'legdict' in locals():
				legdict.update({max(legdict.keys())+1:['so.s.','k','normal']})
			else: legdict = {1:['so.s.: |','k','normal']}

		if 'states' in proplist:
			y0,yh = ydict['states'][0],np.diff(ydict['states'])[0]
			yc,yc2 = y0+0.95*yh,y0+0.05*yh #for plotting markers and counters
			for ii,state in enumerate(timesort_objs(self.states)):
				ax.add_patch(patches.Rectangle((state.start*ufac,y0),state.dur*ufac,yh,**this.hatchdict[state.state]))
				xc = (state.start+0.5*state.dur)*ufac

				if counterOn:
					ax.annotate(str(ii),xy=(xc,yc),xytext=(xc,yc),color='r',ha='center',va='top',fontsize=fs2)

				if hasattr(state,'whole') and showwhole:
					marker = 'o' if state.whole else '>'
					ax.annotate(marker,xy=(xc,yc2),xytext=(xc,yc2),color='r',ha='center',va='bottom',fontsize=fs2)
		ax.set_ylim([cummin,cummax])
		ax.set_xlim([self.start*ufac,self.stop*ufac])
		if xlabOn: ax.set_xlabel('Time [%s]'%(unit),fontsize=fs,fontweight='normal')
		#if legendOn: draw_legend(gcf(),legdict,space=1/40.)
		if legendOn: draw_legend2(gcf(),legdict,space=1/10.)
		ax.set_ylabel(ylab,fontsize=fs,fontweight='normal')
		ax.set_yticks(np.arange(-4,4.1,2))



class Rec(EAPeriod):

	def __init__(self,id='',animalObj=None,logger_on=True,**kwargs):
		self.id = id
		self.type = 'rec'
		self.start = 0.

		#composition
		self.logger_on = logger_on
		if animalObj: self.animal = animalObj
		if 'sompath' in kwargs: self.set_som(kwargs['sompath'])
		if 'init_ymlpath' in kwargs:
			self.init_ymlpath = kwargs['init_ymlpath']
			with io.open(kwargs['init_ymlpath'], 'r') as ymlfile: ymldict = yaml.safe_load(ymlfile)
			self.read_from_dict(ymldict)
		if 'init_datapath' in kwargs:
			#this is only ment for a direct fast readout, not for analysis
			#print('Warning: accessing via resultsfile is only for readout, not analysis runthrough!')
			self.read_from_data(kwargs['init_datapath'])


	def _get_default_config_path(self,basename='configAnalysis.yml'):
		dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		return os.path.join(dirname,'config',basename)

	def read_from_data(self,datapath,raw_from_datapath=False,raw_ext='__raw500.h5'):
		if not len(self.id)=='': self.id = os.path.basename(datapath).split('__')[0]
		self._resultsfileH = datapath
		self.fullpath = os.path.dirname(datapath)
		self.raw_ext=raw_ext

		if 'analysisConfig' in os.environ:self.configpath_ana = os.environ['analysisConfig']
		else:self.configpath_ana = self._get_default_config_path(basename='configAnalysis.yml')
		with io.open(self.configpath_ana, 'r') as ymlfile: self.cfg_ana = yaml.safe_load(ymlfile)        #for [extkey,extval] in self.cfg_ana['path_ext'].items(): setattr(self,extkey,extval)

		try:
			if raw_from_datapath: self._rawfileH = open_hdf5(self.resultsfileH,'/'+self.cfg_ana['Preprocessing']['groupkey']+'/path')#looks up address of rawfile in hdf5
			else: self._rawfileH = self.rawfileH
		except:
			self._rawfileH = 'NA'
			logger.warning('raw-data file path is not listed in %s'%(self._resultsfileH))

		#self.artifactfile = self.get_filepath(self.artifact_ext)
		#if not os.path.isfile(self.artifactfile): logger.warning('yml-specified artifact file %s does not exist yet.'%(self.artifactfile))
		#setting polarity
		self._polarity = open_hdf5(self.resultsfileH,'/'+self.cfg_ana['EdDetection']['groupkey']+'/data/polarity')





	def read_from_dict(self,ymldict):

		paths = ymldict['data_paths']
		settings = ymldict['settings']
		subdict = ymldict['analysis']

		self.id = settings['data_id']
		self.fullpath = subdict['savepaths']['data_save']
		if 'fig_save' in subdict['savepaths']: self.figpath = subdict['savepaths']['fig_save']
		self.create_paths()
		# retreive the input data

		if not settings['logging_enabled'] or not self.logger_on:logger.disabled = True
		else: self.logger = self.get_logger()

		try:
			self.configpath_ana = ymldict['config_files']['analysis']
		except:
			self.configpath_ana = self._get_default_config_path(self,basename='configAnalysis.yml')
			logger.warning('Guessing analysis config path, could not be retrieved from ymldict')

		with io.open(self.configpath_ana, 'r') as ymlfile: self.cfg_ana = yaml.safe_load(ymlfile)
		self.sr = self.cfg_ana['Preprocessing']['sr']
		for [extkey,extval] in self.cfg_ana['path_ext'].items(): setattr(self,extkey,extval)

		self._rawfileH = self.rawfileH if paths['raw_array'] == 'default' else paths['raw_array']
		if not os.path.isfile(self._rawfileH):
			logger.warning('yml-specified raw-data file %s does not exist yet. I hope you are running extraction protocol.'%(self._rawfileH))

		self.artifactfile = self.get_filepath(self.artifact_ext) if paths['artifacts'] == 'default' else paths['artifacts']
		if not os.path.isfile(self.artifactfile): logger.warning('yml-specified artifact file %s does not exist yet.'%(self.artifactfile))
		#setting polarity

		self._polaritypath = self.get_filepath(self.polarity_ext) if paths['polarity'] == 'default' else paths['polarity']
		if os.path.isfile(self._polaritypath): self.retrieve_polarity(self._polaritypath)
		else: logger.warning('yml-specified polarity file %s does not exist yet.'%(self._polaritypath))


		if 'som' in paths:
			if os.path.isfile(paths['som']):  self.set_som(paths['som'])
			else: logger.warning('yml-specified SOM-path %s does not exist'%(paths['som']))



	@lazy_property
	def odmlpath(self):
		fullpath = str(self.fullpath)
		odmlfile = None
		for ii in np.arange((fullpath.count(os.sep)),0,-1)+1:#loop through path to find odml-file
			newpath = os.sep+os.path.join(*(fullpath.split(os.path.sep)[:ii]))
			odmlfiles =  glob(os.path.join(newpath,'*.odml*'))
			if len(odmlfiles)>0:
				odmlfile = odmlfiles[0]
				break
		return odmlfile



	def get_filepath(self,ext):
		return os.path.join(self.fullpath,self.id+ext)

	@property
	def resultsfileH(self):
		if not hasattr(self,'_resultsfileH'):
			self._resultsfileH = self.get_filepath(self.results_ext)
		return self._resultsfileH

	@resultsfileH.setter
	def resultsfileH(self,resultsfile):
		self._resultsfileH = resultsfile


	@property
	def rawfileH(self):
		if not hasattr(self,'_rawfileH'):
			self._rawfileH = self.get_filepath(self.raw_ext)
		return self._rawfileH

	@rawfileH.setter
	def rawfileH(self,rawfile):
		self._rawfileH = rawfile



	def _save_byGroup(self,data,groupkey,mergewithexisting=True,overwrite_groups=True,
					   overwrite_file=False):

		saveto_hdf5({groupkey:data}, self.resultsfileH,mergewithexisting=mergewithexisting, overwrite_groups=overwrite_groups,
					   overwrite_file=overwrite_file)


	def get_logger(self,**kwargs):

		if 'to_file' in kwargs: logpath = kwargs['to_file']
		else: logpath = os.path.join(self.fullpath,self.id+'__analysis.log')
		msg2 = 'Appended logger at %s'%(logpath) if os.path.isfile(logpath) else 'Created logger at %s'%(logpath)

		lconfigpath = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace('core','config'),'configLogging.yml')

		if os.path.exists(lconfigpath):
			msg1 = 'Loading logconfig from %s'%lconfigpath
			with open(lconfigpath, 'rt') as yamlfile:
				ydict = yaml.safe_load(yamlfile.read())
			ydict['handlers']['file']['filename'] = logpath
			logging.config.dictConfig(ydict)
		else:
			msg1 = 'Using basic log-config.'
			logging.basicConfig(level='DEBUG')

		logger = logging.getLogger()
		logger.info(msg1)
		logger.info(msg2)
		return logger


	def create_paths(self,path_attrs=['fullpath','figpath']):

		for pathattr in path_attrs:
			path = getattr(self,pathattr)
			if not os.path.isdir(path):
				#logger.debug('Creating path %s'%(path))
				os.makedirs(path)
			else: pass#logger.debug('Path already exists %s'%path())

	def strippath(self,path):
		return path.replace(self.fullpath+os.sep,'')

	@property
	def polarity(self):
		if not hasattr(self,'_polarity'):
			tf = open(self._polaritypath,'r')
			pol = tf.readline()
			tf.close()
			self._polarity = pol.strip()
		return self._polarity

	def retrieve_polarity(self,polpath):
		tf = open(polpath,'r')
		pol = tf.readline()
		tf.close()
		self._polarity = pol.strip()

	@polarity.setter
	def polarity(self,polarity):
		valid_polarities = ['neg','pos','mix']
		if not polarity.strip() in valid_polarities:
			logger.error('Polarity %s is not valid, choose form %s'%(polarity,str(valid_polarities)))
		else:
			logger.debug('Setting polarity: %s'%(polarity))
			self._polarity = polarity.strip()


	@property
	def _quality_range(self):
		return self.cfg_ana['quality_range']

	@property
	def quality(self):
		if not hasattr(self,'_quality'):
			tf = open(self.get_filepath(self.quality_ext),'r')
			quality = tf.readline()
			tf.close()
			self._quality = quality.strip()
		return self._quality

	@quality.setter
	def quality(self,quality):
		if not quality.strip() in self._quality_range:
			logger.error('Quality %s is not valid, choose form %s'%(quality,str(self._quality_range)))
		else:
			logger.debug('Setting quality: %s'%(quality.strip()))
			self._quality = quality.strip()


	@property
	def genpath(self):
		if not hasattr(self,'_genpath'):
			self._genpath = self.id
		return self._genpath

	@genpath.setter
	def genpath(self,genpath):
		logger.debug('Setting genpath: %s'%(genpath))
		self._genpath =genpath


	@property
	def fullpath(self):
		if not hasattr(self,'_fullpath'):
			self._fullpath = os.path.join(default_path,self.genpath)
		return self._fullpath

	@fullpath.setter
	def fullpath(self,fullpath):
		logger.debug('Setting fullpath: %s'%(fullpath))
		self._fullpath =fullpath

	@property
	def figpath(self):
		if not hasattr(self,'_figpath'):
			self._figpath = os.path.join(default_path,self.genpath)
		return self._figpath

	@figpath.setter
	def figpath(self,figpath):
		logger.debug('Setting figpath: %s'%(figpath))
		self._figpath =figpath

	@property
	def sr(self):
		if not hasattr(self,'_sr'):
			self._sr = open_hdf5(self.rawfileH, '/data/sr')
		return self._sr

	@sr.setter
	def sr(self,sampling_rate):self._sr = sampling_rate

	@property
	def _raw(self):
		if not hasattr(self,'_rawtrace'):
			try:
				self._rawtrace = open_hdf5(self.rawfileH, '/data/trace')
			except:
				self._rawtrace = open_hdf5(self.rawfileH, '/data')
		return self._rawtrace

	@_raw.setter
	def _raw(self,rawtrace):
		self._rawtrace = rawtrace

	def get_groupkey(self,anaclass):
		return '/'+self.cfg_ana[anaclass]['groupkey']

	@lazy_property
	def _EDdict(self):
		return open_hdf5(self.resultsfileH,self.get_groupkey('SpikeSorting'))

	@lazy_property
	def _EDdict0(self):
		return open_hdf5(self.resultsfileH,self.get_groupkey('EdDetection'))


	@property
	def _spiketimes(self):
		if not hasattr(self,'_spiketimesTemp'):
			self._spiketimesTemp = open_hdf5(self.resultsfileH,self.get_groupkey('SpikeSorting')+'/data/spikes')
		return self._spiketimesTemp

	@_spiketimes.setter
	def _spiketimes(self,spiketimes):
		self._spiketimesTemp = spiketimes


	@lazy_property
	def spikes0(self):
		return open_hdf5(self.resultsfileH,self.get_groupkey('EdDetection')+'/data/spikes')

	@property
	def stop(self):
		if not hasattr(self,'_stop'):
			try: self._stop = open_hdf5(self.resultsfileH,self.get_groupkey('EdDetection')+'/data/t_total')
			except: self._stop = len(self._raw)/self.sr
		return self._stop

	@stop.setter
	def stop(self,stoptime):
		self._stop = stoptime

	@property
	def offset(self):
		if not hasattr(self,'_offset'):
			self._offset = open_hdf5(self.resultsfileH,self.get_groupkey('EdDetection')+'/data/t_offset')
		return self._offset

	@property
	def dur0(self):
		return self.dur-self.offset

	@property
	def durAnalyzed(self):
		''' = total duration - offset - artifacts outside of offset'''
		if np.size(self.artifactTimes)>0: artdur =  np.sum(np.diff(np.clip(self.artifactTimes,self.offset,self.artifactTimes.max())))
		else: artdur = 0
		return self.dur0-artdur

	@property
	def _artifacts(self):
		if not hasattr(self,'_my_artifacts'):
			try:
				self._my_artifacts = open_hdf5(self.resultsfileH,self.get_groupkey('EdDetection')+'/data/mask_startStop_sec').T
			except:
				self.retrieve_artifacts_txt(self.artifactfile)
		return self._my_artifacts

	@property
	def artifactfile(self):
		#necessary because artifact file might come from elsewhere
		if not hasattr(self,'_artifactfile'):
			self._artifactfile = self.get_filepath(self.artifact_ext)
		return self._artifactfile

	@artifactfile.setter
	def artifactfile(self,txtpath):self._artifactfile = txtpath #useful when you dont have a standard path for the artifacts

	def retrieve_artifacts_txt(self,txtpath):
		artdict = artisfaction.readArtdict_txt(txtpath)
		cfg = self.cfg_ana['EdDetection']
		mindist = cfg['mindist_arts']
		marg_arts = cfg['marg_arts']
		marg_sats = cfg['marg_sats']
		arts = artisfaction.fuse_artifacts([artdict['arts'], artdict['sats']],[marg_arts,marg_sats],mindist=mindist).T
		self._my_artifacts = arts

	def set_som(self,path,id='thisSOM'):
		self.som = SOM(path,id=id)

	@property
	def burstdict(self):
		if not hasattr(self,'_burstdict'):
			try:
				burstpath = self.get_groupkey('BurstClassification')+'/data'
				burstdata = open_hdf5(self.resultsfileH,burstpath)
				self._burstdict = read_burstdata(burstdata['values'],burstdata['params'])
				del burstdata
				logger.info('Reading burstclasses')
			except:
				self._burstdict = {}
				logger.warning('Opening empty burstclasses')
		return self._burstdict

	@burstdict.setter
	def burstdict(self,mydict):
		self._burstdict = mydict

	def get_burststore(self):
		if not hasattr(self,'burststore'):
		   logger.debug('Getting bursts from dict')
		   rois = sorted([key for key in self.burstdict.keys() if not key=='params'])
		   allburstlist = [Burst(roi,*self.burstdict[roi][0],parentobj=self) for roi in rois]
		   params = self.burstdict['params']
		   for param in params[1:]:
			   idx =params.index(param)
			   attrname = cdict_translator[param] if param in cdict_translator else str(param)
			   for B in allburstlist: setattr(B,attrname,self.burstdict[B.id][idx])
		   self.burststore = allburstlist
		   self._bursts = allburstlist
		#return self._bursts

	@property
	def statedict(self):
		if not hasattr(self,'_statedict'):
			try:
				self._statedict = open_hdf5(self.resultsfileH, self.get_groupkey('StateAnalysis') + '/data')
				logger.info('Reading states')
			except:
				self._statedict = {}
				logger.warning('Opening empty states')
		return self._statedict

	@statedict.setter
	def statedict(self,mydict):
		self._statedict = mydict

	def get_statestore(self):
		if not hasattr(self,'statestore'):
			logger.debug('Getting states from dict')
			rois = [key for key in  sorted(self.statedict.keys())]
			allstatelist = [State(roi,'void',0.,1.,parentobj=self) for roi in rois]
			for S in allstatelist:
				for attr,val in list(self.statedict[S.id].items()): setattr(S,attr,val)
			self.statestore = allstatelist
			self._states = allstatelist
		#return self._states

	@statedict.setter
	def statedict(self,mydict):
		self._statedict = mydict


	@lazy_property
	def _burstfeaturedict(self):

		return open_hdf5(self.resultsfileH, self.get_groupkey('BurstDetection') + '/data')

	def check_coverage(self):
		dur_states = np.sum([state.dur for state in self.states])
		equalLen = np.isclose(dur_states,(self.dur-self.offset))
		if not equalLen: logger.warning('Mismatched durations cum(States) (%1.2f s) vs Rec (%1.2f s)'\
											%(dur_states,self.dur-self.offset))

		statemat = np.vstack([state.roi for state in timesort_objs(self.states)])
		isContinuum = np.allclose(statemat[:-1,1],statemat[1:,0])
		if not isContinuum:logger.warning('Recording not continuously covered: Gaps or Overlaps.')
		return isContinuum and equalLen


	@property
	def cats_available(self):
		if not hasattr(self,'_cats_available'):
			self._cats_available = np.unique([burst.cname for burst in self.bursts])
		return self._cats_available

	@property
	def diagn_tfracs(self):
		if not hasattr(self,'_diagn_tfracs'):
			self._diagn_tfracs = {category:self.get_tBurst(category)/self.durAnalyzed for category in self.cats_available}
		return self._diagn_tfracs

	@property
	def diagn_rates(self):
		if not hasattr(self,'_diagn_rates'):
			self._diagn_rates = {category:self.get_nCategory(category)/self.durAnalyzed for category in np.append(self.cats_available,'ES')}
			self._diagn_rates.update({'discharges':len(self.spiketimes)/self.durAnalyzed})
		return self._diagn_rates


class State(EAPeriod):
	def __init__(self,id,state,start,stop,parentobj=None):
		if parentobj: self.set_parentObj(parentobj)
		self.id = id
		self.state = state
		self.start = start
		self.stop = stop

		self.type = 'state'
		assert state in ['free','IP','IIP','depr','postUp','preUp','temp','dense','Mdense','void','art'],\
							 'Unknown state: %s' % state

	def set_begins(self,mybool):
		self.begins = mybool

	def set_ends(self,mybool):
		self.ends = mybool

	@property
	def whole(self):
		return (self.begins and self.ends)

	@property
	def proxdist(self):
		if not self.state in ['preUp','postUp']: return None

		htime = 0.5*self.dur
		n1 = len(self.spiketimes[self.spiketimes<(self.start+htime)])
		n2 = len(self.spiketimes[self.spiketimes>(self.stop-htime)])
		if self.state == 'preUp': prox,dist = n2,n1
		elif self.state == 'postUp':prox,dist = n1,n2
		if dist==0: return np.inf
		else : return prox/np.float(dist)


	def set_state(self,state):
		assert state in ['free','IP','IIP','depr','postUp','preUp','temp','dense','Mdense','void','art','IIPshort'],\
							 'Unknown state: %s' % state
		self.state = state


	@lazy_property
	def SW(self):
		return filter_objs(self.bursts,[lambda x: x.cname in this.cfg['severe']])

	@lazy_property
	def nSW(self):
		return len(self.SW)


class Burst(EAPeriod):
	def __init__(self,id,start,stop,parentobj=None):

		if parentobj: self.set_parentObj(parentobj)
		self.id = id
		self.start = start
		self.stop = stop
		self.type = 'burst'


	@property
	def isi(self):
		return np.diff(self.spiketimes)

	@property
	def color(self):
		if not hasattr(self,'_color'):

			if self.cname == 'XS':col =  this.cfg['xs_color']
			elif self.si == 1: col = this.cfg['si1_color']
			else: col  = self.som.colors[np.int(self.rank)]
			#if  col == 'darkgrey': col = '#001e90'
			self._color = col
		return self._color

	@color.setter
	def color(self,mycolor):
	   self._color = mycolor

	@property
	def cname(self):
		if not hasattr(self,'_cname'):
			self._cname = this.rank_names[self.rank]
		return self._cname

	@cname.setter
	def cname(self,mycname):
		self._cname = mycname

	@property
	def rank(self):
		if not hasattr(self,'_rank'): self._rank = None
		return self._rank

	@rank.setter
	def rank(self,thisrank):
		self._rank = thisrank

	@property
	def bmu(self):
		if not hasattr(self,'_bmu'): self._bmu = None
		return self._bmu

	@bmu.setter
	def bmu(self,thisbmu):
		self._bmu = thisbmu


	@property
	def si(self):
		if not hasattr(self,'_si'): self._si = None
		return self._si

	@si.setter
	def si(self,thissi):
		self._si = thissi

	def setseiz(self,boolseiz,boolseizG):
		self.is_seiz = boolseiz
		self.is_seizG = boolseizG





class SOM(object):
	def __init__(self,path,id='ASom'):
		self.id = id
		self.path = path


	@lazy_property
	def _mapdict(self):
		return open_hdf5(self.path)

	@lazy_property
	def colors(self):
		return string_decoder(self._mapdict['dcolors'])

	@lazy_property
	def nclust(self):
		return len(np.unique(self._mapdict['clusterids']))





