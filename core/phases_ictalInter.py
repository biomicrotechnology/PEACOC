from __future__ import division


import logging
import numpy as np
import core.ea_management as eam
from core.helpers import mergeblocks

logger = logging.getLogger('root')


overlaps = lambda x,y: (x[0] <= y[1]) and (y[0] <= x[1])


def merge_bursts(startstop_array,maxint):
	'''startstop_array : nx2, n is number of bursts
	maxint is maximum interval bursts can be separated to be considered not belonging together'''
	logger.debug('Merging bursts')
	logger.debug('Shape startstop: %s'%str(startstop_array.shape))

	if np.size(startstop_array)<=2:
		logger.debug('size<=2: returning original bursts')
		return startstop_array
	else:
		starts = startstop_array[:,0]
		stops = startstop_array[:,1]
		intervals = starts[1:]-stops[:-1]

		startinds = np.r_[0,np.where(intervals>=maxint)[0]+1]
		stopinds = np.r_[np.where(intervals>=maxint)[0],len(stops)-1]

		newstarts = starts[startinds]
		newstops = stops[stopinds]
		return np.vstack([newstarts,newstops]).T


def getNextEvent(tpt,eventObjs):
	laterObjs = eam.filter_objs(eventObjs,[lambda x: x.start>tpt])
	myind = np.argsort([obj.start for obj in laterObjs])[0]
	return laterObjs[myind]


def getNeighborIdx(target,times,mode='before'):
	'''target is a single timepoint
	times is and array of timepoints
	modes can be before and after
	fn find the time in times, that is the last(first) time before(after) time'''

	if mode == 'before': return np.where(times<target)[0][-1] if (times<target).any() else np.nan
	if mode == 'after': return np.where(times>target)[0][0] if (times>target).any() else np.nan


def set_IP_nonIPs(recObj,ip_times):

	IPs = [eam.State('IP'+str(ii+1),'IP',start,stop,parentobj=recObj) for ii,[start,stop] in enumerate(ip_times) ]
	logger.debug('IPs as State objects')
	if np.size(ip_times)>0:
		IPfree_ints = mergeblocks([ip_times.T],output='free',t_offset=recObj.offset,\
												t_start=0,t_stop=recObj.stop).T
	else: IPfree_ints = np.array([[0,recObj.stop]])
	logger.debug('N intervals %d'%(IPfree_ints.shape[0]))

	nonIPs = [eam.State('X'+str(ii+1),'temp',start,stop,parentobj=recObj) for ii,[start,stop] in enumerate(IPfree_ints) ]
	logger.debug('non-IPs as temporary State objects')

	recObj.states = eam.timesort_objs(nonIPs+IPs)

def break_interstates(recObj,cpt_thresh,cpt_drift,depri_thresh):
	allstates = eam.filter_objs(recObj.states,[lambda x: x.state=='IP'])
	nonIPs = eam.filter_objs(recObj.states,[lambda x: x.state=='temp'])
	logger.debug('N inter-IP-states: %d'%(len(nonIPs)))
	for tempstate in nonIPs:
		substates = break_state(tempstate,cpt_thresh=cpt_thresh,cpt_drift=cpt_drift,depri_thresh=depri_thresh)
		allstates+=substates
	recObj.states = eam.timesort_objs(allstates)

def get_cpts_CUSUM(eventtimes,snip,cpt_thresh=1,drift=0,zscore=True):
	'''eventtimes: n x 2, at 0: starts at 1 stops'''


	Z = lambda data: (data-np.mean(data))/np.std(data)

	intervals = np.r_[eventtimes[0,0]-snip[0],eventtimes[1:,0]-eventtimes[:-1,1],snip[1]-eventtimes[-1,1]]
	eventtimes_start = np.r_[snip[0],eventtimes[:,0],snip[-1]]
	eventtimes_stop = np.r_[snip[0],eventtimes[:,1],snip[-1]][::-1]

	if zscore: intervals = Z(intervals)
	x = intervals[:]#forwards for pre-time
	x2 = intervals[::-1]#backwards for depression
	if zscore: x = Z(x)

	#ascending direction : pre
	ta, tai, tai, amp = detect_cusum(x, cpt_thresh, drift, ending=True,show=False,gn_only=True)
	allcpts = eventtimes_start[ta] if np.size(ta)>0 else np.array([])


	#descending: post
	ta2, tai, taf, amp = detect_cusum(x2, cpt_thresh, drift, ending=True,show=False,gn_only=True)
	allcpts_reverse = eventtimes_stop[ta2][::-1] if np.size(ta2)>0 else np.array([])
 

	return [allcpts,allcpts_reverse]


def break_state(tempstate,cpt_thresh=1.,cpt_drift=0.,depri_thresh=120.,**kwargs):
	'''
	tempstate is eam State object that will be broken down into substates
	depri_thresh is in seconds
	'''
	if hasattr(tempstate,'parent'):
		createstate = lambda mytype,mystart,mystop:  eam.State('tba',mytype,mystart,mystop,parentobj=tempstate.parent)
	else:createstate = lambda mytype,mystart,mystop:  eam.State('tba',mytype,mystart,mystop)


	if 'inter_clusts' in kwargs:
		mybursts = eam.filter_objs(tempstate.bursts,[lambda x: x.cname in kwargs['inter_cnames']])
	else: mybursts = tempstate.bursts[:]
	logger.debug('N bursts %d'%(len(mybursts)))

	#1)run cpt to get pre and post
	if not len(mybursts) == 0:
		bursttimes = np.vstack([burst.roi for burst in mybursts])


		cptspre,cptspost = get_cpts_CUSUM(bursttimes,[tempstate.start,tempstate.stop],cpt_thresh=cpt_thresh,drift=cpt_drift,zscore=True)
		logger.debug('N cptspre %d, N cptspost %d'%(np.size(cptspre),np.size(cptspost)))

		#2)get time to next event
		ttevent = mybursts[0].start-tempstate.start
		logger.debug('ttevent %1.2f s'%(ttevent))
	else:
		ttevent = float(tempstate.dur)
		logger.info('No bursts in interval')
		if ttevent>=depri_thresh: return [ createstate('depr',tempstate.start,tempstate.stop)]
		else: return [createstate('IIP',tempstate.start,tempstate.stop,)]

	#3)break up by creating sub-events

	#3a)depression
	if ttevent >= depri_thresh:
		post = createstate('depr',tempstate.start,tempstate.start+ttevent)
		tempstate.start = tempstate.start+ttevent
		logger.debug('has depression')

	#post-increase
	elif np.size(cptspost)>0:
		post = createstate('postUp',tempstate.start,cptspost[0])
		tempstate.start = cptspost[0]
		logger.debug('has postUp')
	#neither depression nor postUp
	else:
		post = createstate('postUp',tempstate.start,tempstate.start)
		logger.debug('post does not exist')
	#3b)pre-state
	if np.size(cptspre)>0:
		pre = createstate('preUp',cptspre[-1],tempstate.stop)
		tempstate.stop = cptspre[-1]
		logger.debug('has preUp')
	else:
		pre = createstate('preUp',tempstate.stop,tempstate.stop)
		logger.debug('pre does not exist')
	#4) fuse to substate list
	#make dense if pre and post overlap
	if post.stop>pre.start:
		logger.info('pre and post overlap --> whole intervals gets dense')
		substates =  [createstate('dense',post.start,pre.stop)]
	#else concatenate pre and post and insert the state in-between as inter-ictal period
	else:
		interstate = createstate('IIP',post.stop,pre.start)
		substates = [substate for substate in (post,interstate,pre) if not substate.dur==0]
	return substates


def insertPredatorStates(statelist,predator_times,predator_type='art',mindur=0.,output=True):
	'''
	predators can be for example artifacts: nx2 [start,stop]

	'''

	if hasattr(statelist[0],'parent'):
		createstate = lambda id,mytype,mystart,mystop:  eam.State(id,mytype,mystart,mystop,parentobj=statelist[0].parent)
	else:createstate = lambda id,mytype,mystart,mystop:  eam.State(id,mytype,mystart,mystop)


	statelist = eam.timesort_objs(statelist)
	#account for mindur of predators
	predator_times = predator_times[np.diff(predator_times).flatten()>mindur]


	for ii,[start,stop] in enumerate(predator_times):

		#intercept predators that happen before the beginnings of all times
		P_isValid = False if stop<statelist[0].start else True
		start = statelist[0].start if start<statelist[0].start else start

		P = createstate('tba',predator_type,start,stop)#P for predator

		#find out which states share time with the predator
		affected = [state for state in statelist if overlaps(state.roi,P.roi)]

		for vv,V in enumerate(affected):#V for victim, it is a state object

			if (P.start>V.start) and (P.stop<V.stop):
				logger.debug('Predator sits within state, splits in two')
				newstates = [createstate(V.id+'1',V.state,V.start,P.start),createstate(V.id+'2',V.state,P.stop,V.stop)]

			elif (P.start<=V.start) and (P.stop>=V.stop):
				logger.debug('Predator occupies completely, will be destroyed')
				newstates = []

			elif (P.start>V.start and P.stop>=V.stop):
				logger.debug('Predator occupies the end, Victim is shortened')
				newstates = [createstate(V.id+'1',V.state,V.start,P.start)]

			elif (P.start<=V.start and P.stop<V.stop):
				logger.debug('Predator occupies the beginning, Victim is shortened')
				newstates = [createstate(V.id+'2',V.state,P.stop,V.stop)]

			else: assert 0, 'Unforeseen predator conditions, inds:(%d,%d) \n Vroi: {%1.2f,%1.2f}, Proi:{%1.2f,%1.2f}'\
						%(ii,vv,V.start,V.stop,P.start,P.stop)

			statelist.remove(V)
			statelist += newstates
		if P_isValid:statelist += [P]
	statelist = eam.timesort_objs(statelist)
	if output: return statelist


def nibble_endstates(statelist,befbuff,endbuff,output=True):
	'''
	befbuff and endbuff are in seconds
	'''

	if hasattr(statelist[0],'parent'):
		createstate = lambda id,mytype,mystart,mystop: eam.State(id,mytype,mystart,mystop,parentobj=statelist[0].parent)
	else:createstate = lambda id,mytype,mystart,mystop: eam.State(id,mytype,mystart,mystop)


	logger.info('Nibbling off endstates')
	B = createstate('tba','void',statelist[0].start,statelist[0].start+befbuff)#beginning
	E = createstate('tba','void',statelist[-1].stop-endbuff,statelist[-1].stop)#beginning
	B_protected,B_attackable, B_removable = ['IP','preUp'],['IIP','art'],['postUp','depr','dense']
	E_protected,E_attackable, E_removable = ['IP','postUp','depr'],['IIP','art'],['preUp','dense']

	#shorten B and E according to whether they are overlapped by protected
	firstprot = eam.filter_objs(statelist,[lambda x: x.state in B_protected])[0]
	if firstprot.start<B.stop: B.stop = float(firstprot.start)
	lastprot = eam.filter_objs(statelist,[lambda x: x.state in E_protected])[-1]
	if lastprot.stop>E.start: E.start = float(lastprot.stop)


	for P,protected,attackable,removable in zip([B,E],[B_protected,E_protected],[B_attackable,E_attackable],[B_removable,E_removable]):
		#check which states are potentially affected by buffer start
		victims =  [state for state in statelist if overlaps(state.roi,P.roi)]
		for V in victims:
			if V.state in attackable:
				newstates = insertPredatorStates([V],np.array(P.roi)[None,:],predator_type='void',output=True)
				for nS in newstates:#preventing overlaps
					if nS.state=='void':
						 if nS.start<V.start:nS.start = float(V.start)
						 if nS.stop> V.stop: nS.stop = float(V.stop)
			elif V.state in protected:
				newstates = [V]
			elif V.state == 'void':
				newstart,newstop = np.min([V.start,P.start]),np.max([V.stop,P.stop])
				logger.debug('Merging void states [%1.2f,%1.2f]'%(newstart,newstop))
				newstates = [createstate('mergeVoid','void',newstart,newstop)]
			elif V.state in removable:
				newstates = [createstate('mergeVoid','void',V.start,V.stop)]
			else: logger.warning('Invalid state: %d'%(V.state))

			statelist.remove(V)
			statelist += newstates

	statelist = eam.timesort_objs(statelist)
	if output: return statelist

def clear_artifactNeighbours(statelist,set_to='void',IIP_mindur=120.):
	logger.info('Annihilating neighbours of artifacts!')
	removable = ['postUp','preUp','depr','dense','IIP']
	affected = eam.filter_objs(statelist,[lambda x:x.state in removable])
	for V in affected:
		my_neighbors = [neigh for neigh in eam.neighbour_objs(statelist,V) if not neigh is None]
		if np.array([state.state=='art' for state in my_neighbors]).any():

			if V.state == 'IIP' and V.dur<IIP_mindur: V.set_state(set_to)
			elif not V.state=='IIP':V.set_state(set_to)

def merge_statebursts(statelist):
	succ = np.array([state.state for state in statelist])
	toDelete = []
	toAdd = []
	for statetype in np.unique(succ):
		occurs = np.where(succ==statetype)[0]
		ocdiff = np.diff(occurs)
		if (ocdiff==1).any():#checking whether there are any bursts
			logger.debug('Merging %s'%(statetype))
			startinds = np.r_[occurs[0],occurs[np.where(ocdiff>1)[0]+1]]
			stopinds = np.r_[occurs[np.where(ocdiff>1)[0]],occurs[-1]]
			burstlen = stopinds-startinds
			for starti,stopi in zip(startinds,stopinds):
				if stopi-starti>0:
					logger.debug('Merge-remove %s'%( str(np.arange(starti,stopi+1))))
					toDelete += statelist[starti:stopi+1]
					toAdd += [merge_statetrain(statelist[starti:stopi+1])]
	#statelist = [S for S in statelist if S not in toDelete]
	for doomed in toDelete: statelist.remove(doomed)
	#statelist.remove(toDelete)
	statelist += toAdd
	statelist = eam.timesort_objs(statelist)

def merge_statetrain(slist,newtype=None):
	'''
	slist is list of states
	'''
	if hasattr(slist[0],'parent'):
		createstate = lambda id,mytype,mystart,mystop: eam.State(id,mytype,mystart,mystop,parentobj=slist[0].parent)
	else:createstate = lambda id,mytype,mystart,mystop: eam.State(id,mytype,mystart,mystop)

	slist2 = eam.timesort_objs(slist)
	newtype = newtype if newtype else slist2[0].state
	newstate = createstate('merged',newtype,slist2[0].start,slist2[-1].stop)
	if hasattr(slist2[0],'begins'):newstate.set_begins(slist2[0].begins)
	if hasattr(slist2[-1],'ends'):newstate.set_ends(slist2[-1].ends)
	return newstate

def set_state_integrity(statelist):
	'''
	if a state is preceded by a dragon-state, it will not have a defined beginning
	if a state is followed by a dragon-state, it will not have a defined end
	'''
	logger.debug('Setting state integrity')
	protected = ['void','art']
	dragons = ['void','art']
	for S in eam.filter_objs(statelist,[lambda x: x.state not in protected]):

		bef,aft = eam.neighbour_objs(statelist,S)
		if not bef or (bef.state in dragons): S.set_begins(False)
		else: S.set_begins(True)

		if S.state=='depr' and aft is None:S.set_ends(False)
		elif S.state=='depr' and aft.state=='void':S.set_ends(True)
		elif not aft or (aft.state in dragons):S.set_ends(False)
		else: S.set_ends(True)


def reject_cpts(statelist,minTotalIIP=15*60.):
	'''
	removes preUp and postUp if their spikerate is lower than that the average rate of spikes in IIP
	if there is too little IIP (minTotalIIP) preUp,postUp and IIP will be changed to void
	'''
	IIPdur = np.sum([iip.dur for iip in eam.filter_objs(statelist,[lambda x: x.state=='IIP'])])
	logger.info('(total IIP dur: %1.2fmin > threshold: %1.2fmin): %s'%(IIPdur/60.,minTotalIIP/60.,IIPdur>=minTotalIIP))

	if IIPdur<minTotalIIP:
		victims = ['preUp','postUp','IIP']
		logger.debug('Whole trace is invalid, setting %s to void'%(str(victims)))
		for state in eam.filter_objs(statelist,[lambda x:x.state in victims]):
			state.set_state('void')
			[delattr(state,myattr) for myattr in ['begins','ends','whole'] if hasattr(state,myattr)]
	else:
		logger.debug('Checking for invalid changepoints...')


		temp  = np.sum(np.array([[len(state.spiketimes),state.dur] for state in \
								 eam.filter_objs(statelist,[lambda x: x.state=='IIP'])]),axis=0)
		rateIIP = temp[0]/temp[1]
		logger.info('Rate of spikes in IIP is %1.2f /min'%(rateIIP*60.))

		for victim,neighidx in zip(['preUp','postUp'],[0,1]):
			affected =  eam.filter_objs(statelist,[lambda x: x.state==victim])
			toDelete = []
			toAdd = []
			for ss,S in enumerate(affected):
				if S.spikerate > rateIIP: pass
				else:
					logger.debug('Lowrate %s %1.2f < %1.2f /min   count: %d'%(victim,S.spikerate*60.,rateIIP*60.,ss))
					N =  eam.neighbour_objs(statelist,S)[neighidx]#idx 0 is the neighbor before, 1 is after
					logger.debug('... merge to Neighbour-state %s'%(N.state))
					#newtype = N.state if not N.state=='depr' else 'IIP'
					#M = merge_statetrain([S,N],newtype=N.state)
					#toDelete += [N,S]
					#toAdd += [M]
					S.state = 'IIP'
			'''for doomed in toDelete: 
				logger.debug('Deleting 1 %s'%(victim))
				statelist.remove(doomed)
			#statelist.remove(toDelete)
			statelist += toAdd'''
			statelist = eam.timesort_objs(statelist)
	return eam.timesort_objs(statelist)

def swallow_densities(statelist,fracMax=0.15,burstclass=['M']):
	'''
	looks for IIPs with high (fracMax) Timefrac of burstclass and stets
	these IIPs along with the neighbouring preUp and postUp states to
	state Mdense
	'''

	IIPs = eam.filter_objs(statelist,[lambda x: x.state=='IIP'])
	toDelete = []
	toAdd = []
	for ss,S in enumerate(IIPs):
		bursts = eam.filter_objs(S.bursts,[lambda x: x.cname in burstclass])
		burstfrac = np.sum([burst.dur for burst in bursts])/S.dur
		if burstfrac > fracMax:
			logger.debug('IIP:%d - Burstfrac %1.2f > fracMax %1.2f'%(ss,burstfrac,fracMax))

			Ns = eam.neighbour_objs(statelist,S)
			fuselist = [N for N in Ns if N and N.state in ['preUp','postUp']]
			toDelete += fuselist+[S]
			toAdd += [merge_statetrain(fuselist+[S],newtype='Mdense')]
				#statelist = [S for S in statelist if S not in toDelete]
	for doomed in toDelete: statelist.remove(doomed)
	statelist += toAdd
	statelist = eam.timesort_objs(statelist)

def states_to_dict(statelist):

	attrs = ['state','start','stop','begins','ends']
	statedict = {}
	for ss,state in enumerate(eam.timesort_objs(statelist)):
		ID = 'S'+str(ss+1).zfill(2)
		statedict[ID] = {attr:getattr(state,attr) for attr in attrs if hasattr(state,attr)}

	return statedict



def plot_classifiedTrace(recObj,pandur=43.*60.,fwidth=13.,showitems=['singlets','bursts','states'],**kwargs):
	from matplotlib.ticker import MultipleLocator
	from matplotlib.pyplot import subplots

	p_h = kwargs['p_h'] if 'p_h' in kwargs else 0.1
	hspacing = kwargs['hspacing'] if 'hspacing' in kwargs else 0.8
	t_h =0.4
	b_h = 0.8
	lr_marg = kwargs['lr_marg'] if 'lr_marg' in kwargs else 0.6
	legOn = kwargs['legendOn'] if 'legendOn' in kwargs else False

	npans = int(np.ceil((recObj.dur-recObj.offset)/pandur))
	fheight = npans*p_h+(npans-1)*hspacing + t_h + b_h

	f, axarr = subplots(npans,1, figsize=(fwidth,fheight),facecolor='w')
	f.subplots_adjust(left = lr_marg/fwidth,right=1.-lr_marg/fwidth,bottom = b_h/fheight,top=1.-t_h/fheight,\
						  hspace=hspacing)
	for pp in range(npans):
		start,stop = recObj.offset+pp*pandur,recObj.offset+(pp+1)*pandur

		xlabOn=True if pp == npans-1 else False
		ax = axarr[pp] if npans>1 else axarr
		if pp==0:recObj.plot(showitems,unit='min',counterOn=True,xlabOn=xlabOn,ax=ax,legendOn=legOn)
		else:recObj.plot(showitems,unit='min',counterOn=True,xlabOn=xlabOn,ax=ax,legendOn=False)
		ax.set_xlim([start/60.,stop/60.])

		## you could also plot by episode, but then you would loose the counting
		#episode = eam.EAPeriod(start,stop,parentobj=)
		#episode.plot(['bursts','states'],unit='min',counterOn=True,xlabOn=xlabOn,ax=axarr[pp])
		ax.xaxis.set_major_locator(MultipleLocator(10))
	return f




def detect_cusum(x, threshold=1, drift=0, ending=False, show=True, ax=None,gp_only=False,gn_only=False,verbose=False):
	"""Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.
	Parameters
	----------
	x : 1D array_like
		data.
	threshold : positive number, optional (default = 1)
		amplitude threshold for the change in the data.
	drift : positive number, optional (default = 0)
		drift term that prevents any change in the absence of change.
	ending : bool, optional (default = False)
		True (1) to estimate when the change ends; False (0) otherwise.
	show : bool, optional (default = True)
		True (1) plots data in matplotlib figure, False (0) don't plot.
	ax : a matplotlib.axes.Axes instance, optional (default = None).
	Returns
	-------
	ta : 1D array_like [indi, indf], int
		alarm time (index of when the change was detected).
	tai : 1D array_like, int
		index of when the change started.
	taf : 1D array_like, int
		index of when the change ended (if `ending` is True).
	amp : 1D array_like, float
		amplitude of changes (if `ending` is True).
	Notes
	-----
	Tuning of the CUSUM algorithm according to Gustafsson (2000)[1]_:
	Start with a very large `threshold`.
	Choose `drift` to one half of the expected change, or adjust `drift` such
	that `g` = 0 more than 50% of the time.
	Then set the `threshold` so the required number of false alarms (this can
	be done automatically) or delay for detection is obtained.
	If faster detection is sought, try to decrease `drift`.
	If fewer false alarms are wanted, try to increase `drift`.
	If there is a subset of the change times that does not make sense,
	try to increase `drift`.
	Note that by default repeated sequential changes, i.e., changes that have
	the same beginning (`tai`) are not deleted because the changes were
	detected by the alarm (`ta`) at different instants. This is how the
	classical CUSUM algorithm operates.
	If you want to delete the repeated sequential changes and keep only the
	beginning of the first sequential change, set the parameter `ending` to
	True. In this case, the index of the ending of the change (`taf`) and the
	amplitude of the change (or of the total amplitude for a repeated
	sequential change) are calculated and only the first change of the repeated
	sequential changes is kept. In this case, it is likely that `ta`, `tai`,
	and `taf` will have less values than when `ending` was set to False.
	See this IPython Notebook [2]_.


	Author: Marcos Duarte, https://github.com/demotu/BMC'
	Output modified by Katharina Heining
	References
	----------
	.. [1] Gustafsson (2000) Adaptive Filtering and Change Detection.
	.. [2] hhttp://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectCUSUM.ipynb
	Examples
	--------
	>>> from detect_cusum import detect_cusum
	>>> x = np.random.randn(300)/5
	>>> x[100:200] += np.arange(0, 4, 4/100)
	>>> ta, tai, taf, amp = detect_cusum(x, 2, .02, True, True)
	>>> x = np.random.randn(300)
	>>> x[100:200] += 6
	>>> detect_cusum(x, 4, 1.5, True, True)
	>>> x = 2*np.sin(2*np.pi*np.arange(0, 3, .01))
	>>> ta, tai, taf, amp = detect_cusum(x, 1, .05, True, True)
	"""

	x = np.atleast_1d(x).astype('float64')
	gp, gn = np.zeros(x.size), np.zeros(x.size)
	ta, tai, taf = np.array([[], [], []], dtype=int)
	tap, tan = 0, 0
	amp = np.array([])
	# Find changes (online form)
	for i in range(1, x.size):
		s = x[i] - x[i-1]
		gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
		gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
		if gp[i] < 0:
			gp[i], tap = 0, i
		if gn[i] < 0:
			gn[i], tan = 0, i
		if gp_only:cond = gp[i] > threshold
		elif gn_only: cond = gn[i] > threshold
		else: cond = gp[i] > threshold or gn[i] > threshold
		if gp[i] > threshold or gn[i] > threshold:  # change detected!
			if gp_only:
				if gp[i] > threshold:
					ta = np.append(ta, i)
					tai = np.append(tai, tap if gp[i] > threshold else tan)  # start

			if gn_only:
				if gn[i] > threshold:
					ta = np.append(ta, i)
					tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
			else:
				ta = np.append(ta, i)    # alarm index
				tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
			gp[i], gn[i] = 0, 0      # reset alarm
	# THE CLASSICAL CUSUM ALGORITHM ENDS HERE

	# Estimation of when the change ends (offline form)
	if tai.size and ending:
		_, tai2, _, _ = detect_cusum(x[::-1], threshold, drift, show=False)
		taf = x.size - tai2[::-1] - 1
		# Eliminate repeated changes, changes that have the same beginning
		tai, ind = np.unique(tai, return_index=True)
		ta = ta[ind]
		# taf = np.unique(taf, return_index=False)  # corect later
		if tai.size != taf.size:
			if tai.size < taf.size:
				taf = taf[[np.argmax(taf >= i) for i in ta]]
			else:
				ind = [np.argmax(i >= ta[::-1])-1 for i in taf]
				ta = ta[ind]
				tai = tai[ind]
		# Delete intercalated changes (the ending of the change is after
		# the beginning of the next change)
		ind = taf[:-1] - tai[1:] > 0
		if ind.any():
			ta = ta[~np.append(False, ind)]
			tai = tai[~np.append(False, ind)]
			taf = taf[~np.append(ind, False)]
		# Amplitude of changes
		amp = x[taf] - x[tai]

	if show:
		_plot(x, threshold, drift, ending, ax, ta, tai, taf, gp, gn)

	if verbose: return ta,tai,taf,amp,gp,gn
	else: return ta, tai, taf, amp