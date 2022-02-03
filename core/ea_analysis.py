from __future__ import division
from __future__ import print_function

import numpy as np
import os
import logging
from matplotlib.pyplot import close
from matplotlib.pyplot import subplots

import core.ea_management as eam
import core.phases_ictalInter as pii
import core.somify as somify
import core.helpers as hf
from core.ed_detection import Analysis


logger = logging.getLogger(__name__)
#print(logger)
#logger.info('Hello')
logger.disabled = True



class BurstDetection(Analysis):
    def __init__(self):
        for key,val in list(self.defaults.items()):
            setattr(self,key,val)

        self.method = 'Defining bursts as discharges being closely together.'
        
    
    @property
    def isi_bins(self):
        return np.logspace(np.log10(self.binborders[0]),np.log10(self.binborders[1]),self.nbins)

    def LL(self,trace):
        '''calculates linelength'''
        return np.sum(np.abs(np.diff(trace)))


    @property
    def freefused(self):
        if not hasattr(self,'_freefused'):
            freepts = (self.recObj.freetimes * self.recObj.sr).astype(np.int)
            self._freefused = np.hstack([self.recObj.raw_data[start:stop] for start, stop in freepts])
        return self._freefused


    def ztify(self,datavals):

        freestd = np.std(self.freefused)
        freemean = np.mean(self.freefused)
        return (datavals - freemean)/ freestd  # to z-score amplitudes based on periods of peace


    @property
    def featfuncs(self):
        return {'b_n':lambda X:len(X.spiketimes),\
                'isi_std': lambda X: np.std(X.isis),\
            'isi_mean':lambda X: np.mean(X.isis),\
            'isi_med':lambda X:np.median(X.isis),\
            'isi_fano':lambda X: np.std(X.isis)**2/np.mean(X.isis),\
            'isi_cv': lambda X: np.std(X.isis)/np.mean(X.isis),\
            'isi_peak':lambda X:self.isi_bins[np.argmax(np.histogram(X.isis,self.isi_bins)[0])],\
            'l_len':lambda X: self.LL(X.raw_data),\
            'l_rate':lambda X: self.LL(X.raw_data)/X.dur,\
            'lz_len':lambda X: self.LL(self.ztify(X.raw_data)),\
            'lz_rate':lambda X: self.LL(self.ztify(X.raw_data))/X.dur}
    
    @property
    def featuresAllowed(self):
        return list(self.featfuncs.keys())
    
    @property
    def features(self):
        if not hasattr(self,'_features'): return self.featuresAllowed
        else: return self._features
    
    @features.setter
    def features(self,featurelist):
        logger.debug('Setting features: %s'%(str(featurelist)))
        nonfeats = [feat for feat in featurelist if not feat in self.featuresAllowed]#check whether features are known (have functions to calculate)
        assert len(nonfeats)==0, 'Unknown features:%s ,please choose from %s'%(str(nonfeats),str(self.featureAllowed))
        self._features =featurelist
    

    
    def run(self,recObj):
        self.recObj = recObj
        burstdict = hf.find_bursts(recObj.spiketimes,self.maxdist)
        bursttimes0 = np.vstack([burstdict['start'],burstdict['stop']]).T
        logger.info('N bursts detected: %d'%(bursttimes0.shape[0]))
        bursttimes = pii.merge_bursts(bursttimes0,self.mergelim)
        logger.info('Merging bursts closer than %1.2f s, N bursts: %d'%(self.mergelim,bursttimes.shape[0]))
        
        
        burstObjs = [eam.Burst(ii+1,start,stop,parentobj=recObj) for ii,[start,stop] in enumerate(bursttimes)]
        #recObj.set_bursts(burstObjs)
        recObj._bursts = burstObjs
        

        for bobj in  recObj.bursts:
            for feature in self.features:
                val = self.featfuncs[feature](bobj)
                setattr(bobj,feature,val)
        if not 'isi_peak' in self.features:
            self.setparam('binborders','NA')
            self.setparam('bins','NA')

        if self.save_data:
           logger.info('Saving burstfeatures to dictionary')
           self.datadict = {}
           for bb,burst in enumerate(recObj.bursts):
                 subdict = {key:getattr(burst,key) for key in self.features+['start','stop']}
                 self.datadict['B'+str(bb+1)] = subdict
                 
           self._save_resultsdict(recObj)
           
        
        
class BurstClassification(Analysis):
    
    def __init__(self):
        for key,val in list(self.defaults.items()):
            setattr(self,key,val)                    
        for key,val in list(self.plot_dict.items()):
            setattr(self,key,val)
        self.method = 'Classifying bursts on a given SOM.'
        
   
    @property
    def plot_dict(self):
        return {'maxdur': 6.*60.,\
                'xdur': 60.,\
                'margint': [2.,2.],\
                'pandur2': 20.*60.,\
                'fwidth2': 17.,\
                'panheight2': 0.5,\
                'figformat': '.png',\
                'mydpi': 100}
    
    
    def plotOnMap(self,recObj,burstObjs):
        
        logger.debug('Sorting mapbursts according to Seizure-Index.')
        burstObjs = eam.attrsort_objs(burstObjs,'si')
        burstids = np.array([burst.id for burst in burstObjs])
        
        
        nbursts = len(burstObjs)

        logger.debug('Selecting bursts to get the greatest variety in SeizureIndex')
        adders = np.unique([np.int((nbursts-1)//(2**ii)) for ii in range(20)])[::-1][:-1]
        order_pool = np.array([0])
        for adder in adders:
            order_pool = np.r_[order_pool,order_pool+adder]
        order_pool = order_pool[order_pool<=(nbursts-1)]
        alldurs = np.array([np.clip(burstObjs[oo].dur,0,self.maxdur/12.) for oo in order_pool])#TODO: find a cleaner solution to avoid the case just one gets plotted when a long one follows

        boolinds = np.cumsum(alldurs)<=self.maxdur
        #if np.sum(boolinds) < len(alldurs): boolinds [np.where(boolinds==False)[0][0]] = True #one more

        mapindices = order_pool[boolinds]
        logger.debug('Number of bursts displayed on SOM %d'%(len(mapindices)))
        burstsOnMap = np.array(burstObjs)[mapindices]
        burstsOnMap_sorted = eam.attrsort_objs(burstsOnMap,'si')[::-1]#from high to low, so the highest appear on top
        
        
        ddict = {}
        ddict['params'] = ['roi','roi_int','btimes','data']
        logger.debug('Packaging bursts into dictionary for plotting.')
        #logger.info('recKeys: %s'%str(recObj.__dict__.keys()))
        for bb,burst in enumerate(burstsOnMap_sorted):
            periObj = eam.EAPeriod(burst.start-self.margint[0],burst.stop+self.margint[1],recObj)
            ddict[bb]=[burst.id,[burst.start,burst.stop],periObj.spiketimes,periObj.raw_data]
        cdict = {}
        for burst in burstObjs:
            cdict[burst.id] = [burst.roi,burst.rank,burst.si,burst.bmu]
        cdict['params'] = ['roi_int','clustid','seizidx','bmu']
        inputdict = {'kshape':recObj.som.kshape,'dcolors':recObj.som.dcolors,'clusterids':recObj.som.clusterids,\
                     'cdict':cdict,'ddict':ddict,'margint':self.margint}
        logger.debug('Inputdict done')
        flist,idtranslator = somify.plotexamples_hexhits(inputdict,sr=recObj.sr,allOnOne=True,verbose=True,forceclip=True)
        return flist[0],idtranslator

    def run(self,recObj):

        logclipper = lambda x: np.log10(x) if x>0. else 10.**-5
        tranfn = lambda logTrue: logclipper if logTrue else np.float
        
        logger.debug('Transferring SOM-attributes')
        self.setparam('sompath', recObj.som.path)
        
        mysom = recObj.som._mapdict
        for key in list(mysom.keys()):
            if key == 'dcolors': setattr(recObj.som,key,hf.string_decoder(mysom['dcolors']))
            else: setattr(recObj.som,key,mysom[key])
        vardict = {var:{'transfn':tranfn(logbool),'fac':fac} for var,logbool,fac in list(recObj.som.som_vars.values())}
        transform_value = lambda varname,value:vardict[varname]['transfn'](value)


        varnames = [vals[0] for vals in sorted(recObj.som.som_vars.values())]#you must keep the order!
        logger.info('Constructing feature array. Features: %s'%(str(varnames)))
        self.setparam('features',varnames)


        if not hasattr(recObj,'_bursts'): get_bursts = True
        else: 
            get_bursts = False if np.array([hasattr(recObj.bursts[0],attr) for attr in varnames]).all() else True

        if get_bursts:

            logger.info('Detecting bursts and extracting features.')
            BD = BurstDetection()
            BD.features = varnames
            BD.run(recObj)
            newdict = {key:val for key,val in list(self.defaults.items()) if not key in ['save_data','groupkey']}
            newdict.update({key:getattr(BD,key) for key in list(BD.defaults.keys())})

            for key in list(BD.defaults.keys()):
                if not key in ['save_data','groupkey']: setattr(self,key,getattr(BD,key))
            self.defaults = newdict
            
            




        mapbursts = eam.filter_objs(recObj.bursts,[lambda x:x.b_n>=self.nmin])
        roimat = np.zeros((len(mapbursts), len(varnames)))
        logger.info('%d bursts have N(spikes)>=%d and will be classified on SOM.'%(len(mapbursts),self.nmin))
        for bb, burst in enumerate(mapbursts):
            entries = np.array([transform_value(var, getattr(burst, var)) for var in varnames])
            roimat[bb] = entries


        logger.info('Zscoring to map-dataset.')
        roimat = (roimat - recObj.som.mean_std_mat[0]) / recObj.som.mean_std_mat[1]#zscoring for the others
        
        weightfacs = np.array([vardict[var]['fac'] for var in varnames])
        logger.info('Multiplying by variable weights: %s'%(str(weightfacs)))
        self.setparam('weights',weightfacs)
        roimat = roimat*weightfacs
        
        logger.info('Indentifying BMUs')
        bmus = somify.get_bmus(roimat,recObj.som.weights)
        
        logger.info('Setting rank and si attributes.')
        clusts = recObj.som.clusterids[bmus].astype(np.int)
        sinds = recObj.som.seizidx[bmus]
        for bb,burst in enumerate(mapbursts):    
            for attr,val in zip(['rank','si','bmu'],[clusts[bb],sinds[bb],bmus[bb]]):             
                         setattr(burst, attr, val) 
        
                         
        smallbursts = [burst for burst in recObj.bursts if not burst in mapbursts]
        logger.info('%d bursts have N(spikes)<%d, rank,si and bmu are set to None.'%(len(smallbursts),self.nmin))
        for bb,burst in enumerate(smallbursts):
            for attr in ['rank','si','bmu']: setattr(burst,attr,None)

        if self.save_figs:
            fdir = os.path.join(recObj.figpath,recObj.id+'__burstClassification')
            hf.createPath(fdir)
            
            if len(mapbursts)==0: logger.info('No bursts on map available - SOM-plot will not be generated.')
            else:
                logger.debug('Plotting examples on SOM.')
                
                f, translationdict = self.plotOnMap(recObj,mapbursts[:])
                figname = os.path.join(fdir,recObj.id+'__examplesOnMap_'+recObj.som.id+self.figformat)
                self.figsave(f,figname)
                logger.info('Saved SOM-example bursts at %s'%(figname))

                txtpath = os.path.join(fdir,recObj.id+'__mapIds_to_ROIids.txt')
                somify.writeExampleIDdict_toTXT(translationdict,txtpath)   

            #and now the bursts classified on the trace
            #recObj.set_bursts(recObj.bursts)
            figname = os.path.join(fdir,recObj.id+'__traceClassified_'+recObj.som.id+self.figformat)
            f = pii.plot_classifiedTrace(recObj,pandur=self.pandur2,fwidth=self.fwidth2,showitems=['raw','artifacts','bursts','singlets'],\
                                     ph=self.panheight2,legendOn=True)
            
            
            self.figsave(f,figname)
            logger.info('Saved whole trace classified at %s'%(figname))

        recObj._burstdict = {}
        print ('MAPb',len(mapbursts),len(smallbursts),len(recObj.burstdict))
        for burst in mapbursts+smallbursts:
            recObj.burstdict[burst.id] = [burst.roi,burst.rank,burst.si,burst.bmu]
        recObj.burstdict['params'] = ['roi_int','clustid','seizidx','bmu']  
          
        if self.save_data:
            logger.info('Saving SOM-classified bursts to dictionary.')

            #converting burstdict to hdf5-savable/matlab readable data format
            datamat = np.vstack( [np.r_[key, np.array(val[0]), np.array(val[1:])] for key, val in recObj.burstdict.items() \
                                  if  not key == 'params'])
            datamat[datamat == None] = np.nan
            paramvals = ['key_id', 'start', 'stop'] + recObj.burstdict['params'][1:]

            self.datadict = {'values': datamat.astype('float'),'params': paramvals}
            self._save_resultsdict(recObj)
            
        
        
        

class StateAnalysis(Analysis):
    
    
    def __init__(self,logger=None):
        for key,val in list(self.defaults.items()):
            setattr(self,key,val)
        for key,val in list(self.plot_dict.items()):
            setattr(self,key,val)
        self.method = 'Defining IP,IIP,depr.,transition based on clustering and changepoint analysis.'
        #if logger is not None: self.logger=logger
        #logger.info('Hello')  


    @property
    def plot_dict(self):
        return {'InchPSec': 8/60./60.,\
                'figformat': '.png',\
                'mydpi': 100}

    
    def _detectIPs(self,recObj):
        logger.info('Level1: Detecting IPs and non-IPs')
        severe_bursts = eam.filter_objs(recObj.bursts,[lambda x: x.cname in self.severe])
        bad_times = np.vstack([burst.roi for burst in severe_bursts]) if len(severe_bursts)>0 else np.array([[],[]]).T
        logger.debug('N severe bursts: %d'%(bad_times.shape[0]))
        ip_times = pii.merge_bursts(bad_times,self.maxint_ips)
        logger.debug('N IPs: %d'%(ip_times.shape[0]))
        pii.set_IP_nonIPs(recObj,ip_times)
        
        self.ip_times = ip_times
        
    
    def _cpt_analysis(self,recObj):
        logger.info('Level2: Breaking up non-IPs')
        logger.debug('N states (IP and temp): %d'%(len(recObj.states)))
        pii.break_interstates(recObj,self.cpt_thresh,self.cpt_drift,self.depri_thresh)
        logger.debug('Trace is covered by states: %s'%str(recObj.check_coverage()))
        logger.info('N states after breakup: %d'%(len(recObj.states)))
        
    
    def _insert_artifacts(self,recObj):
        logger.info('Level3: Inserting Artifacts')
        newstatelist = pii.insertPredatorStates(recObj.states[:],recObj.artifactTimes,predator_type='art',\
                                                mindur=self.mindurArtifact,output=True)
        recObj.states = newstatelist
        logger.info('N states after inserting artifacts: %d'%(len(recObj.states)))
    
    def _nibble_endstates(self,recObj):
        logger.info('Level4: Nibbling endstates and checking state integrity')
        newstatelist = pii.nibble_endstates(recObj.states[:],self.befbuff,self.endbuff,output=True)
        recObj.states = newstatelist
        pii.clear_artifactNeighbours(recObj.states)
        pii.set_state_integrity(recObj.states)
        recObj.check_coverage()
        pii.merge_statebursts(recObj.states)
        
    def _check_increase_validity(self,recObj):
        logger.info('Level5: Can peri-increases be assigned or are IIPs too short?')
        newstates = pii.reject_cpts(recObj.states,minTotalIIP=self.minTotalIIP)
        recObj.states = newstates
        pii.merge_statebursts(recObj.states)
        recObj.check_coverage()
        self.increaseEnabled = True if np.sum([iip.dur for iip in \
                                          eam.filter_objs(recObj.states,[lambda x: x.state=='IIP'])])>self.minTotalIIP else False
    

    def _connect_densities(self,recObj): 
        logger.info('Level6: Connecting densities of %s'%(str(self.denseclasses)))
        pii.swallow_densities(recObj.states,fracMax=self.MfracMax,burstclass=self.denseclasses)
        
    
    def run(self,recObj):
        

        npans = 7
        
        f, axarr = subplots(npans,1, sharex=True,figsize=(self.InchPSec*recObj.dur,npans*0.4),facecolor='w')
        f.subplots_adjust(left=0.,right=1.,hspace=0.,wspace=0.,bottom=0.,top=1.)
        recObj.plot(['bursts'],ylab='B',unit='min',ax=axarr[-1])
        
        self._detectIPs(recObj)
        recObj.plot(['states'],ylab='L1',unit='min',ax=axarr[-2])
        
        self._cpt_analysis(recObj)
        recObj.plot(['states'],ylab='L2',unit='min',ax=axarr[-3])
        
        self._insert_artifacts(recObj)
        recObj.plot(['states'],ylab='L3',unit='min',ax=axarr[-4])
        
        self._nibble_endstates(recObj)
        recObj.plot(['states'],ylab='L4',unit='min',counterOn=True,ax=axarr[-5])
        
        
        self._check_increase_validity(recObj)
        recObj.plot(['states'],ylab='L5',unit='min',counterOn=True,ax=axarr[-6])
        
        self._connect_densities(recObj)
        recObj.plot(['states'],ylab='L6',unit='min',counterOn=True,ax=axarr[-7])

        if self.save_figs:
            
            fdir = os.path.join(recObj.figpath,recObj.id+'__states')
            hf.createPath(fdir)
            figname = os.path.join(fdir,recObj.id+'__stateGenesis'+self.figformat)
            self.figsave(f,figname,bbox_inches='tight')
            logger.info('Saved genesis figure to '+figname)
            
            f = pii.plot_classifiedTrace(recObj)
            figname = os.path.join(fdir,recObj.id+'__statesClassified'+self.figformat)
            self.figsave(f,figname)
            logger.info('Saved classification figure to '+figname)
        else: close('all')
        
        self.setparam('increaseEnabled',self.increaseEnabled)
        
        recObj.statedict = pii.states_to_dict(recObj.states)
        if self.save_data:
            logger.info('Saving states to dictionary')
            self.datadict = recObj.statedict
            
            self._save_resultsdict(recObj)
            
        
       
class Diagnostics(Analysis):
    def __init__(self):
        self.method = 'Diagnostic measures for asssessing severity of epilepsy.'
        for key,val in list(self.defaults.items()):setattr(self,key,val)

    def write(self,recObj):

        self.datadict = {'tfracs': recObj.diagn_tfracs, 'rates': recObj.diagn_rates}
        self._save_resultsdict(recObj)
