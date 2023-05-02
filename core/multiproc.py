# TODOS main
# TODO logger info spike sorting and burst detection
# TODO save plotting --> and projection plotting if desired for spike detection, sorting and burst classification
# TODO plot example snippets

# TODO burst classification: without aRec object (implying whole loading of spiketrains)

# TODO integrate into runthrough script

# EVENTUAL TODOS
# TODO make clusters selectable in SpikeSorting multiprocessing mode
# TODO make sequential clustering in SpikeSorting (instead of later projection of nonclustered spikes)
# TODO make artifact-detection sequential for large datasets
# TODO make polarity detection sequential


import core.helpers as hf
import core.ed_detection as edd
from core import blipS,artisfaction,blipsort,somify
from core import ea_management as eam

import numpy as np
import logging
import time
import os
import h5py
import deepdish.io as dio

from scipy.stats import scoreatpercentile
from sklearn import mixture
# import pathos as pa
import pathos.multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

logger = logging.getLogger(__name__)


def resample_save(F,hdataset,chidx,snipstart,snipstop,sr_old,sr_new):
    from scipy.signal import resample

    rate_ratio = sr_old/sr_new
    ww = snipstop-snipstart
    snip = F.readSignal(chidx, snipstart, ww)
    snip2 = hf.checkadapt_mv(hf.checkadapt_zeromean(snip))
    # print (len(snip),snip.min(),snip.max())
    resampled = resample(snip2,np.int(ww/rate_ratio))
    # print (len(resampled), resampled.min(),resampled.max())
    ptstart,ptstop = np.int(snipstart/rate_ratio),np.int(snipstop/rate_ratio)
    hdataset[ptstart:ptstop] = resampled


def retrieve_channidx_edf(filename,**kwargs):
    """ only works with one channel so far"""

    import pyedflib
    stopword = 'd'

    if 'chanlist' in kwargs:
        chanlist = kwargs['chanlist']
        if len(chanlist)>1: logger.warning("Multiprocessing supports only one channel for extraction, you have %i"%(len(chanlist)))

    F = pyedflib.EdfReader(filename)
    logger.info('Opening %s ...' % (filename))
    starttime = time.time()
    logger.debug('Opening took %1.2f s' % (time.time() - starttime))
    allchans = F.getSignalLabels()

    # rename properly (before: ugly nested strings) when its python 3 and causing a weired clash with NEO
    if 'chanlist' not in kwargs:
        print('Channels available: \n' + ''.join([chan + '\n' for chan in allchans]))
        chanlist = []
        while True:
            chstr = input('Enter channame (press %s to exit): ' % (stopword))
            if chstr.strip() == stopword: break
            chanlist += [chstr]
    if len(chanlist) > 1: logger.warning("Multiprocessing supports only one channel for extraction, you have %i; taking only first channel" % (len(chanlist)))

    chan = chanlist[0]
    chidx = allchans.index(chan)

    return F,chidx


class Multiprocessing(edd.Analysis):
    def __init__(self):
        for key, val in list(self.defaults.items()):
            setattr(self, key, val)

    def update_settings(self,cfgdict):
        self.classname = type(self).__name__
        if self.classname in cfgdict:
            if 'setparams' in cfgdict[self.classname]:
                settingsdict = cfgdict[self.classname]['setparams']
                for key,infradict in settingsdict.items():getattr(self, key).update(infradict)

class Preprocessing(edd.Analysis):
    def __init__(self, sourcefile, chanid='NA'):

        for key, val in list(self.defaults.items()):
            setattr(self, key, val)
            #

        self.setparam('chanid', chanid)


        self.method = 'Resampling from source file, multiprocessing.'
        self.sourcefile = sourcefile
        self.sourcetype = sourcefile.split('.')[-1]

    def writeget_result(self, recObj, **kwargs):

        logger.info('Multiprocessing Preprocessing')
        if self.sourcetype == 'smr': logger.warning('Not yet available for .smr')
        elif self.sourcetype == 'edf':
            if self.chanid == 'NA':
                F,chidx = retrieve_channidx_edf(self.sourcefile)
            else: F,chidx = retrieve_channidx_edf(self.sourcefile, chanlist=[self.chanid])

            self.sr_before = F.samplefrequency(chidx)
            minpts = np.int(self.mindur_snip * self.sr_before)
            datalen = F.getNSamples()[chidx]
            rate_ratio = self.sr_before / self.sr
            newlen = np.int(datalen / rate_ratio)
            winarray = hf.smartwins(datalen, minpts, overfrac=0., pow2=True)

            self.datadict = {'sr': self.sr}#here just write sampling rate, the rest comes later
            if 'to_file' in kwargs: self.outfile = kwargs['to_file']
            else: self.outfile = str(recObj.rawfileH)

            #hf.saveto_hdf5(self.savedict, self.outfile, overwrite_file=True, mergewithexisting=False)
            with h5py.File(self.outfile,'w') as fdst:
                dgr = fdest.create_group('data')
                #dgr.create_dataset('trace',data=self.resampled,dtype='f')
                dgr.attrs['sr'] = self.sr
                igr = fdest.create_group('info')
                for key,val in self.infodict.items():
                    igr.attrs[key] = val
                mgr = fdest.create_group('methods')
                mgr.attrs['sr'] = self.sr

            #now running the picking and resampling
            dspath = '/data/trace'
            fdest = h5py.File(self.outfile, 'r+')
            myds = fdest.create_dataset(dspath, data=np.zeros(newlen))

            subfn = lambda mywin: resample_save(F, myds, chidx, mywin[0], mywin[1], self.sr_before, self.sr)

            pool = mp.ThreadPool(mp.cpu_count())
            [pool.apply(subfn, args=([win])) for win in winarray]
            pool.close()

            F._close()
            fdest.close()



class EdDetection(edd.Analysis):
    def __init__(self):
        for key,val in list(self.defaults.items()):
            setattr(self,key,val)
        for key,val in list(self.plot_dict.items()):
            setattr(self,key,val)

    @property
    def plot_dict(self):
        return {}

    def set_mp_params(self,mpobj):
        for key,val in mpobj.EdDetection.items():
            if not key == 'flag': setattr(self,key,val)

        # used to prepare efficient spike sorting, not strictly needed for spike detection"""
        try:
            self.bunchpts = np.int(mpobj.SpikeSorting['bunch_int']*self.sr)

        except:
            self.bunchpts = np.int(self.pts_tot)
            logger.warning('Not accessing bunchpoints!')

    @property
    def threshx(self):
        return np.arange(self.thr_range[0], self.thr_range[1] + self.thr_res, self.thr_res)

    def get_spikes(self, thresh, avg_power):
        return blipS.zthreshdetect_blips(avg_power, sr=self.sr, thresh=thresh, pruneOn=self.prune_on,
                                         dtime=self.dtime)

    def get_fofthresh(self,apower):
        if type(apower) == np.ma.core.MaskedArray:
            avg_power = apower[apower.mask == False].data
        else:
            avg_power = apower[:]

        return np.array([len(self.get_spikes(zthresh, avg_power)) for zthresh in self.threshx])

    def get_zthresh(self,fofthresh):
        zthresh = blipS.getZthresh_derivMethod(fofthresh, self.threshx, peakperc=self.peakperc, \
                                               mode=self.threshmode)
        return zthresh


    def catch_ampspikes(self,spectralspikes,data):

        ampspikes = blipS.catch_undetectedBlips(spectralspikes, data, sr=self.sr, \
                                    zthresh=self.amp_thresh, blip_margin=self.spike_margin, \
                                    dtime=self.dtime, pol=self.polarity)
        return ampspikes

    def set_artifactfile(self,artifactfile):

        self.artifactfile = artifactfile


    def get_artsfused(self):

        if os.path.isfile(self.artifactfile):
            logger.info('Loading artifacts.')
            artdict = artisfaction.readArtdict_txt(self.artifactfile)
            self.arts_fused = artisfaction.fuse_artifacts([artdict['arts'], artdict['sats']],
                                                          [self.marg_arts, self.marg_sats],
                                                         mindist=self.mindist_arts)

        else:
            logger.warning('Specified artifactfile does not exist --> disregarding artifacts')



    def init_from_raw(self,rawfileH):
        self.rawfile = h5py.File(rawfileH, 'r')
        grp = self.rawfile['data']
        self.rawhand = grp['trace']
        self.sr = grp.attrs['sr']
        self.pts_tot = self.rawhand.size
        #datadur = np.int(self.pts_tot/ self.sr)

    @property
    def apowlen_tot(self):
        return self.pts_tot-self.window+1-np.int(self.offset*self.sr)

    @property
    def overhang_pts(self):
        return np.mod(self.pts_tot,self.durpts)-self.window//2+1

    @property
    def artpoints(self):
        if not hasattr(self,'_artpoints'):
            self.get_artsfused()
            self._artpoints = (self.arts_fused*self.sr).astype(np.int)
        return self._artpoints

    @property
    def artdur(self):
        return np.sum(np.diff(self.artpoints.T)) / self.sr


    @property
    def winarray(self):
        if not hasattr(self,'_winarray'):
            self.durpts = np.int(self.dicedur*self.sr)
            offpts = np.int(self.offset*self.sr)
            winarray = hf.smartwins(self.pts_tot - offpts, self.durpts, overfrac=0., pow2=False)
            winarray += offpts
            # overlap to maximize spectrum
            winarray[1:, 0] -= self.window // 2
            winarray[:-1, 1] += self.window // 2
            self._winarray = winarray
        return self._winarray

    @property
    def N_snips(self):
        return len(self.winarray)

    @property
    def maskdict(self):
        if not hasattr(self,'_maskdict'):
            maskdict = {}
            for ii,[pstart,pstop] in enumerate(self.winarray):
                arts_temp = self.artpoints.clip(pstart, pstop).T
                cond = np.sum(np.diff(arts_temp)> 0)
                if cond:
                    thisarts = arts_temp[np.where(np.diff(arts_temp) > 0)[0]]
                    maskdict[ii] = thisarts.T - pstart
                else:
                    maskdict[ii] = np.array([[],[]])
            self._maskdict = maskdict
        return self._maskdict

    @property
    def winarray_adj(self):
        if not hasattr(self,'_winarray_adj'):
            winarray_adj = np.zeros_like(self.winarray)
            for ii in np.arange(self.N_snips):
                pstart, pstop = self.winarray[ii]
                N_maskpts = np.sum(np.diff(self.maskdict[ii].T))
                if N_maskpts == 0:
                    winarray_adj[ii] = self.winarray[ii]
                else:
                    if pstop + N_maskpts <= self.winarray[-1, 1]:
                        winarray_adj[ii] = pstart, pstop + N_maskpts
                    else:
                        winarray_adj[ii] = pstart - N_maskpts, pstop
            self._winarray_adj = winarray_adj
        return self._winarray_adj

    def get_datasnip(self,pstart,pstop):
        return self.rawhand[pstart:pstop]

    def mask_datasnip(self,snip,artpts_of_snip):
        '''artsfused_of_snip: shape is 2 x N_aritfacts in snippet; starts and stops in points of artifact episodes'''
        if np.size(artpts_of_snip) > 0:
            return artisfaction.mask_data(snip, artpts_of_snip)
        else: return snip

    def getmask_snip(self,idx):
        pstart, pstop = self.winarray_adj[idx]
        snip0 = self.get_datasnip(pstart, pstop)
        snip = self.mask_datasnip(snip0,self.maskdict[idx])
        return snip

    def get_spikes_for_snip(self, snip):

        sp = blipS.get_spectrogram(snip, self.window)
        sp_norm, normfacs = blipS.dynamic_normalisation(sp, norm=self.norm, verbose=True)
        del sp
        avg_power_save = blipS.average_spectralBins(sp_norm, sr=self.sr, avg_lim=self.avg_lim)
        del sp_norm
        avg_power = blipS.preprocess_data(avg_power_save,
                                          len(snip))  # take care, here we should only pad at the edges if necessary

        fofthresh = self.get_fofthresh(avg_power)
        zthresh = self.get_zthresh(fofthresh)
        spikes_spectral = self.get_spikes(zthresh, avg_power)
        spikes_amp = self.catch_ampspikes(spikes_spectral, snip)
        return {'spikes_spectral': spikes_spectral, 'spikes_amp': spikes_amp, 'zthresh': zthresh,
                'fofthresh': fofthresh, 'avg_power_save': avg_power_save, 'normfacs': normfacs}

    def get_events_admitted(self,eventtimes, idx):
        offsetter = self.winarray_adj[idx,0]/self.sr
        if idx < self.N_snips - 1:
            times = eventtimes[eventtimes <= self.dicedur]
        elif idx == self.N_snips-1:
            times = eventtimes[eventtimes >= (self.dicedur - self.overhang_pts/self.sr)]
        return times + offsetter

    def setmake_barray(self):
        self.N_bunchpts = np.int(np.ceil(self.pts_tot / self.bunchpts))
        self.barray = np.arange(self.N_bunchpts) * self.bunchpts
        self.barray[-1] = self.pts_tot #the last bunch is the overlap longer

    def get_spikeslice_indices(self, spikes, idx):
        ptmin, ptmax = np.array([idx * self.durpts, (idx + 1) * self.durpts])
        tsavail = self.barray[(self.barray >= ptmin) & (self.barray < ptmax)] / self.sr
        #print ("tsavail",tsavail)
        spikeinds = np.array([np.argmin(np.abs(tavail - spikes)) for tavail in tsavail]).astype(np.int)
        return spikeinds


    @property
    def apow_cutwins(self):
        if not hasattr(self,'_apow_cutwins'):
            apow_fillpts_bord = self.durpts - self.window // 2
            inciders = np.tile(np.array([0, self.durpts]), (self.N_snips - 1, 1))
            inciders[0, 1] = apow_fillpts_bord
            lastinc = np.array([apow_fillpts_bord - self.overhang_pts, apow_fillpts_bord])
            self._apow_cutwins = np.vstack([inciders, lastinc])
        return self._apow_cutwins

    @property
    def apow_inwins(self):
        if not hasattr(self,'_apow_inwins'):
            temp = np.r_[0, np.cumsum(np.diff(self.apow_cutwins))]
            self._apow_inwins = np.vstack([temp[:-1], temp[1:]]).T
        return self._apow_inwins


       

    def plot_snipresults(self, rdict, idx):

        tstart, tstop = self.winarray_adj[idx] / self.sr
        raw = self.getmask_snip(idx)
        pow = blipS.preprocess_data(rdict['avg_power_save'], len(raw))
        spikes_spectral = self.get_events_admitted(rdict['spikes_spectral'], idx)
        spikes_amp = self.get_events_admitted(rdict['spikes_amp'], idx)
        tvec = np.linspace(tstart, tstop, len(pow))
        blankbord = self.dicedur / 50.

        f, axarr = plt.subplots(3, 1, figsize=(16, 5), sharex=True,
                                gridspec_kw={'height_ratios': [1, 3, 3.], 'wspace': 0.01})
        f.subplots_adjust(left=0.05, right=0.99, top=0.95)
        f.suptitle('Snip Nb %i' % idx)
        axarr[2].plot(tvec, raw, 'k')
        axarr[2].set_ylabel('mV')
        axarr[1].plot(tvec, pow, 'r')
        axarr[1].set_ylabel('Z(avg. power)')
        axarr[1].axhline(rdict['zthresh'], color='r', linestyle='--')
        axarr[0].vlines(spikes_spectral, 0., 1., color='r', alpha=0.5)
        axarr[0].vlines(spikes_amp, 0., 1., color='b', alpha=0.5)
        axarr[0].set_ylim([0., 1.])
        axarr[2].set_xlim([tstart - blankbord, tstop + blankbord])
        axarr[2].set_xlabel('Time [s]')
        if idx == self.N_snips - 1:
            x0, x1 = tstart, tvec[-self.overhang_pts]
        else:
            x0, x1 = tvec[self.durpts], tstop
        for ax in axarr:
            ax.axvspan(x0, x1, color='gray', alpha=0.5, zorder=100)
            # for bord in [x0,x1]: ax.axvline(bord,color='c',linewidth=2)
        return f
    

    def save_basics(self,recObj):
        logger.info('Saving basics')
        # prepare a saving structure and save some stuff right away
        hf.saveto_hdf5({recObj.cfg_ana['Preprocessing']['groupkey']: {'path': recObj.rawfileH}}, recObj.resultsfileH,
                       mergewithexisting=True, overwrite_groups=True, overwrite_file=False)
        if os.path.isfile(recObj.rawfileH): hf.makelink_hdf5(recObj.rawfileH, recObj.resultsfileH,
                                                           recObj.cfg_ana['Preprocessing']['groupkey'] + '/link')
        self.datadict = {'winarray_adj': self.winarray_adj, 'polarity': self.polarity, 't_offset': self.offset, \
                        't_total': self.pts_tot / self.sr}
        self.datadict['mask_startStop_sec'] = self.arts_fused if self.consider_artifacts else None
        temp = self.datadict['t_total'] - self.offset
        artdur = np.sum(np.diff(np.clip(self.arts_fused, self.offset, self.arts_fused.max()))) if np.size(
            self.arts_fused) > 0 else 0
        self.datadict['t_analyzed'] = temp - artdur
        self._save_resultsdict(recObj)
        
    @property
    def dsdimdict(self):
        if not hasattr(self,'_dsdimdict'):
            self._dsdimdict = {'zthresh':(self.N_snips),'fOfThresh':(self.N_snips,len(self.threshx)),'spikes_spectral':(),\
                               'spikes_amp':(),'spikes':(),'spikeslicers':(self.N_bunchpts)}
            if self.save_apower:
                apowsize = (2,self.apowlen_tot) if np.size(self.artpoints)>0 else (self.apowlen_tot)
                self._dsdimdict.update({'apower':apowsize,'normfacs':(self.N_snips,2,self.window//2 + 1)})
        return self._dsdimdict

    def create_empty_nest(self,filename):
        logger.info('Creating empty nest')
        with h5py.File(filename, 'r+') as fdest:
            #fdest = h5py.File(filename, 'r+')
            for dsname, dsshape in self.dsdimdict.items():
                dspath = '%s/data/%s' % (self.groupkey, dsname)
                if dspath in fdest:
                    logger.warning('Overwriting dataset %s in %s' % (dspath, filename))
                    del fdest[dspath]
                if dsname in ['spikes_spectral', 'spikes_amp', 'spikes']:
                    fdest.create_dataset(dspath, data=np.array([]), maxshape=(None,))
                else:
                    fdest.create_dataset(dspath, data=np.zeros(dsshape))
            #fdest.close()
            
    @property
    def bunchinds(self):
        if not hasattr(self,'_bunchinds'):
            temp = np.array([len(self.barray[(self.barray >= idx * self.durpts) & (self.barray < (idx + 1) * self.durpts)]) for idx in
                 np.arange(self.N_snips)])
            cumbunch = np.r_[0, np.cumsum(temp)]
            self._bunchinds = np.vstack([cumbunch[:-1], cumbunch[1:]]).T
        return self._bunchinds


    def write_snipresults(self,fhand,rdict,idx):

        logger.info('Writing snipresults, snipnb %i'%idx)
        dsdict = {dsname:fhand['%s/data/%s'%(self.groupkey,dsname)] for dsname in self.dsdimdict.keys()}

        dsdict['zthresh'][idx] = rdict['zthresh']
        dsdict['fOfThresh'][idx] = rdict['fofthresh']
        # spiky stuff
        spikes_spectral = self.get_events_admitted(rdict['spikes_spectral'], idx)
        spikes_amp = self.get_events_admitted(rdict['spikes_amp'], idx)
        spikes = np.sort(np.r_[spikes_spectral, spikes_amp])

        #know efficently which # spikes makes the border of a time window --> used later for spike sorting
        sliceinds = self.get_spikeslice_indices(spikes, idx) + dsdict['spikes'].shape[0]
        b0, b1 = self.bunchinds[idx]
        #print ('SLICEINDS',sliceinds)
        dsdict['spikeslicers'][b0:b1] = sliceinds

        hf.append_dataset_hdf5(dsdict['spikes'], spikes)
        hf.append_dataset_hdf5(dsdict['spikes_spectral'], spikes_spectral)
        hf.append_dataset_hdf5(dsdict['spikes_amp'], spikes_amp)

        if self.save_apower:
            dsdict['normfacs'][idx] = rdict['normfacs']
            in0, in1 = self.apow_inwins[idx]
            cut0, cut1 = self.apow_cutwins[idx]
            apowdata = rdict['avg_power_save'][cut0:cut1]
            apowhand = dsdict['apower']
            if np.size(self.artpoints) == 0:
                apowhand[in0:in1] = apowdata
            elif type(apowdata) == np.ma.core.MaskedArray:
                apowhand[0, in0:in1] = apowdata.data
                apowhand[1, in0:in1] = apowdata.mask.astype(np.int)
            else:
                apowhand[0, in0:in1] = apowdata


    def get_snipresults(self,idx):
        logger.info('Getting snipresults, snipnb %i'%idx)
        snip = self.getmask_snip(idx)
        return self.get_spikes_for_snip(snip)

    def getwrite_snipresults(self,fhand,idx):
        rdict = self.get_snipresults(idx)
        self.write_snipresults(fhand,rdict,idx)

    def writeget_results(self,recObj):
        self.recObj = recObj
        self.polarity = recObj.polarity
        self.setmake_barray()
        self.save_basics(self.recObj)
        self.create_empty_nest(self.recObj.resultsfileH)
        logger.info('Start Multiprocessing %s'%(self.__class__))
        with h5py.File(self.recObj.resultsfileH, 'r+') as fdest:
            with mp.ThreadPool(mp.cpu_count()) as pool:
                [pool.apply(self.getwrite_snipresults, args=(fdest,idx)) for idx in np.arange(self.N_snips)]

        logger.info('Finished Multiprocessing%s'%(self.__class__))



def get_newrow(initrow,maxlen,powlevel):
    adder = maxlen/(2**powlevel)
    newrow = initrow+adder#[initrow<(maxlen+adder)]
    return newrow[newrow<maxlen]#np.r_[initrow,newrow]


def get_indexarray(sniplen):
    maxlen = sniplen-1
    npows = maxlen.bit_length()
    currrow = np.array([0,maxlen])
    for ii in np.arange(1,npows+1):
        newrow = get_newrow(currrow,maxlen,ii)
        currrow = np.r_[currrow,newrow]
    introw = currrow.astype(np.int)
    seen = set()
    outarray = np.array([x for x in introw if not (x in seen or seen.add(x))])
    assert len(outarray) == (maxlen+1),'len mismatch %i vs %i'%(len(outarray),maxlen+1)
    return outarray





def project_on_comps_(snips, pcadict):
    evecs,evals,meanstd = [pcadict[key] for key in ['evecs','evals','meanstd']]
    zdata = (snips - meanstd[0]) / meanstd[1]
    scores = np.dot(evecs.T, zdata.T)
    return scores / np.sqrt(evals[:,np.newaxis])


class SpikeSorting(edd.Analysis):
    def __init__(self):
        for key,val in list(self.defaults.items()):
            setattr(self,key,val)
        for key,val in list(self.plot_dict.items()):
            setattr(self,key,val)

    @property
    def plot_dict(self):
        return {}

    def set_mp_params(self,mpobj):
        """used to prepare efficient burst detection, not strictly needed for spike detection"""
        for key,val in mpobj.BurstDetection.items():
            if not key == 'flag': setattr(self,key,val)
        for key,val in mpobj.SpikeSorting.items():
            if not key == 'flag': setattr(self,key,val)


    def init_from_prev(self,fhand,groupname):
        #group_prev = aRec.cfg_ana['EdDetection']['groupkey']
        self.fhand = fhand
        grphand = fhand['/%s' % (groupname)]
        slicers = grphand['data/spikeslicers'][:]
        self.slicewins = np.vstack([slicers[:-1], slicers[1:]]).T.astype(np.int)
        self.n_slicewins = len(self.slicewins)
        self.sliceseq = get_indexarray(self.n_slicewins)
        self.srcspikehand = grphand['data/spikes']

        self.mask_startStop_sec = grphand['data/mask_startStop_sec'].value

        for attrname,attrkey in zip(['offset','t_total','t_analyzed','polarity'],['t_offset','t_total','t_analyzed','polarity']):
            val = dio.load(fhand.filename,'/%s/data/%s'%(groupname,attrkey))
            setattr(self,attrname,val)


    def save_basics(self, recObj):
        logger.info('Saving basics')
        # prepare a saving structure and save some stuff right away
        self.datadict = {'slicewins': self.slicewins, 'sliceseq': self.sliceseq,'polarity': self.polarity, 't_offset': self.offset, \
                         't_total': self.t_total,'t_analyzed':self.t_analyzed,'mask_startStop_sec':self.mask_startStop_sec}
        self._save_resultsdict(recObj)

    def get_slicespikes(self, idx):
        sliceidx = self.sliceseq[idx]
        startidx, stopidx = self.slicewins[sliceidx]
        if sliceidx == self.n_slicewins-1: stopidx = stopidx+1
        spikes = self.srcspikehand[startidx:stopidx]
        return np.array([spike for spike in spikes if not spike in self.protected_spikes])

    def get_sparse_nonsparse_spikes(self,spikes):
        # protect the endblips
        # lowint,upint = self.bunch_int*np.array([idx,idx+1])
        # startspikes,endspikes = np.array([spike for spike in spikes if spike-np.sum(cutwin)<lowint])np.array([spike for spike in spikes if spike-np.sum(cutwin)<lowint])
        # protectedEndblips = np.array([blip for blip in spiketimes if len(data)/sr-blip<np.sum(cutwin)])
        sparseSpikes = blipsort.get_sparseBlips(spikes, intWidth=[3., 4., 2.], nblip_limit=[3., 4., 4.], verbose=False,
                                                strictBef=True)
        groupspikes = np.array([spike for spike in spikes if not spike in sparseSpikes])
        return [sparseSpikes,groupspikes]



    def make_empty_nest(self):
        dslist = ['spikes','noisetimes']
        handlist = []
        for dsname in dslist:
            #dsname = 'spikes'
            dspath = '%s/data/%s' % (self.groupkey, dsname)
            if dspath in self.fhand:
                logger.warning('Overwriting dataset %s in %s' % (dspath, str(self.fhand.file)))
                del self.fhand[dspath]
            hand = self.fhand.create_dataset(dspath, data=np.array([]), maxshape=(self.srcspikehand.size,))
            handlist += [hand]
        self.spikehand,self.noisehand = handlist




    def collect_sparsespikes(self):
        #sparse_in_win = np.array([])
        self.allsparse = np.array([])
        for idx in np.arange(self.n_slicewins):
            slicespikes = self.get_slicespikes(idx)
            sparsesp,groupsp = self.get_sparse_nonsparse_spikes(slicespikes)
            #sparse_in_win = np.r_[sparse_in_win,len(sparsesp)]
            #write the groupspikes to the empty nest
            hf.append_dataset_hdf5(self.spikehand, groupsp)
            self.allsparse = np.r_[self.allsparse,sparsesp]
            if len(self.allsparse) >= self.Nspikes_max: break

        self.waiting_wins = self.sliceseq[idx:]
        self.stopidx = idx

    def get_protect_borderspikes(self):
        up_bord = self.t_total - self.bigwin[1]/self.sr
        lastspikes = self.srcspikehand[-10:]
        upspikes = lastspikes[lastspikes>up_bord]

        low_bord = self.bigwin[0]/self.sr
        firstspikes = self.srcspikehand[:10]
        lowspikes = firstspikes[firstspikes<low_bord]
        self.protected_spikes = np.r_[lowspikes,upspikes]
        logger.info('N protected spikes at borders: %i'%len(self.protected_spikes))
        if self.protected_spikes.size > 0:
            logger.info ('Saving protected')
            hf.append_dataset_hdf5(self.spikehand, self.protected_spikes)



    def init_from_raw(self,rawfileH):
        self.rawfile = h5py.File(rawfileH, 'r')
        grp = self.rawfile['data']
        self.rawhand = grp['trace']
        self.sr = grp.attrs['sr']
        self.pts_tot = self.rawhand.size
        #datadur = np.int(self.pts_tot/ self.sr)

    @property
    def ptssearch(self):
        if not hasattr(self, '_ptssearch'):
            self._ptssearch = (np.array(self.minsearchwin) * self.sr).astype(np.int)
        return self._ptssearch
    
    @property
    def ptscut(self):
        if not hasattr(self, '_ptscut'):
            self._ptscut = (np.array(self.cutwin) * self.sr).astype(np.int)
        return self._ptscut

    @property
    def bigwin(self):
        if not hasattr(self,'_bigwin'):
            self._bigwin = self.ptssearch+self.ptscut
        return self._bigwin
    
    @property
    def subsearchwin(self):
        if not hasattr(self, '_subsearchwin'):
            self._subsearchwin = np.array([0,np.sum(self.ptssearch)])+self.ptscut[0]
        return self._subsearchwin


    def get_cutout(self,spike,polfac=1):

        """for positive polarity, signfac=-1, for negative use signfac= +1"""
        spikept = np.int(spike * self.sr)
        tempcut = self.rawhand[spikept-self.bigwin[0]:spikept+self.bigwin[1]]*polfac
        searchwin = tempcut[self.subsearchwin[0]:self.subsearchwin[1]]#here the minimum is searched
        abspt = np.argmin(searchwin)+self.subsearchwin[0] #point in tempcut where minimum is
        return tempcut[abspt-self.ptscut[0]:abspt+self.ptscut[1]]*polfac

    def get_polfac_snip(self,snip):
        return np.sign(np.abs(snip.max())-np.abs(snip.min()))*-1

    @property
    def polpts(self):
        """points around spike around which polarity gets assessed when having mixed polarity"""
        if not hasattr(self,'_polpts'):
            self._polpts = np.int(0.02*self.sr)
        return self._polpts

    def get_cutout_polfac(self,spike):
        """used for mixed polarity spikes"""
        spikept = np.int(spike * self.sr)
        tempcut = self.rawhand[spikept-self.bigwin[0]:spikept+self.bigwin[1]]
        polfac = self.get_polfac_snip(tempcut[self.bigwin[0]-self.polpts:self.bigwin[0]+self.polpts])
        tempcut = tempcut*polfac
        searchwin = tempcut[self.subsearchwin[0]:self.subsearchwin[1]]  # here the minimum is searched
        abspt = np.argmin(searchwin) + self.subsearchwin[0]  # point in tempcut where minimum is
        return [tempcut[abspt - self.ptscut[0]:abspt + self.ptscut[1]] * polfac,polfac]

    def get_snipmat(self, spikes):

        if self.polarity in ['pos', 'neg']:
            if self.polarity == 'neg': polfac = 1
            elif self.polarity == 'pos': polfac = -1
            return [np.vstack([self.get_cutout(spike, polfac=polfac) for spike in spikes]),polfac]

        elif self.polarity == 'mix':
            tempmat = np.vstack([self.get_cutout_polfac(spike) for spike in spikes])
            datamats = [np.vstack(tempmat[tempmat[:,1]==1][:,0]),np.vstack(tempmat[tempmat[:,1]==-1][:,0])]#negative first, then positive spikemat
            polinds = [np.where(tempmat[:,1]==1)[0],np.where(tempmat[:,1]==-1)[0]]
            return  [datamats,polinds]


    def do_pca(self,snipmat):
        if self.polarity == 'mix':
            assert type(snipmat) == list,'mixed polarity requires a list of two snipmats'
            return [self.do_pca_(submat) for submat in snipmat]
        else:
            return self.do_pca_(snipmat)

    def do_pca_(self,snipmat):
        #print ('Snipmatshape',snipmat.shape)
        meansnip,stdsnip = np.mean(snipmat,axis=0),np.std(snipmat,axis=0,ddof=0)
        zdata = (snipmat-meansnip)/stdsnip
        K = np.cov(zdata.T)
        evals, evecs = np.linalg.eig(K)
        order = np.argsort(evals)[::-1]
        evecs = np.real(evecs[:, order])
        evals = np.abs(evals[order])
        scores = np.dot(evecs[:, :self.ncomp].T, zdata.T)
        scores = scores / np.sqrt(evals[:self.ncomp, np.newaxis])
        return {'evecs':evecs[:,:self.ncomp],'evals':evals[:self.ncomp],'meanstd':[meansnip,stdsnip],'scores':scores}

    def make_gmm(self,pcadict):
        if self.polarity == 'mix':
            assert type(pcadict) == list,'mixed polarity requires a list of two pcadicts'
            return [self.make_gmm_(subdict['scores']) for subdict in pcadict]
        else:
            return self.make_gmm_(pcadict['scores'])

    def make_gmm_(self,scores):
        
        try:
            clf = mixture.GaussianMixture(n_components=self.nclust,
                                          covariance_type='full')  # if you dont set random_state to a specific seed/int
        # this will lead to slighthly different results each run
        except:
            clf = mixture.GMM(n_components=self.nclust, covariance_type='full')

        clf.fit(scores.T)
        return clf


    def predict_clusterind(self,clf,pcadict):
        if self.polarity == 'mix':
            assert type(clf) == list and type(pcadict)==list,'mixed polarity requires a list of clusterfns and scores'
            return [clf0.predict(pca0['scores'].T) for clf0,pca0 in zip(clf,pcadict)]
        else: return clf.predict(pcadict['scores'].T)

    def get_meansnip(self,snips,clinds,myind):
        return np.mean(snips[clinds == myind], axis=0)

    def get_ptp_snip(self,snip):
        return np.max(snip[self.ptscut[0]:]) - snip[self.ptscut[0]]

    def get_ordervec(self,snipmat,clinds):
        if self.polarity == 'mix':
            outlist = []
            assert type(snipmat) == list and type(clinds)==list,'mixed polarity requires a list of snipmats and clusterindices'
            for submat,subinds,polfac in zip(snipmat,clinds,[1,-1]):
                ordvec = self.get_ordervec_(submat*polfac,subinds)
                outlist += [ordvec]
            return outlist
        else: return self.get_ordervec_(snipmat,clinds)

    def get_ordervec_(self,snipmat, clinds):
        # ptpsnip = ptp_snip(get_meansnip(spikemat,2))
        ptp_vals = np.array([self.get_ptp_snip(self.get_meansnip(snipmat, clinds,cln)) for cln in np.unique(clinds)])
        ptp_order = np.argsort(ptp_vals)[::-1]
        return ptp_order

    def resort(self,intvec,ordervec):
        if self.polarity == 'mix':
            assert type(intvec) == list and type(ordervec)==list,'mixed polarity requires a list of clusterindices and ordervecs'
            return [self.resort_(subinds,ordvec) for subinds,ordvec in zip(intvec,ordervec)]
        else:
            return self.resort_(intvec,ordervec)

    def resort_(self,intvec, ordervec):

        newinds = np.zeros((intvec.shape))
        for rankord, oldord in enumerate(ordervec): newinds[intvec == oldord] = rankord
        return newinds.astype(int)

    def project_on_comps(self, snips, pcadict):
        if self.polarity == 'mix':
            assert type(snips) == list and type(
                pcadict) == list, 'mixed polarity requires list of snippets and pcadicts'
            return [project_on_comps_(submat, subdict) for submat, subdict in zip(snips, pcadict)]
        else:
            return project_on_comps_(snips, pcadict)

    def get_project_spikes(self,idx, pcadict, clustfn, ordervec):

        #if self.polarity == 'neg': polfac = 1
        #elif self.polarity == 'pos': polfac = -1

        slicespikes = self.get_slicespikes(idx)
        sparsesp, groupsp = self.get_sparse_nonsparse_spikes(slicespikes)
        hf.append_dataset_hdf5(self.spikehand, groupsp)
        snipmat,polinds = self.get_snipmat(sparsesp)
        scores = self.project_on_comps(snipmat, pcadict)


        if self.polarity == 'mix':
            scoredict = [{'scores':scores[0]},{'scores':scores[1]}]
        else:
            scoredict = {'scores':scores}
        clinds0 = self.predict_clusterind(clustfn, scoredict)
        clinds = self.resort(clinds0, ordervec)
        # self.plot_clusterpanel(snipmat,clinds,pol_factor=polfac)
        return sparsesp,clinds,polinds

    def write_spikes_noisespikes(self,spikes,clinds,polinds=1):
        if self.polarity == 'mix':
            assert type(clinds) == list and type(polinds)==list,'mixed polarity requires a list of clusterindices and polaritiy indices'
            for thispolinds,thisclinds in zip(polinds,clinds):
                thisspikes = spikes[thispolinds]#selecting the spikes of right polarity
                self.write_spikes_noisespikes_(thisspikes,thisclinds)
        else:
            self.write_spikes_noisespikes_(spikes,clinds)

            #for subclinds,


    def write_spikes_noisespikes_(self,spikes,clinds):
        reject_inds = np.array([ii for ii,clind in enumerate(clinds) if clind+1 in self.noiseclust])
        accept_inds = np.array([ii for ii,clind in enumerate(clinds) if clind+1 not in self.noiseclust])

        if accept_inds.size > 0: hf.append_dataset_hdf5(self.spikehand, spikes[accept_inds])  # save noisespikes
        if reject_inds.size > 0: hf.append_dataset_hdf5(self.noisehand, spikes[reject_inds])  # save truespikes

    def get_project_write_spikes(self,idx,pcadict,clustfn,ordervec):
        sparsesp,clinds,polinds = self.get_project_spikes(idx, pcadict, clustfn, ordervec)
        self.write_spikes_noisespikes(sparsesp,clinds,polinds)

    def order_results(self):
        sortedspikes = np.unique(self.spikehand)
        self.spikehand[:] = sortedspikes
        if hasattr(self,'minspikes_block') and hasattr(self,'minburstdiff'):self.make_save_burstslicer(sortedspikes)


    def make_save_burstslicer(self,spikes):
        """Later used for efficiently detecting and classifying bursts"""
        sortedspikes = np.unique(self.spikehand)
        diffspikes = np.diff(sortedspikes)
        sepinds = np.where(diffspikes > self.minburstdiff)[0]
        sepinds2 = np.r_[0, sepinds + 1, len(sortedspikes)]
        sepslicer = np.vstack([sepinds2[:-1], sepinds2[1:]]).T

        # cumspikes = np.cumsum(np.diff(sepslicer).flatten())

        burstslicer = np.array([[], []]).T.astype(np.int)
        cumcount = 0
        fixedstart = sepslicer[0, 0]
        for ii in np.arange(sepslicer.shape[0]):
            pstart, pstop = sepslicer[ii]
            selected = sortedspikes[pstart:pstop]
            spikecount = len(selected)
            cumcount += spikecount
            if cumcount >= self.minspikes_block:
                burstslicer = np.vstack([burstslicer, np.array([fixedstart, pstop])])
                cumcount = 0
                fixedstart = np.int(pstop)
        burstslicer[-1, 1] = len(sortedspikes)
        dspath = '%s/data/%s' % (self.groupkey, 'burstslicer')
        if dspath in self.fhand:
            logger.warning('Overwriting dataset %s in %s' % (dspath, self.fhand.filename))
            del self.fhand[dspath]
        self.fhand.create_dataset(dspath, data=burstslicer)
        print (burstslicer)
        logger.info('Made and saved burstslicer, N:slices %i'%burstslicer.shape[0])

    def writeget_results(self,aRec,plot_clusters=False,mp_mode='map'):
        with h5py.File(aRec.resultsfileH, 'r+') as fhand:


            self.init_from_prev(fhand, aRec.cfg_ana['EdDetection']['groupkey'])
            self.init_from_raw(aRec.rawfileH)
            self.save_basics(aRec)
            self.make_empty_nest()
            self.get_protect_borderspikes()
            # SPS.polarity = 'mix'

            self.collect_sparsespikes()

            spikemat, polinds = self.get_snipmat(self.allsparse)  # polfacs is only needed for mixed

            pcadict = self.do_pca(spikemat)  # is a list when polarity is mixed
            clf = self.make_gmm(pcadict)

            clinds0 = self.predict_clusterind(clf, pcadict)
            ordervec = self.get_ordervec(spikemat, clinds0)
            clinds = self.resort(clinds0, ordervec)

            # plot
            if plot_clusters:
                self.plot_clusterpanel(spikemat, clinds)
                plt.show()

            self.write_spikes_noisespikes(self.allsparse, clinds, polinds)  # polinds only needed for mixed

            #now do it for the windows not scanned yet
            tstart = time.time()
            if mp_mode=='map':
                with mp.ThreadPool(mp.cpu_count()) as pool:
                    subfn = lambda myidx: self.get_project_write_spikes(myidx,pcadict,clf,ordervec)
                    idxlist = list(np.arange(self.stopidx + 1,self.n_slicewins))
                    pool.map(subfn,idxlist)  # stopidx is for the last window examined for the basis clustering

            else:
                with mp.ThreadPool(mp.cpu_count()) as pool:
                    [pool.apply(self.get_project_write_spikes, args=(idx, pcadict, clf, ordervec)) for idx in
                     np.arange(self.stopidx+1,self.n_slicewins)]#stopidx is for the last window examined for the basis clustering

            self.execdur = time.time()-tstart
            self.order_results()



    @property
    def clustfig(self):
        if not hasattr(self,'_clustfig'):
            self.make_clusterfigure()
        return self._clustfig

    def make_clusterfigure(self):
        #def plot_clusts():

        ncols = self.nclust
        self.colorlist = blipsort.get_cluster_colors(nclust=self.nclust)
        if self.polarity == 'mix':
            nrows = 2
            figD, fbottom, ftop = (ncols * 2.5 + 2, 5), 0.12, 0.92  # that is mixed polarity
        
        else:
            nrows = 1
            figD, fbottom, ftop = (ncols * 2.5 + 2, 3), 0.2, 0.85
        
        

        f = plt.figure(facecolor='w', figsize=figD)
        f.subplots_adjust(left=0.08, bottom=fbottom, right=0.98, top=ftop)
        f.suptitle('Polarity: %s' % self.polarity)
        self._clustfig = f
        self.gsMain = gridspec.GridSpec(nrows, ncols)


    def plot_clusterpanel(self,spikemat,clinds):
        if self.polarity in ['pos','neg']:
            polfac = -1 if self.polarity == 'pos' else 1
            self.plot_clusterpanel_(spikemat,clinds,polfac=polfac)
        else:
            assert type(spikemat) == list and type(clinds)==list,'mixed polarity requires a list of spikemats and clusterindices'
            # print (spikemat[1].shape,clinds[1].shape)
            for submat,subinds,polfac in zip(spikemat,clinds,[1,-1]):
                #print ('Plotting clusterpanel - bef %i'%polfac)
                self.plot_clusterpanel_(submat,subinds,polfac=polfac)
                #print ('Plotting clusterpanel %i'%polfac)

    def plot_clusterpanel_(self,spikemat,clinds,polfac=1):
        #datastd = np.std(spikemat)
        y_lim = [scoreatpercentile(spikemat,0.1), scoreatpercentile(spikemat,99.9)]
        #if polfac == -1: y_lim = [-1 * y_lim[1], -1 * y_lim[0]]
        tvec = np.arange(-self.cutwin[0],self.cutwin[1]-1./self.sr,1./self.sr)

        if self.polarity in ['neg','pos'] or (self.polarity=='mix' and polfac==1): row=0
        else: row = 1
        axlist = [self.clustfig.add_subplot(self.gsMain[row,cc]) for cc in range(self.nclust)]

        #print ('Plotting row %i'%row)

        for cc in np.arange(self.nclust):
            ax = axlist[cc]
            clustmat = spikemat[clinds==cc]
            avg = np.mean(clustmat, axis=0)
            ptp = self.get_ptp_snip(avg*polfac)
            for snip in clustmat: ax.plot(tvec,snip,color='grey')
            ax.plot(tvec,avg,self.colorlist[cc],linewidth=2)
            ax.text(0.98, 0.05, 'PTP %1.2f' % (ptp), fontsize=12, color=self.colorlist[cc], transform=ax.transAxes, ha='right',
                    va='bottom', fontweight='bold')
            ax.text(0.98, 0.97, 'N: %d' % (clustmat.shape[0]), fontsize=10, color='k', transform=ax.transAxes, ha='right',
                    va='top')
        
            ax.set_xticks([-0.2,-0.1,-0.,0.1,0.2,0.3,0.4,0.5])
            ax.set_yticks(np.arange(-8,8,1))
            if self.polarity in ['neg','pos'] or row==1: ax.set_xlabel('Time [s]')
            else: ax.set_xticklabels([])
            ax.set_ylim(y_lim)
            if cc==0: ax.set_ylabel('mV')
            else: ax.set_yticklabels([''])
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_xlim([-self.cutwin[0],self.cutwin[1]])


class BurstClassification(edd.Analysis):

    def __init__(self):
        for key, val in list(self.defaults.items()):
            setattr(self, key, val)
        for key, val in list(self.plot_dict.items()):
            setattr(self, key, val)
        self.method = 'Classifying bursts on a given SOM.'
        self.params = ['key_id', 'start', 'stop','clustid','seizidx','bmu']

    def init_from_prev(self,fhand,groupname):
        #group_prev = aRec.cfg_ana['SpikeSorting']['groupkey']
        self.fhand = fhand
        self.burstslicer = fhand[groupname+'/data/burstslicer'].value
        self.n_blocks = self.burstslicer.shape[0]
        self.spikehand = fhand[groupname+'/data/spikes']

    def set_maxdist_mergelim(self,maxdist,mergelim):
        self.maxdist = maxdist
        self.mergelim = mergelim

    @property
    def plot_dict(self):
        return {}


    def init_vars(self,recObj):
        logclipper = lambda x: np.log10(x) if x > 0. else 10. ** -5
        tranfn = lambda logTrue: logclipper if logTrue else np.float

        logger.debug('Transferring SOM-attributes')
        self.setparam('sompath', recObj.som.path)

        mysom = recObj.som._mapdict
        for key in list(mysom.keys()):
            if key == 'dcolors':
                setattr(recObj.som, key, hf.string_decoder(mysom['dcolors']))
            else:
                setattr(recObj.som, key, mysom[key])
        self.vardict = {var: {'transfn': tranfn(logbool), 'fac': fac} for var, logbool, fac in
                   list(recObj.som.som_vars.values())}
        self.transform_value = lambda varname, value: self.vardict[varname]['transfn'](value)

        varnames = [vals[0] for vals in sorted(recObj.som.som_vars.values())]  # you must keep the order!
        logger.info('Constructing feature array. Features: %s' % (str(varnames)))
        self.setparam('features', varnames)


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


    def save_basics(self, recObj):
        logger.info('Saving basics')
        # prepare a saving structure and save some stuff right away
        self.datadict = {'params':self.params}
        self._save_resultsdict(recObj)

    def make_empty_nest(self):
        dspath = '%s/data/%s' % (self.groupkey, 'values')
        if dspath in self.fhand:
            logger.warning('Overwriting dataset %s in %s' % (dspath, self.fhand.filename))
            del self.fhand[dspath]
        nparams = len(self.params)
        self.datahand = self.fhand.create_dataset(dspath, data=np.empty((0,nparams)), maxshape=(np.int(self.spikehand.size/2),nparams))

    def classify_bursts_in_block(self,idx,aRec):
        #get spikes
        sstart,sstop = self.burstslicer[idx]
        myspikes = self.spikehand[sstart:sstop]
        #print (idx,sstart,sstop)
        #print(myspikes.min(),myspikes.max(),idx)
        #detect bursts
        if len(myspikes)>1:
            bursttimes = hf.findmerge_bursts(myspikes,self.maxdist,self.mergelim)
            burstObjs = [eam.Burst(ii+1,start,stop,parentobj=aRec) for ii,[start,stop] in enumerate(bursttimes)]
            # TODO more efficient creation of bustObjs without having to rely on aRec!
            #assign features
            for bobj in burstObjs:
                for feature in self.features:
                    val = self.featfuncs[feature](bobj)
                    setattr(bobj,feature,val)

            #make the roimat
            mapbursts = eam.filter_objs(burstObjs,[lambda x:x.b_n>=self.nmin])
            roimat = np.zeros((len(mapbursts), len(self.features)))
            for bb,burst in enumerate(mapbursts):
                entries = np.array([self.transform_value(var, getattr(burst, var)) for var in self.features])
                roimat[bb] = entries

            # and zscore
            roimat = (roimat - aRec.som.mean_std_mat[0]) / aRec.som.mean_std_mat[1]  # zscoring for the others
            #multiply by weights
            weightfacs = np.array([self.vardict[var]['fac'] for var in self.features])
            self.setparam('weights', weightfacs)
            roimat = roimat * weightfacs
            #project on SOM
            bmus = somify.get_bmus(roimat, aRec.som.weights)
            clusts = aRec.som.clusterids[bmus].astype(np.int)
            sinds = aRec.som.seizidx[bmus]
            for bb, burst in enumerate(mapbursts):
                for attr, val in zip(['rank', 'si', 'bmu'], [clusts[bb], sinds[bb], bmus[bb]]): setattr(burst, attr, val)
            smallbursts = [burst for burst in burstObjs if not burst in mapbursts]
            for bb,burst in enumerate(smallbursts):
                for attr in ['rank','si','bmu']: setattr(burst,attr,np.nan)
            datamat =  np.vstack( [np.hstack([burst.id, burst.roi[0],burst.roi[1],burst.rank,burst.si,burst.bmu]) for burst in eam.timesort_objs(mapbursts+smallbursts)])
            #datamat[datamat == None] = np.float(np.nan)
            datamat[:,0] = datamat[:,0]+self.idcounter
            self.idcounter += datamat.shape[0]
            return datamat.astype(np.float)
        else: return np.empty((0,len(self.params)))


    def writeget_results(self,aRec):
        with h5py.File(aRec.resultsfileH, 'r+') as fhand:
            self.init_vars(aRec)
            self.init_from_prev(fhand, aRec.cfg_ana['SpikeSorting']['groupkey'])
            self.save_basics(aRec)
            self.make_empty_nest()
            self.idcounter = 0

            tstart = time.time()
            with mp.ThreadPool(mp.cpu_count()) as pool:
                for idx in np.arange(self.n_blocks):
                     datamat = self.classify_bursts_in_block(idx, aRec)
                     if datamat.size>0: hf.append_dataset_hdf5(self.datahand, datamat)

                '''def subfn(idx):
                    datamat = self.classify_bursts_in_block(idx, aRec)
                    if datamat.size>0: hf.append_dataset_hdf5(self.datahand, datamat)

                idxlist = list(np.arange(self.n_blocks))
                pool.map(subfn,idxlist)'''

            self.execdur = time.time()-tstart
