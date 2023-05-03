#--------------------------------------------------------------------------- 
# PURPOSE: DETECTION OF EPILEPTIC EVENTS ('BLIPS') IN LFP DATA
# DATE: April 2014
# WRITTEN BY: Katharina Heining
# PYTHON VERSION: 2.7
# MODULES USED: numpy, scipy, matplotlib, gc

# DATA PREREQUISITES: 500 Hz resampled, negative blip polarity

# OUTLINE OF ANALYSIS FLOW: 1. Spectral Method: Average Normalised Power followed by Z-Threshold determination --> blips_spectral
##2. Catching undetected large amplitude blips --> blips_amplitude

# HOW TO USE: To reproduce/understand the individual analysis steps please go through the sub-functions
# contained within the wrapper function 'blip_detect' and/or refer to the manual.

# EXAMPLE:

#    ### here I pass 'the path to save the average power trace ('savepath_power') along the way 
#    ### as keyword argument to the main wrapper function 'blip_detect'
#    savepath = os.path.join(os.sep,'home','user','testsdata','avg_power_test')
#    bliptimes = blipS.blip_detect(data,savepath_power=savepath)

# EXTENSION1 - MASKED ARRAY INUPT: 5. Sept. 2014: 
# data input can also be of type np.ma.core.MaskedArray. intermediate results are also returned as masked arrays (if continuous)
# event times are returned with respect to the time in masked arrays
# purpose: masking your input array is useful when you want to exclude artifacts from calculations while preserving temporal structure

#------------------------------------------------------------------------------ 
#IMPORTS
from __future__ import division
from __future__ import print_function
import numpy as np
import logging

logger = logging.getLogger(__name__)
from core.helpers import open_obj, save_object
#------------------------------------------------------------------------------ 
# THE MAIN WRAPPER FUNCTION

def blip_detect(data, **kwargs):
    '''Procedure from original data to blip-times wrapped in one function
    
    INPUT:
        data: LFP trace, 500 Hz resampled, 1 D array, blips should have negative polarity!
        
        **kwargs
        
            -- General Parameters --
            sr: default 500, Hz, it is highly recommended to use this rate (balance memory-issues/spectral resolution)
            dtime: default 1/12., s, refractory time after each blip, before a new blip will be considered
            
            -- Calculation Average Power --
            window: default 2**7, points, window-width for spectrogram calculation
            norm: default (5,95), tuple of numbers indicating confidence interval for normation or str 'minMax'
            avg_lim: default (4.,40.), Hz, frequency boundaries over which the spectral average will be calculated
            avg_power: 1D array, if already present it will not be calculated again --> calculation is way faster!
                        # For efficiency it is highly recommended to calculate avg_power only once and save it,
                        #if you want to try out several other parameters or generate diagnostic plots later!
            savepath_power: path/under/which/your/avg_power/should/be/saved, in case you want it to be saved
            
            
            
            
            -- Z - Threshold determination by derivative Method --
            thresh_range: default (-0.5,6.), z-scores, interval for which the number of blips will be calculated
            thresh_resolution: default 0.05, z-scores, resolution for the interval thresh_range
            peakperc: default 35., percentage, lower values that will be considered in the estimation of the region 
                            of shallowest slope
            zthresh_mode: firstcross, bump or manual, see respective function
            zthreshPrune: if True (default) for each tested zthresh, the blips will be pruned according to dtime.
                            setting zthreshPrune=False will make the threshold determination faster (about 5s/2hr_rec)
            
            -- Catching undetected large amplitude blips --
            blip_margin: default 0.2, s, region before and after each blip detected with the derivative method that will
                                        #be cut out to yield the compound ea-free-trace subjected to further removal of blips
            amp_thresh: default 4.5, scores, threshold for detection of large amplitude blips
            pol: polarity of your blips can be from ['neg','pos','mix'], default is neg
            
    RETURNS:
        bliptimes: in seconds, 1D array 
            
    '''

    import gc 
    
    sr = kwargs['sr'] if 'sr' in kwargs else 500. # Hz, your data should be 500 Hz, this kwarg exists only
    #for experimental purposes
    dtime = kwargs['dtime'] if 'dtime' in kwargs else 1/12. #seconds
    
    thresh_range = kwargs['thresh_range'] if 'thresh_range' in kwargs else (-0.5,6.) #scores
    thresh_resolution = kwargs['thresh_resolution'] if 'thresh_resolution' in kwargs else 0.05 #scores
    peakperc = kwargs['peakperc'] if 'peakperc' in kwargs else 35. #percentage
    zthresh_mode = kwargs['zthresh_mode'] if 'zthresh_mode' in kwargs else 'firstcross'
    zthreshPrune = kwargs['zthreshPrune'] if 'zthreshPrune' in kwargs else True
    
    blip_margin = kwargs['blip_margin'] if 'blip_margin' in kwargs else 0.2 #seconds
    amp_thresh = kwargs['amp_thresh'] if 'amp_thresh' in kwargs else 4.5 #scores
    pol = kwargs['pol'] if 'pol' in kwargs else 'neg'

    if 'avg_power' not in kwargs:
        logger.info('Calculating Avgerage Power')
        
        window = kwargs['window'] if 'window' in kwargs else 2**7 #points
        norm = kwargs['norm'] if 'norm' in kwargs else (5,95)#percentage
        avg_lim = kwargs['avg_lim'] if 'avg_lim' in kwargs else (4.,40.)

        #calculate the average spectrogram
        sp = get_spectrogram(data,window) # periodogram method
        sp_norm = dynamic_normalisation(sp,norm=norm) #see Waldert 2013
        del sp
        avg_power = average_spectralBins(sp_norm,sr=sr,avg_lim=avg_lim)
        del sp_norm
        gc.collect()
        
        if 'savepath_power' in kwargs:
            logger.info('Saving apower')
            save_object(avg_power,kwargs['savepath_power'],quiet=True)
    else: avg_power = kwargs['avg_power']
        
    avg_power = preprocess_data(avg_power,len(data))
    
    if 'manthresh' in kwargs: 
        zthresh = kwargs['manthresh']
        logger.debug('Using manual threshold')
    else:
        logger.info('Derivative Method: Determining Z-Threshold    mode: '+zthresh_mode)
        fofblip = nblips_asFnOfThreshold(avg_power,thresh_range,thresh_resolution,sr=sr,pruneOn=zthreshPrune,dtime=dtime)
        
        threshvec = np.arange(thresh_range[0],thresh_range[1]+thresh_resolution,thresh_resolution)
        zthresh = getZthresh_derivMethod(fofblip,threshvec,peakperc=peakperc,mode=zthresh_mode,verbose = False)
        logger.debug('Zthresh    %1.2f'%(zthresh))

    bliptimes_spectral = zthreshdetect_blips(avg_power,sr=sr,thresh=zthresh,dtime=dtime)
    logger.debug('# spectral EDs: %d'%(len(bliptimes_spectral)))
    
    logger.debug('Catching Undetected Large Amplitude EDs')
    bliptimes_amp = catch_undetectedBlips(bliptimes_spectral,data,sr=sr\
                    ,zthresh = amp_thresh,blip_margin = blip_margin,dtime=dtime,pol=pol)
    logger.debug('# amp EDs: %d'%(len(bliptimes_amp)))
    return np.sort(np.r_[bliptimes_spectral,bliptimes_amp])

#------------------------------------------------------------------------------ 
# ESTIMATION OF THE AVERAGE POWER SPECTRUM: calculate spectrogram --> normalise --> average

def get_spectrogram(data,ww):
    '''
    Calculates spectromgram of data, using the classic periodogram method
    using a hanning window
    INPUT 
        data: 1D array, real
        ww: int,window with
    OUTPUT
        spgrm: 2D array (timepoints x frequency bins), spectrogram
    '''
    
    tpoints = len(data) - ww + 1
    bins = ww//2 + 1
    spgrm = np.zeros((tpoints, bins))
    

    hann = np.hanning(ww)#
        
    for ii in range(tpoints):
        spgrm[ii,:] = (np.abs (np.fft.rfft (data[ii:ii+ww]*hann, ww))) ** 2. / ww

    if type(data) == np.ma.core.MaskedArray:
        mask_1D = data.mask[ww//2-1:-ww//2]
        mask_2D = np.transpose(np.reshape(np.tile(mask_1D,spgrm.shape[1]),(spgrm.shape[1],len(mask_1D))))
        return np.ma.array(spgrm,mask= mask_2D)
        
    else: return spgrm


def dynamic_normalisation(sp,norm=(5,95),verbose=False):
    '''Normalises spectrogram dynamically for each frequency bin (see Walder 2013)
    INPUT
        sp: 2D array, spectrogram (timepoints x frequency bins)
        norm: default (5,95), percentages,either confidence tuple or 'minMax'
    RETURNS
        sp_norm: 2D array normalised spectrogram (timepoints x frequency bins)
    '''
    
    from scipy.stats import scoreatpercentile
    
    logger.debug('dynamic normalization    '+str(norm))
    if type(sp) == np.ma.core.MaskedArray:
        mask_sp = sp.mask
        sp = sp[mask_sp[:,0]==False]#resect masked snippets, sp contains now the unmasked snippets fused
    
    if norm =='minMax':
        minB = np.min(sp,0)[None,:] # get minimum for each frequency bin along time axis
        maxB = np.max(sp,0)[None,:] # get maximum for each frequency bin along time axis
        sp_norm = (sp-minB)/(maxB-minB) # normalise each frequeny bin to values in [0,1]
        normfacs = np.vstack([maxB,minB])
        
    elif type(norm) in [list,tuple]: # accordingly but with percentiles instead of min and max
        lowscore,highscore = norm[0],norm[1]
        lowInt = np.array([scoreatpercentile(sp[:,ii],lowscore) for ii in range(sp.shape[1])])[None,:]
        highInt = np.array([scoreatpercentile(sp[:,ii],highscore) for ii in range(sp.shape[1])])[None,:]
        sp_norm = (sp-lowInt)/(highInt-lowInt)
        sp_norm[sp_norm<0.]=0.
        sp_norm[sp_norm>1.]=1.
        normfacs = np.vstack([highInt,lowInt])
    
    if 'mask_sp' in locals():
        fullnorm = np.ones(np.shape(mask_sp))
        fullnorm [mask_sp[:,0]==False] = sp_norm # recreate a masked array with locations masked where sp had masks
        normedspec =  np.ma.array(fullnorm,mask= mask_sp)
        
    else: normedspec = sp_norm
    
    if verbose: return normedspec,normfacs
    else: return normedspec


def average_spectralBins(spgrm,sr=500.,avg_lim=(4.,40.)):
    '''Averages spectrogram along spectral axis within given interval
    INPUT
        sprm: 2D array, spectrogram (timepoints x frequency bins)
        *args
        sr: default 500, Hz, sampling rate
        avg_lim: default (4.,40.), Hz, tuple of two, frequency boundaries within which to average
    RETURNS
        avg_power: 1D array averaged power
    '''
    freq_vec = np.linspace(0.,sr/2.,spgrm.shape[1])
    sp = spgrm[:,(freq_vec>avg_lim[0])&(freq_vec<avg_lim[1])]
    
    if type(sp) == np.ma.core.MaskedArray:avg_power = np.ma.mean(sp,1)
    else: avg_power = np.mean(sp,1)
    
    return avg_power



#------------------------------------------------------------------------------ 
# DETECTING BLIPS preprocess --> fofblip: find shallowest region: zthreshold


def preprocess_data(data,newdatalen):
    '''Zscores and paddes data such that it has desired length of data
    INPUT:
        data: 1D array
        newdatalen: int, length to which avg power should be padded (usually len(rawdata))
    RETURNS:
        zscored and padded data, 1 D array
    
    '''
    from scipy.stats import zscore
    
    
    if type(data) == np.ma.core.MaskedArray:
        avg_mask = data.mask
        dataZ = zscore(data[avg_mask==False])
        datadiff = newdatalen - len(avg_mask)
    
    else:
        dataZ = zscore(data)
        datadiff = newdatalen - len(data)
        
    
    padlen,padval,padadd = datadiff//2,0.,np.mod(datadiff,2)
    dataZnPad = np.r_[padval*np.ones((padlen)),dataZ,padval*np.ones((padlen+padadd))]

    if type(data) == np.ma.core.MaskedArray:
        fullavg = np.ones((newdatalen))
        padmask = np.r_[np.zeros((padlen),dtype='bool'),avg_mask,np.zeros((padlen+padadd),dtype='bool')]
        fullavg[padmask==False] = dataZnPad # recreate a masked array with locations masked where sp had masks
        return np.ma.array(fullavg,mask= padmask)
    
    else: return dataZnPad


def zthreshdetect_blips(trace,sr=500.,thresh=0.,pruneOn=True,**kwargs):
    ''' Detects maxima in data trace above the threshold, and takes the middle of the peak plateau, should the
    peak be flat (due to clipping)
    INPUT:
        trace: 1 D array
        *args
        sr: default 500, Hz, sampling rate 
        thresh: default 0, score or absolute value, threshold
        **kwargs: 
            dtime: refractory time which has to elapse after a blip detected until a new blip will be considered
    RETURNS:
        bliptimes [s], 1 D array
    '''
    from scipy.stats import zscore

    if type(trace) == np.ma.core.MaskedArray:
        trace_mask = trace.mask
        trace = trace[trace_mask==False]


    a = zscore(trace)
        
    maxima = np.r_[False, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], False] & np.r_[a[1:]>thresh,False]
    blip_pos = np.where(maxima)[0]

    #peaks are flat when you clip your data above/below a certain level, as in conf-int normation,
    #so these flat 'peaks' are no maxima in the strict sense.
    #in the case of flat peaks the blip will be positioned at the middle of the flat plateau
    try:
        flatpos = np.where(a==a.max())[0]
        firstflat = flatpos[0]+int(np.where(np.diff(flatpos)>1)[0][0])//2
        fflat = flatpos[np.where(np.diff(flatpos)>1)[0]+1]
        fflat_end = np.append(flatpos[np.where(np.diff(flatpos)>1)[0][1:]-1],flatpos[-1])
        center_flat = np.append(firstflat, fflat+(fflat_end - fflat)//2)
        
        blip_pos = np.sort(np.append(blip_pos,center_flat))
    except:
        pass
        

    if 'dtime' in kwargs and kwargs['dtime']>0. and pruneOn:
        bpos = prune(blip_pos,kwargs['dtime']*sr)
    else:
        bpos = blip_pos
    
    if 'trace_mask' in locals(): 
        
        #identify locations of mask in datapoints
        maskpos = np.where(trace_mask==True)[0]
        maskstart = np.r_[maskpos[0],maskpos[np.where(np.diff(maskpos)>1)[0]+1]]
        maskstop = np.r_[maskpos[np.where(np.diff(maskpos)>1)[0]],maskpos[-1]]
        
        gaplens = (maskstop-maskstart)

        newbpos = []
        for blip in bpos:
            for ii in range(len(gaplens)):
                if blip > maskstart[ii]: blip+=gaplens[ii]
            newbpos.append(blip)
                
        bpos = newbpos[:]
    
    return np.array(bpos)/sr
    

def prune(events,dtime,LoggerOn=False):
    '''
    returns array of events where events happening in dtime after any event have been removed
    '''
    parts = np.where(np.diff(events)<dtime)[0]#event indices that are part of a burst
    if np.size(parts)<1: return events
    #logger.debug('Prune-parts %s'%str(parts.shape))
    #beginning and end indx of burstevent
    startpts = np.r_[parts[0],parts[np.where(np.diff(parts)>1)[0]+1]]
    stoppts = np.r_[parts[np.where(np.diff(parts)>1)]+1,parts[-1]+1]
    
    starts,stops, Npts = events[startpts],events[stoppts],stoppts-startpts+1#convert pts to time
    invalids = stops[Npts==2]#obviously in bursts with two elements, the second one happens in dtime
    bursts = np.vstack([starts,stops])[:,Npts>2]#bursts bigger than 2 events get a closer examination
    
    #for every burst, prune those that are closer than dtime to the predecessor
    for start,stop in bursts.T:
        train = events[(events>=start)&(events<=stop)]#events in the burst
        invals = np.array([])
        while len(train)>=2:#recursively walk through train
            invals = np.r_[invals,train[(train-train[0])<dtime][1:]]
            train = train[train>invals[-1]]
        invalids = np.r_[invalids,invals]
    valids = np.sort([event for event in events if not event in invalids])
    #if LoggerOn: logger.debug('N_pruned: %d, Ratio N_kept/N_pruned %1.2f'%(len(invalids),len(valids)/float(len(invalids))))
    return valids



def nblips_asFnOfThreshold(avg_power,thresh_range,resolution,sr=500.,pruneOn=True,**kwargs):
    '''Calculates number of blips as a function of threshold. Used to define z-threshold on avg_power.
    INPUT: 
        avg_power: (1 D array, z-scored)
        thresh_range: tuple of two, z-score range for which the number of blips shall be computed
        resolution: resolution of the z-score range.
        *args:
        sr: default 500., Hz, sampling_rate
        pruneOn: if True dtime will be taken into account for every thresh tested (slower but possibly more accurate)
        **kwargs:
            dtime: refractory time before new blip will be considered, [s]
    OUTPUT:
        number of blips as a function of threshold, 1D array (len: (threshrange[1]-threshrange[0])/resolution)
    
    '''
    
    logger.debug('# EDs as fn of thresh')
    thresh_vec = np.arange(thresh_range[0],thresh_range[1]+resolution,resolution)
    logger.debug('threshmin %1.2f | threshmax %1.2f'%(thresh_vec.min(), thresh_vec.max()))
    if type(avg_power) == np.ma.core.MaskedArray:
        avg_power = avg_power[avg_power.mask==False].data
        
    # for each z-threshold find how many blips are found in the data
    num_blips = lambda zthresh: len(zthreshdetect_blips(avg_power,sr=sr,thresh=zthresh,pruneOn=pruneOn,**kwargs))
    
    return np.array([num_blips(zthresh) for zthresh in thresh_vec ])
    


def getZthresh_derivMethod(fobtrace,threshtrace,peakperc=35.,verbose = False,maxOff=False,mode='bump',**kwargs):
    '''Obtain the z-threshold for blip-detection as the region of shallowest slope in the the nblips-as-fn-of-threshold graph
    INPUT:
        fobtrace: nblips-as-fn-of-threshold vector, 1 D array, corresponds to 'y'-values
        threshtrace: trace of thresholds for fobtrace, 1 D array, corresponds to 'x'-values
        *args:
            peakperc: number between 0 and 100, specifiying which upper percentile of derivatives should be considered
            verbose: bool, specifies number of return parameters
            maxOff: only if plotax is given, if true Nmax will not be written to the panel
            mode: string, determines how the
                threshold is evaluated can be 'bump'(middel of the bump is threshold) 
                or 'firstcross'(first time graph crosses peakperc)
        **kwargs:
            plotax: axis on which a panel illustrating the method will be plotted
            manthresh: float, this will be the value forced on the plot and returned
    RETURNS:
        if verbose: middel of the stability box (the z-threshold suggested), start of stability box, end of stability box
        else: z threshold suggested for blip-detection 
    '''
    
    #print ('getZthresh',mode)
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import MaxNLocator,FixedFormatter,ScalarFormatter
    from matplotlib import pyplot
    
    
    logger.info('Zthresh - derivative method')
    fob_full = fobtrace/float(fobtrace.max())
    #thresh_vec,fob = threshtrace[fob_full>np.unique(fob_full)[0]],fob_full[fob_full>np.unique(fob_full)[0]]#inevitably there are zeros at the end
    conds = (fob_full>np.unique(fob_full)[0]) & (fob_full<np.unique(fob_full)[-1])
    thresh_vec,fob = threshtrace[conds],fob_full[conds]#inevitably there are zeros at the end
    dfob = np.diff(fob)
    maxDiff = np.sort(dfob)[-int(np.ceil(len(dfob)*peakperc/100.))]
    logger.debug('Zthresh -maxdiff %1.4f'%(maxDiff))
    logger.debug('Mode '+mode)
    #identify points where the first derivative crosses it peakperc level
    cross_ids = np.where(dfob>=maxDiff)[0]
    if mode == 'bump':
        cross_ends = np.r_[cross_ids[np.r_[np.diff(cross_ids)>1,False]],cross_ids[-1]]
        cross_starts = np.r_[cross_ids[0],cross_ids[np.where(np.diff(cross_ids)>1)[0]+1]]
        cross_dur = cross_ends-cross_starts
        #where is the longest thresh of being above the peakperc level ...
        stab_pt = (cross_starts[cross_dur==np.max(cross_dur)][0],cross_ends[cross_dur==np.max(cross_dur)][0])#indices where derivative is low
        box_ylim = np.min(fob[stab_pt[0]:stab_pt[1]]),np.max(fob[stab_pt[0]:stab_pt[1]])#for plotting only
        ptidx = np.int(stab_pt[0]+np.ceil((stab_pt[1]-stab_pt[0])/2))
        x_thresh = thresh_vec[ptidx]
        y_thresh = fob[ptidx]
    elif mode == 'firstcross':
        x_thresh = thresh_vec[cross_ids[0]]
        y_thresh = fob[cross_ids[0]]
    elif mode.count('man'):
        x_thresh = kwargs['manthresh']
        y_thresh = fob[np.argmin(np.abs(thresh_vec-x_thresh))]        
    if 'plotax' in kwargs:
        ax = kwargs['plotax']
        ax.plot(threshtrace,fob_full,color='k',lw=2)
        #if kwargs.has_key('manthresh'):
        #    plot_markerLine(ax,x_thresh,axis='vert',marker='>',num=15,color='grey',alpha=0.6)
        #    plot_markerLine(ax,kwargs['manthresh'],axis='vert',marker='>',num=15,color='r',alpha=0.6)
        if mode=='bump':
            p_bbox = FancyBboxPatch((thresh_vec[stab_pt[0]], box_ylim[0]),
                                    thresh_vec[stab_pt[1]]-thresh_vec[stab_pt[0]], box_ylim[1]-box_ylim[0],
                                    boxstyle="square,pad=0.", transform = ax.transData,
                                    ec='grey', fc="grey",alpha=0.6)
            ax.add_patch(p_bbox)
            ax.vlines([thresh_vec[stab_pt[0]],thresh_vec[stab_pt[1]]],box_ylim[0],1.,linestyle = '--',color='grey',alpha=0.7)
        ax.plot(x_thresh,y_thresh,'o',markerfacecolor='k',markeredgecolor='w',markeredgewidth=2,markersize=8)
        ax.plot(x_thresh,y_thresh,'o',markerfacecolor='none',markeredgecolor='r',markeredgewidth=2,markersize=11)
        ax.set_xlim([-0.5,4.5])
        if not maxOff:
            ax.text(0.03,0.96,'Nmax: '+str(fobtrace.max()),color='k',fontsize=13,transform=ax.transAxes)
            ax.text(threshtrace[fob_full==fob_full.min()][0]+0.1,fob_full.min()+0.01,str(fobtrace.min()),color='grey',fontsize=13)
        
        ax2 = ax.twinx()
        ax2.plot(thresh_vec[:-1],dfob,color='g',lw=2)
        ax2.hlines(maxDiff,thresh_vec[0],thresh_vec[-1],color='grey',lw=2,alpha=0.7,linestyle='--')
        ax2.fill_between(thresh_vec[:-1], maxDiff, dfob,color='grey',where = (dfob>=maxDiff),interpolate=True)
        ax2.axvspan(thresh_vec[-1],threshtrace[-1],color='grey',alpha=0.2)
        ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        pyplot.ticklabel_format(axis='y',style='sci',scilimits=(1,2))
        ax2.set_xlim([-0.5,4.5])
        

        ax.set_ylabel('norm(ED-count)')
        ax.set_xlabel('Threshold [z-score]')
        
        ax.spines['right'].set_color('g')
        ax2.tick_params(axis='y', colors='g')
        ax2.set_ylabel('diff(ED-count)',rotation=-90,color='green',labelpad=15)
        mymode = mode if not mode.count('man') else 'man'#for text display in figure
        ax.text(0.8,0.5,'mode: %s\nthr: %1.2f'%(mymode,x_thresh),ha='right',va='center',color='k',transform=ax.transAxes)
        
        
    if verbose:
        return (x_thresh,thresh_vec[stab_pt[0]],thresh_vec[stab_pt[1]])#box middel, start,end --> verbose works only with bump!
    else:
        if 'manthresh' in kwargs: return kwargs['manthresh']
        else:return x_thresh


#------------------------------------------------------------------------------ 
# BLIP DETECT: HONING: CATCH LARGE AMPLITUDE BLIPS OVERLOOKED BY THE SPECTRAL METHOD

def catch_undetectedBlips(bliptimes,rawdata,sr=500.,zthresh = 4.5,blip_margin = 0.2,dtime=1./12.,pol='pos',**kwargs):
    from scipy.stats import zscore
    '''Finds blips that have been overlooked by the avg-power method
    METHOD: remove region around blip with blip_margin (secs) distance from indiviual blip for each blip,
    zscore the whole remaning data set and detect where zthresh (in scores) is crossed,
    thereby detecting large amplitude blips that have been overlooked
    INPUT:
        bliptimes: bliptimes from avg_power method [s], 1D array
        rawdata: 500 Hz original data trace(blips should have positive polarity!), 1Darray
        *args
            sr: sampling_rate [Hz]
            zthresh: z-threshold above which peaks will be considered blips in the supposedly free data
            blip_margin: margin around blip that shall be removed to jield a patchwork of free data snippets [s]
            dtime: refractory time that has to elapse till a new blip will be considered
        **kwargs:
            plotax: axis on which a panel illustrating the method will be plotted
    RETURNS
        bliptimes2: additional blips found, 1D array, [s]
    '''
    minibi = 2*blip_margin #minmal duration of a blip-free period to be taken into account
    
    
    #if rawdata input is a masked array(artifacts):
    #### 1.remove the masked areas from the rawdata (yielding a concatenated effectively unmasked array)
    #### 2. adapt the bliptimes such that they match the unmasked concatenated rawdata
    if type(rawdata) == np.ma.core.MaskedArray:
        data_mask = rawdata.mask
        rawdata = rawdata[data_mask==False]#reshrink data to mask-free version
        
        maskpos = np.where(data_mask==True)[0]
        maskstart = np.r_[maskpos[0],maskpos[np.where(np.diff(maskpos)>1)[0]+1]]/sr
        maskstop = np.r_[maskpos[np.where(np.diff(maskpos)>1)[0]],maskpos[-1]]/sr
        
        gaplens = (maskstop-maskstart)
        
        #get blips for time points in mask-free version
        bliptimes = np.array([blip- np.sum(gaplens[blip>maskstop]) for blip in bliptimes])

    #copied from blip_detect.find_free
    if np.size(bliptimes)==0:
        freefused,timefused = zscore(rawdata),np.linspace(0.,len(rawdata)/sr,len(rawdata))
    else:
        ibis = np.diff(bliptimes)
        free_start = bliptimes[np.r_[ibis>minibi,False]]+blip_margin
        free_stop = bliptimes[np.where(ibis>minibi)[0]+1]-blip_margin
        frees = [rawdata[int(start):int(stop)] for start,stop in zip(free_start*sr,free_stop*sr)]
        times = [np.linspace(start,stop,len(frees[ii])) for ii,(start,stop) in enumerate(zip(free_start,free_stop))]
        freefused,timefused =  zscore(np.hstack(frees)),np.hstack(times)
    
    if pol =='pos':crossers = np.where(freefused>zthresh)[0]
    elif pol=='neg':crossers = np.where(freefused*-1>zthresh)[0]
    elif pol=='mix':
        crossersPos = np.where(freefused>zthresh)[0]
        crossersNeg = np.where(freefused*-1>zthresh)[0]
        crossers = np.unique(np.r_[crossersPos,crossersNeg])
        
    
    if len(crossers)==0:
        bliptimes2 = np.array([])
    else:
        moreblips = np.r_[crossers[0],crossers[np.where(np.diff(crossers)>int(dtime*sr))[0]+1]]
        bliptimes2 = timefused[moreblips]
        
        if pol == 'mix' and not np.size(bliptimes)==0:#can be that the slow wave after an already detected spike gets detected too
            maxdiff = 0.5#maximal diff in seconds between an ampblip and its predecessor to be considered mockblip
            cr = 0.05#range around bliptime to be considered for polarity check
            polarity = lambda snip: np.sign(np.abs(snip.max())-np.abs(snip.min()))
            btimes = np.sort(np.r_[bliptimes2,bliptimes])
            prec_blips = np.array([np.max(btimes[btimes<ablip]) for ablip in bliptimes2 if not np.size(btimes[btimes<ablip])==0])
            bliptimes2 = bliptimes2[len(prec_blips)-len(bliptimes2):]#in case first one has no ablip
            mockblips = []
            for ablip,pblip in zip(bliptimes2,prec_blips):
                #cond1: polarities between predecessor and amplitude blip must be identical if they are close enough together(maxdiff)
                cond1 = polarity(rawdata[np.int((ablip-cr)*sr):np.int((ablip+cr)*sr)]) == polarity(rawdata[np.int((pblip-cr)*sr):np.int((pblip+cr)*sr)])
                cond2 = (ablip-pblip)>= maxdiff
                if not cond1 and not cond2: mockblips.append(ablip)
            bliptimes2 = np.array([blip for blip in bliptimes2 if not blip in mockblips])
            
    
    if 'maskstart' in locals(): #that is if the original rawdata input was masked
        #rescale bliptimes2 such that they match the originally masked rawdata input
        ampblips_mask = []
        for blip in bliptimes2:
            for ii in range(len(gaplens)):
                if blip > maskstart[ii]: blip+=gaplens[ii]
            ampblips_mask.append(blip)
                
        bliptimes2 = np.array(ampblips_mask)
    
           
        
    if 'plotax' in kwargs:
        ax = kwargs['plotax']
        
        hist_list,bin_list = [],[]
        newbliptimes = np.sort(np.r_[bliptimes2,bliptimes])#original bliptimes+newly found bliptimes
        for bltimes in [bliptimes, newbliptimes]:#calculate histograms of z-scored amplitude values for the original bliptimes
            #and original bliptimes+newly found bliptimes
            frees = get_frees(bltimes,blip_margin,rawdata,sr=sr)
            freefused =  zscore(np.hstack(frees))
            mybins = np.int(np.sqrt(len(freefused))/3.)
            myhist,mybins = np.histogram(freefused,mybins)
            yhist = myhist/float(len(freefused))
            hist_list.append(yhist)
            bin_list.append(mybins)
        
        nnew,nold = len(bliptimes2),len(bliptimes)#number of additionally found blips and orignial blips
        ymin,ymax = 10**-7,hist_list[0].max()+10**-2


        ax.plot(bin_list[0][:-1],hist_list[0],'k',lw=2)
        ax.plot(bin_list[1][:-1],hist_list[1],'g',lw=2,alpha=0.7)
        if pol=='pos' or pol=='mix':
            ax.vlines(zthresh,ymin,ymax,'r',linestyle='--',linewidth=3,alpha=0.6)
        if pol=='neg'or pol=='mix':
            ax.vlines(-1*zthresh,ymin,ymax,'r',linestyle='--',linewidth=3,alpha=0.6)
            
        infostr = 'threshold: %1.2f z \n #(amp-EDs): %d \n --> %1.2f%% ED incr.'%(zthresh,nnew,100*nnew/float(nold+nnew))
        ax.text(0.99,0.99,infostr,transform=ax.transAxes,ha='right',va='top',color='r')
        ax.text(0.01,0.99,'free before',color = 'k',transform=ax.transAxes,ha='left',va='top')
        ax.text(0.01,.89,'free after',color = 'g',transform=ax.transAxes,ha='left',va='top')
        
        ax.set_yscale('log')
        ax.set_ylim(ymin,ymax)
        x1,x2 = np.min(np.hstack(bin_list)),np.max(np.hstack(bin_list))
        frees = np.hstack(get_frees(bltimes,blip_margin,rawdata,sr=sr))
        ax.set_xlim([x1,x2])
        
        #the actual mV amplitude scaleing
        ax2 = ax.twiny()
        ax2.set_xlim([x1*frees.std()-frees.mean(),x2*frees.std()-frees.mean()])
        ax2.set_xlabel('Amplitudes before [mV]')
        ax2.tick_params(axis='x', which='major')
        
        #if pol=='pos':ax.set_xlim([bin_list[1].min(),bin_list[0].max()])
        #elif pol=='neg':ax.set_xlim([bin_list[1].min(),bin_list[0].max()])
        #elif pol=='mix':ax.set_xlim([np.min(np.hstack(bin_list)),np.max(np.hstack(bin_list))])
        ax.set_xlabel('Amplitudes (z-score)')
        ax.set_ylabel('Probability')
        
        
    return bliptimes2


def get_frees(bliptimes,blip_margin,rawdata,sr=500.):
    '''
    Extract EA-Free region from datatrace.
    INPUT
        bliptimes: [s], 1D array
        blip_margin: margin around blip that shall be removed to jield a patchwork of free data snippets [s]
        rawdata: original data trace(blips should have positive polarity!), 1Darray
        
        *args
            sr: sampling rate, Hz
    RETURNS
        list of 1D arrays, each array being a ea-free snippet
    '''
    
    minibi = 2*blip_margin
    ibis = np.diff(bliptimes)
    free_start = bliptimes[np.r_[ibis>minibi,False]]+blip_margin
    free_stop = bliptimes[np.where(ibis>minibi)[0]+1]-blip_margin
    frees = [rawdata[int(start):int(stop)] for start,stop in zip(free_start*sr,free_stop*sr)]
    return frees


#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#DISPLAY FOR CHECKING THE METHOD --> Spectral figure in the manual
def plot_spectrogramExample(data,zthresh,tbound,sr=500.,**kwargs):
    ''' Creates a figure for an example cutout of data comprising normalised spectrogram (bottom),
        average normalised psd (middle) and raw trace with blips marked (top).
        INPUT:
            data: lfp data 1D array
            zthresh: zthreshold for the average normalised psd, [scores]
            tbound: tuple of two, (start of cutout, stop of cutout) [s]
            *args:
                sr: sampling rate, default 500. [Hz]
            **kwargs:
                dtime: default 1/12., s, refractory time after each blip, before a new blip will be considered
                window: default 2**7, points, window-width for spectrogram calculation
                norm: default (5,95), tuple of numbers indicating confidence interval for normation or str 'minMax'
                avg_lim: default (4.,40.), Hz, frequency boundaries over which the spectral average will be calculated
        
        RETURNS: figure object
    '''
    import matplotlib
    from matplotlib.pyplot import figure
    import gc
    
    tstart,tstop = tbound
    pstart,pstop = int(tstart*sr),int(tstop*sr)
    
    avg_lim = kwargs['avg_lim'] if 'avg_lim' in kwargs else (4.,40.)
    llim,ulim = avg_lim
    window = kwargs['window'] if 'window' in kwargs else 2**7 #points
    norm = kwargs['norm'] if 'norm' in kwargs else (5,95)#percentage
    dtime = kwargs['dtime'] if 'dtime' in kwargs else 1/12. #seconds
        
    data2  = data[pstart:pstop]
    
    
    sp = get_spectrogram(data,window)
    sp_norm = dynamic_normalisation(sp,norm=norm)
    del sp # take care of your memory, if it is low
    avg_power = average_spectralBins(sp_norm,sr=sr,avg_lim=avg_lim)
    sp_norm = sp_norm[pstart:pstop]
    
    avg_power = preprocess_data(avg_power,len(data)) 
    bliptimes = zthreshdetect_blips(avg_power,sr=sr,thresh=zthresh,dtime=dtime)

    avg_power = avg_power[pstart:pstop]
    bliptimes = bliptimes[(bliptimes>tstart) & (bliptimes<tstop)]-tstart

    tvec = np.linspace(0.,len(data2)/sr,len(data2))
    freq_vec = np.linspace(0.,sr/2.,sp_norm.shape[1])
    
    #preparing subplot positions
    #xstart,xlen = 0.07,0.85
    xstart,xlen = 0.1,0.85
    yup,ylow = 0.9,0.1
    ylen = 0.15
    ylen_imag = yup-ylow-2*ylen
    
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14) 
        
    f = figure(facecolor='w',figsize=(10,7))
    rax = f.add_axes([xstart, ylen_imag+ylow+ylen,xlen,ylen])#axes for upper panel: raw trace
    meax = f.add_axes([xstart, ylen_imag+ylow,xlen,ylen])#axes for medium panel: avg power
    imax = f.add_axes([xstart,ylow,xlen,ylen_imag])#lowest panel: spectrogram
    
    #spectrogram
    im = imax.imshow(sp_norm.transpose(),cmap='gray',origin='lower',aspect='auto',interpolation= 'nearest',filternorm=None)#scaleFactor
    im.set_extent([tvec.min(),tvec.max(),freq_vec.min(),freq_vec.max()])
    imax.plot(tvec[::100],np.ones(len(tvec[::100]))*ulim,'vr',markersize = 6,markeredgecolor='none')
    imax.plot(tvec[::100],np.ones(len(tvec[::100]))*llim,'^r',markersize = 6,markeredgecolor='none')
    
    imax.set_xticks(np.arange(int(tvec.min())+2,int(tvec.max())+1,2))
    #imax.set_xticklabels(np.arange(1,20,1))
    imax.set_xlabel('Time [s]')
    imax.set_ylabel('Frequency [Hz]')
    imax.set_ylim([freq_vec.min(),freq_vec.max()])
    
    #rawtrace
    rax.plot(tvec,data2,'k',lw=2)
    mylim = rax.get_ylim()
    if len(bliptimes)>0:rax.vlines(bliptimes,1.,mylim[1],'r',linewidth=2)#mylim[0]
    rax.text(0.01,0.85,'[{0},{1}]'.format(tstart,tstop),transform=rax.transAxes,color='k')
    rax.axis('off')
   
    
    #average psd
    meax.plot(tvec,avg_power,'b',lw=2)
    meax.hlines(zthresh,tvec.min(),tvec.max(),'r',linestyle='--',linewidth=2)
    meax.plot(bliptimes,avg_power[[int(bb) for bb in bliptimes*sr]],'o',markersize = 6,markeredgecolor='r',markeredgewidth=2,markerfacecolor='w')

    meax.text(tvec[len(tvec)//60],avg_power.max()-0.1*(np.abs(avg_power.max())-np.abs(avg_power.min()))\
              ,'avg normalised psd ',color='b')
    meax.text(tvec[len(tvec)//40], zthresh, 'Zthresh {0}'.format(np.around(zthresh,2)), color = 'r',\
         ha="left", va="center",bbox = dict(ec='1',fc='1'))
    meax.axis('off')

    for myax in f.get_axes(): myax.set_xlim([tvec.min(),tvec.max()])

    del sp_norm
    gc.collect()
    
    return f



def plot_blipsDetectedExamples(bt_spectral,bt_amp,data,snipdur=20.,sr=500., **kwargs):
    from matplotlib.pyplot import figure,subplots
    from matplotlib.ticker import MaxNLocator
    from scipy.stats import zscore

    data_mean,data_std = np.mean(data),np.std(data)
    data = zscore(data)
    
    n_panels = kwargs['n_panels'] if 'n_panels' in kwargs else  5
    t_offset = kwargs['t_offset'] if 't_offset' in kwargs else  0.
    
    #tvec = np.arange(t_offset,len(data)/sr+t_offset,1./sr)
    tvec = np.linspace(t_offset,len(data)/sr+t_offset,len(data))
    
    starts = kwargs['starts'] if 'starts' in kwargs \
            else sorted(np.random.choice(tvec[:np.int(-sr*snipdur)],size=n_panels))

    ampcol,speccol = 'r','b'
    
    f, axarr = subplots(n_panels, sharey=True,facecolor='w',\
                            figsize=(16,10))
    f.subplots_adjust(left=0.05,right=0.96,top=0.92,bottom=0.08)
    f.text(0.01,0.99,'spectral EDs',color=speccol,fontsize=15,ha='left',va='top')
    f.text(0.01,0.96,'amplitude EDs',color=ampcol,fontsize=15,ha='left',va='top')
    f.text(0.99,0.99,'artifacts',fontweight='bold',color='khaki',fontsize=15,ha='right',va='top')
    
    for ii,start in enumerate(starts):
        ax = axarr[ii]
        tsnip = tvec[(tvec>=start)& (tvec<=start+snipdur)]
        ttrue = (tvec>=start) & (tvec<=start+snipdur)
        ax.plot(tsnip,data[ttrue],'k',lw=2)
        for events,col in zip([bt_spectral,bt_amp],[speccol,ampcol]):
            try:
                ax.vlines(events[(events>=start)&(events<=start+snipdur)],9.5,11.5,color=col,linewidth=2)
            except:
                pass
        if 'artregions' in kwargs:
            maskmat = kwargs['artregions']
            myarts = np.array([[astart,astop] for [astart,astop] in maskmat.T if start<astop and start+snipdur>astart])
            if np.size(myarts)>0:
                for myart in myarts:
                    ax.axvspan(myart[0],myart[1],facecolor='khaki',alpha=0.4,edgecolor='khaki')
                
        ax.set_ylabel('z',labelpad=-5)  
        y1,y2 = -10,12
        ax.set_ylim([y1,y2])
        ax.set_xlim([tsnip.min(),tsnip.max()])
        #ax.set_yticklabels([])
        if ii==n_panels-1:
            ax.set_xlabel('Time [s]')
            
        ax2 = ax.twinx()
        ax2.set_ylim([y1*data_std-data_mean,y2*data_std-data_mean])
        ax2.set_ylabel('mV',rotation=-90,labelpad=10)
        for myax in [ax,ax2]:myax.yaxis.set_major_locator(MaxNLocator(5))
    return f

def plot_blipsDetected(bt_spectral,bt_amp,data,sr=500., **kwargs):
    from matplotlib.pyplot import figure
    import matplotlib.collections as collections
    from scipy.stats import zscore
    
    data = zscore(data)
    

    t_offset = kwargs['t_offset'] if 't_offset' in kwargs else  0.

    ampcol,speccol = 'r','b'
    
    f = figure(facecolor='w',figsize=(16,4))
    f.subplots_adjust(left=0.05,right=0.97,top=0.87,bottom=0.14)
    f.text(0.01,0.95,'spectral EDs',color=speccol,fontsize=15)
    f.text(0.01,0.88,'amplitude EDs',color=ampcol,fontsize=15)
    
    tvec = np.arange(t_offset,len(data)/sr+t_offset,1./sr)
    

    ax = f.add_subplot(111)
    ax.plot(tvec,data,'k',lw=2)
    for events,col in zip([bt_spectral,bt_amp],[speccol,ampcol]):
        ax.vlines(events,9.5,11.5,color=col,linewidth=2)

    if 'artregions' in kwargs:
        maskmat = kwargs['artregions']
        if np.size(maskmat)>0:
            ax.hlines(0.,maskmat[0],maskmat[1],color='DarkViolet',alpha=0.5,linewidth=20,zorder=1)
            
      
    ax.set_ylim([-10,12])
    ax.set_xlim([tvec.min(),tvec.max()])
    #ax.set_yticklabels([])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [scores]')
    return f

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
# ACCESSORY FUNCTIONS



def plot_markerLine(ax,val,axis='vert',marker='>',num=30,color='r',alpha=1.):
    '''plots a vertical or horizontal line using a marker
    INPUT:
        ax: axes object
        val: value where to plot the line
        *args:
            axis: 'vert' or 'hor', vertical or horizontal
            marker: type of marker to be used
            num: number of markers to appear in line
            color: color (can be anything from str to hex, see pylab.plot kwarg)
            alpha: alpha value
            '''
    if axis=='vert':
        my_lim = ax.get_ylim()
        yvec = np.linspace(my_lim[0],my_lim[1],num)
        xvec = val*np.ones(len(yvec))
    elif axis=='hor':
        my_lim = ax.get_xlim()
        xvec = np.linspace(my_lim[0],my_lim[1],num)
        yvec = val*np.ones(len(xvec))
        
    ax.plot(xvec,yvec,marker=marker,ls='none',markeredgewidth=1\
            ,markerfacecolor=color,markeredgecolor=color,markersize=5,alpha=alpha)
    
    if axis=='vert':
        ax.set_ylim(my_lim)
    elif axis=='hor':
        ax.set_xlim(my_lim)

 
def get_methdict(localvars):
    from time import ctime
    
    date = ctime()
    localvars['date'] = date
    
    varlist = ['avg_lim','wakeup_time','norm','window','sr','thresh_range','thresh_resolution',\
               'peakperc','dtime','amp_thresh','blip_margin','date','mode']
    
    methdict = {k: v for (k,v) in list(localvars.items()) if k in varlist}
    return methdict


def plot_differentThreshs(plotax,threshtrace,fofblip,allthreshs,threshcols,**kwargs):
    from matplotlib.ticker import MaxNLocator,FixedFormatter,ScalarFormatter
    from matplotlib import pyplot
  
    thresh_vec,fob = threshtrace[fofblip>np.unique(fofblip)[0]],fofblip[fofblip>np.unique(fofblip)[0]]#inevitably there are zeros at the end
    dfob = np.diff(fob)
    if 'peakperc' in kwargs:
        peakperc = kwargs['peakperc']
        maxDiff = np.sort(dfob)[-int(np.ceil(len(dfob)*peakperc/100.))]
    
    plotax.plot(threshtrace,fofblip,'k',lw=2)
    plotax.vlines(allthreshs,0.,1.,color=threshcols,lw=2)
    for ii in range(len(allthreshs)-1):
        plotax.fill_betweenx([0.,1.], allthreshs[ii], allthreshs[ii+1],alpha=0.4,color=threshcols[ii])
    ax2 = plotax.twinx()
    ax2.plot(thresh_vec[:-1],dfob,color='grey',lw=2)
    if 'maxDiff' in locals():
        ax2.hlines(maxDiff,thresh_vec[0],thresh_vec[-1],color='grey',lw=2,alpha=0.7,linestyle='--')
        ax2.fill_between(thresh_vec[:-1], maxDiff, dfob,color='grey',where = (dfob>=maxDiff),interpolate=True)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    pyplot.ticklabel_format(axis='y',style='sci',scilimits=(1,2))
    ax2.set_xlim([-0.5,4.5])
    
    plotax.set_ylabel('#(spikes)')
    plotax.set_xlabel('Threshold (scores)')
    ax2.set_ylabel('d(spikes)',rotation=-90,color='grey',labelpad=20)


def plot_threshsAndBlips(data,threshtrace,fofblip,allthreshs,threshcols,allblips,sr=500.,snipdur=20.,t_offset=0.):
    from matplotlib.pyplot import figure
    from scipy.stats import zscore
    
    zdata = zscore(data)
    
    snipstarts = np.sort(np.random.rand(5)*(len(zdata)/sr-t_offset))+t_offset
    
    xstart,ystart,xlen,ylen,xmax = 0.07,0.8,0.2,0.15,0.92
    
    tvec = np.linspace(0,len(zdata)/sr,len(zdata))
    f = figure(figsize=(16,10),facecolor='w')
    fobax = f.add_axes([xstart,ystart,xlen,ylen])
    plot_differentThreshs(fobax,threshtrace,fofblip,allthreshs,threshcols,peakperc=35.)
    
    fobax.set_ylabel('#(Spikes)')
    fobax.xaxis.set_label_position('top') 

    for jj,mystart in enumerate(snipstarts):
        if jj==0: 
            xxstart = xstart+xlen+0.05
            thisdur = snipdur/(xmax-xstart)*(xmax-xxstart)
        else: 
            xxstart = xstart
            thisdur = snipdur
        ax = f.add_axes([xxstart,ystart-(ylen+0.035)*jj,xmax-xxstart,ylen])
        tsnip = tvec[(tvec>=mystart)& (tvec<=mystart+thisdur)]
        ttrue = (tvec>=mystart) & (tvec<=mystart+thisdur)
        ax.plot(tsnip,zdata[ttrue],'k',lw=2)
        for events,col in zip(allblips,threshcols):
            try:
                ax.vlines(events[(events>=mystart)&(events<=mystart+thisdur)],5.5,7.5,color=col,linewidth=2)
            except:
                pass
        ax.set_ylim([-10,8])
        ax.set_xlim([tsnip.min(),tsnip.max()])
        ax.set_yticks([-5,0,5])
        if jj==2:
            ax.set_ylabel('V-Amp [scores]')
        if jj ==4:
            ax.set_xlabel('Time [s]')
    
    return f


#------------------------------------------------------------------------------ 
#TRYJARD

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#JUNKJARD
def purge_blips(raw_blips,waste_space):
    '''Remove  detected blips that are too close to the next blip, thereby introducing a refractory period,
    INPUT:
        raw_blips: 1D array, points 
        waste_space: int, points, refractory time (in points) to be introduced 
    RETURNS:
        blip_pts: 1D array, points, pruned blip positions 
        
    '''
    if len(raw_blips) == 0: return []
    
    blip_pts = [raw_blips[0]]
    
    
    more_blips = True
    while more_blips == True:
        try:
            next_blip = raw_blips[np.where(raw_blips>blip_pts[-1]+waste_space)][0]
            blip_pts.append(next_blip)
        except:
            more_blips = False  
    
    return blip_pts













