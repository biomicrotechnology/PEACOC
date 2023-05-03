from __future__ import division
from __future__ import print_function

from six.moves import input


import logging
import numpy as np
import time
import io
import sys
import os
import pickle
import h5py

logger = logging.getLogger(__name__)


# FUNCTIONS FOR BURST DETECTION
def findmerge_bursts(spiketimes, interval_threshold,maxint=None):
    isis = np.diff(spiketimes)
    spike_parts = np.where(isis < interval_threshold)[0]
    spike_starts = np.append(spike_parts[0], spike_parts[np.where(np.diff(spike_parts) > 1)[0] + 1])
    spike_stops = np.append(spike_parts[np.where(np.diff(spike_parts) > 1)] + 1, spike_parts[-1] + 1)
    bstarts = spiketimes[spike_starts]
    bstops = spiketimes[spike_stops]
    startstop_array = np.vstack([bstarts, bstops]).T
    if maxint is None or np.size(startstop_array) <= 2: return startstop_array
    else:
        starts = startstop_array[:, 0]
        stops = startstop_array[:, 1]
        intervals = starts[1:] - stops[:-1]

        startinds = np.r_[0, np.where(intervals >= maxint)[0] + 1]
        stopinds = np.r_[np.where(intervals >= maxint)[0], len(stops) - 1]

        newstarts = starts[startinds]
        newstops = stops[stopinds]
        return np.vstack([newstarts, newstops]).T


def find_bursts(spiketimes,maxdist):
    '''input: times when blips occured, threshold for maximum ibi
    returns dictionary with start-times and stop-times of bursts, and number of blips within bursts
    '''

    isis = np.diff(spiketimes)
    burst_parts = np.where(isis<maxdist)[0]
    burst_starts = np.append(burst_parts[0],burst_parts[np.where(np.diff(burst_parts)>1)[0]+1])
    burst_stops = np.append(burst_parts[np.where(np.diff(burst_parts)>1)]+1,burst_parts[-1]+1)
    
    burst_dict = {}
    burst_dict['start'] = spiketimes[burst_starts]
    burst_dict['stop'] = spiketimes[burst_stops]
    burst_dict['blips'] = burst_stops-burst_starts+1
    return burst_dict


def mergeblocks(blocklist,output='both',**kwargs):
    '''
    blocklist: items of format 2 x n, n is the number of regions blocked
                item[0]:starts, item[1]:stops
                usage example: [artifactTimes,seizureTimes]
    output: options 'both','block','free'
    
    
    EXAMPLE
        artTimes1 = np.array([[300.,400.],[550.,700.],[1000.,1100.],[5000.,5600.]]).T
        artTimes2 = np.array([[410.,460.],[690.,720.],[740.,790.],[800.,1050.],[1060.,1070.]]).T
        blocklist = [artTimes1,artTimes2]
        blockT,freeT = sa.mergeblocks(blocklist,output='both',t_start=0.,t_offset=600.,t_stop=5000.)
    '''

    t_start = kwargs['t_start'] if 't_start' in kwargs else None
    t_stop = kwargs['t_stop'] if 't_stop' in kwargs else None
    t_offset = kwargs['t_offset'] if 't_offset' in kwargs else None

    if 't_start' in kwargs and 't_offset' in kwargs:
        if t_start == t_offset:offsetTime = np.array([[],[]])
        else:offsetTime = np.array([[t_start,t_offset]]).T
        blocklist.append(offsetTime)
        blocklist = [np.clip(item,t_start,t_stop) for item in blocklist]


    blockmat = np.hstack([block for block in blocklist if np.size(block)>0]) 
    blockmatS = blockmat[:,np.argsort(blockmat[0])]
    cleanblock = np.array([[],[]])
    nextidx = 0
    while True:
        start = blockmatS[0,nextidx]
        stop = blockmatS[1,nextidx]
        for istart,istop in blockmatS.T:
            if (istart>=start) & (istart<=stop) & (istop>=stop):stop = istop
        
        #print start,stop
        newvals = np.array([start,stop])[:,None]
        cleanblock = np.hstack([cleanblock,newvals])
        #print nextidx
        if np.size(np.where(blockmatS[0]>stop))==0:break
        nextidx = np.where(blockmatS[0]>stop)[0][0]
    all_starts,all_stops = cleanblock

    #all_starts,all_stops = all_starts[all_starts<=all_stops],all_stops[all_starts<=all_stops]
    blockmat = np.vstack([all_starts,all_stops])
    freemat = np.vstack([all_stops[:-1],all_starts[1:]])
    

    if 't_start' in kwargs:
        if t_start<all_starts[0]: freemat = np.hstack([np.array([[t_start],[all_starts[0]]]),freemat])

    if 't_stop' in kwargs: 
        if t_stop>all_stops[-1]: freemat = np.hstack([freemat,np.array([[all_stops[-1]],[t_stop]])])

    if output == 'both': return blockmat,freemat
    elif output == 'block': return blockmat
    elif output == 'free': return freemat


def read_burstdata(burstmat,params):

    cvrt = lambda val, datatype: [None] if np.isnan(val) else [val.astype(datatype)]

    cdict = {np.int(vals[0]): [[vals[1], vals[2]]] + cvrt(vals[3], 'int') + cvrt(vals[4], 'float') + cvrt(vals[5],'int')\
             for vals in burstmat}
    cdict.update({'params': ['roi_int'] + string_decoder(params[3:])})
    return cdict


def convert_burstdict(cdict):
    datamat = np.vstack([np.r_[key, np.array(val[0]), np.array(val[1:])] for key, val in cdict['data'].items() if
                         not key == 'params'])
    datamat[datamat == None] = np.nan
    # newdatadict = {np.int(datavec[0]):[datavec[1:3],np.array([np.int(datavec[3]),datavec[4],np.int(datavec[5])])] for datavec in datamat}
    # newdatadict = {str(np.int(datavec[0])):np.r_[datavec[1:3],np.array([np.int(datavec[3]),datavec[4],np.int(datavec[5])])] for datavec in datamat}

    # newdatadict['params'] = cdict['data']['params']
    paramvals = ['key_id', 'start', 'stop'] + cdict['data']['params'][1:]
    #paramvals = np.array([paramvals])
    neworig = {key: cdict[key] for key in ['info', 'methods']}

    neworig['data'] = {'values': datamat.astype('float'),
                       'params': paramvals}

    return neworig

#------------------------------------------------------------------------------ 
#filtering

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from scipy.special import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = list(range(order+1))
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**ii for ii in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


#------------------------------------------------------------------------------ 
#resampling
def smartwins(datalen,winpts,overfrac=1/6.,pow2=True):
    ww = 2**winpts.bit_length() if pow2 else np.int(winpts)
    overest = np.int(ww*overfrac)
    overlap = overest+1 if np.mod(overest,2) else np.int(overest)#make the overlap divisible by 2    
    nwins = (datalen-ww)//(ww-overlap)+1
    winstarts = np.arange(0,nwins*ww,ww)
    winstarts = winstarts[:] if overfrac == 0 else winstarts-np.arange(0.,overlap*nwins,overlap)
    winarray = np.vstack([winstarts,winstarts+ww])
    lastdiff = datalen-winarray[1][-1]
    last_ww = 2*ww if lastdiff>(ww-overfrac) else np.int(ww)    
    lastwin = np.array([datalen-last_ww,datalen])
    winarray = np.hstack([winarray,lastwin[:,None]]).astype(np.int)
    if (winarray[0][-1]-winarray[1][-2])>0:logger.error('last window doesnt overlap')
    return winarray.T


def resample_portions(data,winarray,sr,new_sr):
    '''winarray is n x 2: start,stop of datacutout in points'''
    from scipy.signal import resample
    
  
    rate_ratio = sr/new_sr
    
    
    #tlist = []
    logger.debug('resampling subwindows')
    resampled_list = []
    for start,stop in winarray:
        #print start,stop
        snip = data[start:stop]
        sniplen = stop-start
        resampled = resample(snip,np.int(sniplen/rate_ratio))
        resampled_list.append(np.squeeze(resampled))
        #tlist.append(np.linspace(start/sr,stop/sr,len(resampled)))
    
    #now fuse together
    logger.debug('fusing overlapping windows')
    overlap = winarray[0][1]-winarray[1][0]
    ww = np.diff(winarray[0])
    logger.debug('ww: %1.2f, overlap: %1.2f '%(ww,overlap))
    nwins = winarray.shape[0]
    firstind = np.array([0,float(ww-overlap*0.5)])
    indbody = np.tile(np.array([float(0.5*overlap),float(ww-0.5*overlap)]),(nwins-2,1))
    inds = (np.vstack([firstind,indbody])*new_sr/sr).astype(np.int) 
    fused0 = np.hstack([resampled_list[ii][start:stop] for ii,[start,stop] in enumerate(inds)])
    missing_pts = np.int(winarray[-1][1]*new_sr/sr-len(fused0))
    fused = np.r_[fused0,resampled_list[-1][-missing_pts:]]
    
    #check whether durations match
    datadur = len(data)/sr
    fuseddur = len(fused)/new_sr
    if np.isclose(datadur,fuseddur,atol=0.1):
        logger.debug('durations match %1.2f'%(datadur))
    else: logger.error('duration mismatch raw:%1.2f vs resampled:%1.2f'%(datadur,fuseddur))
    
    return fused

def checkplot_resampling(raw,resampled,sr,new_sr,**kwargs):
    from matplotlib.pylab import figure
    
    traw = np.linspace(0.,len(raw)/sr,len(raw))
    tresampled = np.linspace(0,len(resampled)/new_sr,len(resampled))

    
    f = figure(figsize=(16,5))
    ax = f.add_subplot(111)
    ax.plot(traw,raw,'k')
    ax.plot(tresampled,resampled,'g',lw=3,alpha=0.8)
    #for ii in xrange(len(tlist)):ax.plot(tlist[ii],resampled_list[ii],alpha=0.4)
    if 'winarray' in kwargs:
        winarray = kwargs['winarray']
        nwins = winarray.shape[0]
        yrunner = np.tile(np.array([0,0.4]),np.int(np.ceil(nwins/2.)))[:nwins]+1
        ax.hlines(yrunner,winarray[:,0]/sr,winarray[:,1]/sr,color='b',alpha=0.5,linewidth=2)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time [s]')
    return f


#------------------------------------------------------------------------------ 
#filehandling
def createPath(path):
    if not os.path.isdir(path):
        os.mkdir(path)


ma_to_arr = lambda marray:  np.vstack([marray.data,marray.mask.astype('int')]) if hasattr(marray,'mask') else marray
def arr_to_ma(obj):
    if type(obj)==np.ndarray:
        if len(obj)==2 and np.size(obj)>0:
            if np.max(obj[1])<=1. and np.min(obj[1])==0.:
                if (np.unique(obj[1]) == np.array([0,1])).all() or (np.unique(obj[1])==np.array([0])).all():
                    return np.ma.array(obj[0], mask=obj[1])
    return obj

def modify_elements_iterator(mydict,modfn):
  for key, val in mydict.items():
    if isinstance(val, dict):
      mydict[key] = modify_elements_iterator(val,modfn)
    else:
      mydict[key] = modfn(val)
  return mydict

def modify_elements(obj,modfn):
    from copy import deepcopy
    myobj = deepcopy(obj)
    if type(myobj) == dict:
        return modify_elements_iterator(obj,modfn)
    else:
        return modfn(obj)

def open_hdf5(filename,group=None,read_maskedarr=True):
    logger.info('Opening %s at group %s'%(filename,group))
    print('Fix hdf5 opening individually! - Deepdish is discontinued.')
    if read_maskedarr:
        return modify_elements(output,arr_to_ma)
    else:
        return output

def append_dataset_hdf5(dshand,newdata):
    Newshape = dshand.shape[0]+newdata.shape[0]
    dshand.resize(Newshape,axis=0)
    dshand[-newdata.shape[0]:] = newdata


def insert_dataset_hdf5(obj, filename, dsname, grouplevel='/'):
    if os.path.isfile(filename):
        rmode = 'r+'
        #hfile = h5py.File(filename, 'r+')
    else:
        rmode = 'w'
        #hfile = h5py.File(filename, 'w')

    with h5py.File(filename, rmode) as hfile:
        if checkexists_hdf5(filename, grouplevel):
            mygroup = hfile.get(grouplevel)
            print('Loading group')
        else:
            mygroup = hfile.create_group(grouplevel)

        if checkexists_hdf5(filename, grouplevel + '/' + dsname):
            print('Overwriting content at %s' % (grouplevel + '/' + dsname))
            mygroup[dsname] = obj
        else:
            print('Creating dataset at %s' % (grouplevel + '/' + dsname))
            myds = mygroup.create_dataset(dsname, data=obj)



def merge_hdf5_special(sourcefile,targetfile,targetkey,sourcekey='data',replace_on_target=True,delete_source=True):

    writeflag = True

    with h5py.File(targetfile, 'r+') as hfile:
        with h5py.File(sourcefile,'r') as hfileTemp:
            #hfileTemp =
            #hfileTemp.flush()
            #insert delete option if already exists
            if checkexists_hdf5(targetfile,targetkey):
                if replace_on_target:
                    writeflag = True
                    del hfile[targetkey]
                else:
                    writeflag = False
                    print ('WARNING: entry already exists, doing nothing!')
            if writeflag:
                copy_hdf5(hfileTemp.id, sourcekey, hfile.id,targetkey)
    #hfileTemp.close()
    #hfile.close()
    if delete_source:os.remove(sourcefile)

def merge_hdf5(sourcefile,targetfile,overwrite_groups=True,delete_source=False):
    hfile = h5py.File(targetfile, 'r+')
    hfileTemp = h5py.File(sourcefile)
    hfileTemp.flush()

    for key in hfileTemp.keys():
        writekey = True
        if key in hfile.keys():
            logger.info('%s already exists in target %s' % (key, targetfile))
            if overwrite_groups:
                logger.info('Deleting old group %s' % (key))
                del hfile[key]
            #else:
            #    writekey = False
        if writekey:
            logger.info('Writing group %s' % (key))
            copy_hdf5(hfileTemp.id, key, hfile.id, key)

    hfileTemp.close()
    hfile.close()
    if delete_source:os.remove(sourcefile)

def copy_hdf5(src_loc,src_name,dst_loc,dst_name):
    if (sys.version_info > (3, 0)):
        h5py.h5o.copy(src_loc, str.encode(src_name), dst_loc,
                      str.encode(dst_name))
    else:
        h5py.h5o.copy(src_loc, src_name, dst_loc,
                      dst_name)


def merge_obj_to_hdf5(obj,targetfile,overwrite_groups=True):
    '''this is only top-level merge!'''
    # saving results to be merged temporarily
    fnametemp = targetfile.replace('.h', 'Temp.h')
    save_hdf5(fnametemp, obj, write_maskedarr=True)

    #merging the two files and deleting the temporary file
    merge_hdf5(fnametemp, targetfile, overwrite_groups=overwrite_groups, delete_source=True)


def makelink_hdf5(sourcefile, targetfile, groupkey, overwrite=True, grouplevel='/'):
    '''

    :param sourcefile:
    :param targetfile:
    :param groupkey:
    :param overwrite:
    :param grouplevel:
    :return:

    Usage example:

    makelink_hdf5(rawfilepath, targetfile, 'raw_data')

    1) retrieve stuff:
    hfile = h5py.File(targetfile,'r')
    raw_filepath = hfile['raw_data'].file.filename
    sr = hfile['raw_data']['data'].attrs['sr']
    trace =  hfile['raw_data']['data']['trace'].value

    2) open linked file directly:
    rawstuff = hf.open_hdf5(raw_filepath)
    '''
    rmode = 'r+' if os.path.isfile(targetfile) else 'w'

    with h5py.File(targetfile,rmode) as hfile:

        if os.path.isfile(targetfile):
            #hfile = h5py.File(targetfile, 'r+')
            logger.info('Updating %s' % (targetfile))
            if groupkey in hfile.keys():
                if overwrite:
                    del hfile[groupkey]
                    logger.info('Overwriting %s' % (groupkey))
                else:
                    logger.info('Key %s already exisits, no overwriting' % (groupkey))
                    return 0
        else:
            #hfile = h5py.File(targetfile, 'w')
            logger.info('Creating %s' % (targetfile))

        if groupkey.count('/') > 0:
            subgroup = '/'.join(groupkey.split('/')[:-1])
            if not subgroup in hfile: hfile.create_group(subgroup) #make room for the new group
        hfile[groupkey] = h5py.ExternalLink(sourcefile, grouplevel)


stringsave_h5 = lambda mygroup, dsname, strlist: mygroup.create_dataset(dsname, data= [mystr.encode("ascii", "ignore") for mystr
                                                                             in strlist],dtype='S60')





def checkexists_hdf5(filename,group):
    if not os.path.isfile(filename):
        return False
    hfile = h5py.File(filename, 'r')
    flag = group in hfile
    hfile.close()
    return flag

#REMOVE save_object and open_object when transferred completely to hdf5!
def save_object(obj,name, quiet=False):
    '''saves a python object using cPickle
    INPUT:
        obj: python object
        name: path where to save
    OUTPUT:
        nothing '''
    f = io.open(name,'wb')
    pickle.dump(obj,f,protocol=2)
    f.close()
    if (f.closed) & (quiet==False):
        print(('saved object ',name))

def open_obj(name,quiet=False):
    '''opens a python object using cPickle
    INPUT: 
        name: path to file
    OUTPUT:
        the object opened'''
    f = io.open(name,'rb+')

    #to catch py2-3 incompatibility
    try:  obj = pickle.load(f, encoding='latin1')#py3
    except:  obj = pickle.load(f)#py2

    f.close()
    if (f.closed) & (quiet==False):
        print(('opened object from file '+name))
    return obj

def string_decoder(obj,mode='utf8'):
    '''for compatibility of strings (by themselves, in lists or in arrays) when loading python 2 data into python 3
    when using python 2 your stringish object wont be changed'''
    if type(obj) == np.bytes_:
        return obj.decode(mode)
    elif type(obj) in [list,np.ndarray]:
        if type(obj[0]) == np.bytes_:
            temp = list(map(lambda val: val.decode(mode), obj))
            if type(obj) == list: return temp
            else: return np.array(temp)
        else: return obj
    else: return obj

def extract_smrViaNeo(filename,**kwargs):

    from neo.io import Spike2IO
    
    if 'chanlist' in kwargs: chanlist = kwargs['chanlist'] 
    stopword = 'd'
    
    r = Spike2IO(filename=filename,try_signal_grouping=False)
    logger.info('Opening %s ...'%(filename))
    starttime = time.time()
    seg = r.read_segment()
    logger.debug('Opening took %1.2f s'%(time.time()-starttime))
    
    allchans = [mysig.name for mysig in seg.analogsignals]

    # rename properly (before: ugly nested strings) when its python 3 and causing a weired clash with NEO
    if allchans[0].count("'") == 2:
        logger.info('Reading strings from NEO.')
        for mysig in seg.analogsignals: setattr(mysig, 'name', mysig.name.replace("'", '')[1:])
        allchans = [mysig.name for mysig in seg.analogsignals]
  

    if 'chanlist' not in kwargs:
        print('Channels available: \n' +''.join([chan+'\n' for chan in allchans]))    
        chanlist = []
        while True:
            chstr = input('Enter channame (press %s to exit): '%(stopword))
            if chstr.strip() == stopword:break
            chanlist += [chstr]
    
    mismatch_chans = [chan for chan in chanlist if not chan in allchans] 
    assert len(mismatch_chans)==0,'Specified channel(s):%s not in %s'%(str(mismatch_chans),str(allchans)) 
        
    ddict = {}
    for chan in chanlist:
        subseg = seg.analogsignals[allchans.index(chan)]
        unitstr = str(subseg.units)
        sr = float(subseg.sampling_rate)
        trace = np.squeeze(np.array(subseg))
        moredata = {}
        if hasattr(subseg,'annotations'):moredata.update({key:value for key,value in list(subseg.annotations.items())})
        if hasattr(subseg,'units'):moredata.update({'units':str(subseg.units)}) 
        ddict[chan] = {'trace':checkadapt_mv(checkadapt_zeromean(trace)),'sr':sr,'moreinfo':moredata}
    return ddict


def checkadapt_mv(trace,thresh=10):
    checkval =  np.abs(np.min(trace))
    #print ('RMS',rms)
    if checkval > thresh: return trace/1000.
    else: return trace

def checkadapt_zeromean(trace):
    return trace-np.mean(trace)

def extract_edf(filename,**kwargs):
    import pyedflib

    if 'chanlist' in kwargs: chanlist = kwargs['chanlist']
    stopword = 'd'

    F = pyedflib.EdfReader(filename)


    #r = Spike2IO(filename=filename)
    logger.info('Opening %s ...' % (filename))
    starttime = time.time()
    #seg = r.read_segment()
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

    mismatch_chans = [chan for chan in chanlist if not chan in allchans]
    assert len(mismatch_chans) == 0, 'Specified channel(s):%s not in %s' % (str(mismatch_chans), str(allchans))

    ddict = {}
    for chan in chanlist:

        chidx = allchans.index(chan)
        sr = F.samplefrequency(chidx)
        trace = F.readSignal(chidx)
        trace_adapted = checkadapt_mv(checkadapt_zeromean(trace))
        print ('TRACECHECK',np.mean(trace_adapted),np.max(trace_adapted),np.min(trace_adapted))
        ddict[chan] = {'trace':trace_adapted , 'sr': sr, 'moreinfo': {}}
    F._close()
    return ddict


def apply_to_artfree(obj, myfn, offclip=0.):
	import core.ea_management as eam
	'''obj needs to be an eaperiod, it is usually a rec
	example:
	offclip has to be offset if not specifically looking at intervals
	'''

	if obj.type == 'rec':
		parent = obj
	else:
		parent = obj.parent
	vals = np.array([])
	for freet in obj.artfreeTimes.clip(offclip):
		EAP = eam.EAPeriod(freet[0], freet[1], parentobj=parent)
		vals = np.r_[vals, myfn(EAP)]
	return vals


#------------------------------------------------------------------------------ 

def generate_ymlsetup(rec_id, sourcefile, templatepath,**kwargs):
    repdict = {'ANIMAL_BLOCK_ELECTRODE': rec_id, 'THIS_SOURCE_FILE': sourcefile}
    if 'replace_dict' in kwargs: repdict.update(kwargs['replace_dict'])
    with open(templatepath, 'r') as tempfile: myfile = tempfile.read()
    for orig, new in repdict.items(): myfile = myfile.replace(orig, new)
    if 'setuppath' in kwargs:
        setuppath = kwargs['setuppath']
    else:
        templatepath.replace('template', rec_id)
        setuppath = os.path.join(os.path.dirname(templatepath),'%s_runparams.yml'%(rec_id))
    with open(setuppath, 'w') as outfile: outfile.write(myfile)
    return 0
