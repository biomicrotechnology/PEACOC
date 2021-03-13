#ARTISFACTION - DEVOTED TO THE AUTOMATIC IDENTIFICATION OF ARTIFACTS
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import zscore

def get_bigArtifacts(data,sr=500., window=1., high_thresh = 4.,low_thresh = 2.,high_mindur = 0.5,low_mindur=1.4,int_thresh=0.12,verbose=False):
    '''Finds ostentatious artifacts characterised by high effective voltage amplitudes and a low number of zero-crossings
    input:
        data (voltage values)
        *args:
            sr: sampling rat, Hz, 500.
            window: window width for running root mean square,s,1.
            high_thresh: minimal voltage to be reached to be considered, scores,4.
            low_thresh: threshold that defines the extent of the window to be consindered wherin highthresh has happend, scores,2.
            high_mindur: minmal duration of rms trace being above high_thresh, s,0.5
            low_mindur: minimal duration of artifact snippet,1.4
            int_thresh: minimal interval between zero-crossings of voltage in defined possible artifact window, s,0.12
            verbose: flag if true: return dictionary including intermediate variables derived during computation ,bool, false
    output:
        (2,N) array: N number of artifacts in [0]:start-points, in [1]: stop points of respective artifact snippet
        if verbose: see above
    '''
    
    
    winpts = np.int(sr*window)
    running_rms = lambda vec,ww: np.sqrt(np.convolve(vec**2, np.ones(ww)/float(ww), 'same'))
    
    #calcualte running root-mean-square of data
    rms_data = running_rms(data,winpts)
    
    crosspts = find_crosspoints(zscore(rms_data),high_thresh)#find where the rms trace is higher than the high_thresh
    crosspts = np.vstack([crosspts[0][(crosspts[1]-crosspts[0])>high_mindur*sr]\
                          , crosspts[1][(crosspts[1]-crosspts[0])>high_mindur*sr]])#only consider if rms is longer than high_mindur above high_thresh

    crossstart2,crossstop2 = find_crosspoints(zscore(rms_data),low_thresh)#find where the rms trace is higher than the low_thresh

    crossidx = [np.argmin(np.abs(crossstart2-crossstart)) for crossstart in crosspts[0]]#find the low_thresh crossing closest to the high_thresh crossing
    
    #definig ROIS: long enough stretches of high RMS
    roiStart,roiStop = np.unique(crossstart2[crossidx]),np.unique(crossstop2[crossidx])#roi are the low_thresh crossing comprising the high_thresh crossings
    roi = np.vstack([roiStart[np.where((roiStop-roiStart)>=low_mindur*sr)[0]],roiStop[np.where((roiStop-roiStart)>=low_mindur*sr)[0]]]).astype('int')#additional criterion: longer= than low_mindur
    
    #for each roi find (time of roi)/(number of zero-crossings within that roi)
    cross_int = np.array([float(len(data[start:stop]))/np.shape(find_crosspoints(data[start:stop],0.))[1] for start,stop in zip(roi[0],roi[1])])
    
    #an artifact happens where the cross interval is longer than int_thresh
    arts = np.vstack([roi[0][np.where(cross_int>=int_thresh*sr)[0]],roi[1][np.where(cross_int>=int_thresh*sr)[0]]])

    if not verbose:
        return arts
    else:
        roishort = np.vstack([roiStart[(roiStop-roiStart)<low_mindur*sr],roiStop[(roiStop-roiStart)<low_mindur*sr]])#just in case I am in verbose-mode and want to plot the too short rois!
        return {'rms_data': rms_data,'crosspts': crosspts, 'high_thresh': high_thresh,'low_thresh': low_thresh,'roi': roi\
                , 'cross_int': cross_int, 'roishort':roishort,'arts':arts,'int_thresh': int_thresh}




def get_saturationArtifacts(data, sr=500.,mindur_stumps=0.008,mindur_teeth=0.03,maxdiff_stumps=10**-3,maxdiff_teeth=10**-2,zlim=5.,verbose=False):

    stumps = get_artStumps(data,minpts=np.int(mindur_stumps*sr),maxdiff=maxdiff_stumps,zlim=zlim)
    teeth = get_teeth(data,minpts=np.int(mindur_teeth*sr),maxdiff=maxdiff_teeth,zlim=zlim)
    teeth = np.array([tooth for tooth in teeth if not tooth in stumps])#because some stumps also the characteristic of low diff of data
    
    if not verbose:
        return np.unique(np.r_[stumps,teeth])
    else:
        minpts =  np.max(mindur_teeth,mindur_stumps)*sr
        bigSnips = np.hstack([find_crosspoints(zscore(data),zlim),find_crosspoints(-zscore(data),zlim)])
        try: bigSnips = np.transpose(np.vstack([bigsnip for bigsnip in bigSnips.T if (bigsnip[1]-bigsnip[0])>minpts]))
        except: bigSnips = np.array([[],[]])
        return {'bigSnips': bigSnips, 'teeth': teeth,'stumps':stumps}
    

def get_artStumps(data,minpts=4,maxdiff=10**-3,zlim=5.):
    
    
        
    getMaxNneigh = lambda mysnip: np.max([len(mysnip[(mysnip>ii-maxdiff) & (mysnip<ii+maxdiff)]) for ii in mysnip])

    
    bigSnips = np.hstack([find_crosspoints(zscore(data),zlim),find_crosspoints(-zscore(data),zlim)]).astype('int')
    try: bigSnips = np.transpose(np.vstack([bigsnip for bigsnip in bigSnips.T if (bigsnip[1]-bigsnip[0])>minpts]))
    except: return np.array([])
    
    #find all the stumps
    stumplist=[]
    for start,stop in zip(bigSnips[0],bigSnips[1]):
        mysnip = np.abs(data[start:stop])#abs because we also want the negative stumps!!!
        if getMaxNneigh(mysnip)>=minpts:
            stumplist.append(np.argmax(mysnip)+start)
    return np.sort(np.array(stumplist))  

def get_teeth(data,minpts=15,maxdiff=10**-3,zlim=5.,behaviour='new'):
    
   
    bigSnips = np.hstack([find_crosspoints(zscore(data),zlim),find_crosspoints(-zscore(data),zlim)]).astype('int')
    try: bigSnips = np.transpose(np.vstack([bigsnip for bigsnip in bigSnips.T if (bigsnip[1]-bigsnip[0])>minpts]))
    except: return np.array([])
    if behaviour=='old':   
        getMaxNneigh = lambda mysnip: np.max([len(mysnip[(mysnip>ii-maxdiff) & (mysnip<ii+maxdiff)]) for ii in mysnip])

        teethlist=[]
        for start,stop in zip(bigSnips[0],bigSnips[1]):
            mysnip = np.diff(np.abs(data[start:stop]))#abs because we also want the negative stumps!!!
            if getMaxNneigh(mysnip)>=minpts:
                teethlist.append(np.argmax(np.abs(data[start:stop]))+start)
          
    
    elif behaviour=='new':
        samesign = lambda p: not np.min(p) < 0 < np.max(p)
        getlongs = lambda vec,crossmat: [vec[start:stop+1] for [start,stop] in (crossmat.T) if len(vec[start:stop+1])>=minpts-1 and samesign(vec[start:stop+1])]
        
        teethlist =[]
        for start,stop in zip(bigSnips[0],bigSnips[1]):
            mysnip = data[start:stop]
            crpts = find_crosspoints(-np.abs(np.diff(np.diff(mysnip))),-maxdiff)
            if len(getlongs(mysnip,crpts))>0:
                teethlist.append(np.argmax(np.abs(mysnip))+start)
    
    return np.sort(np.array(teethlist))  

def fuse_artifacts(arts_list,marg_list,**kwargs):
    ext_arts_list = []
    for arts,marg in zip(arts_list,marg_list):
        if np.size(arts)==0: pass
        else:
            if len(np.shape(arts))==1:# if there is only a non-dimensional tick as art-event
                ext_arts = np.transpose(np.vstack([[art-marg,art+marg] for art in arts]))
            elif len(np.shape(arts))==2: # if the detected artifact has a duration
                ext_arts = np.transpose(np.vstack([[art[0]-marg,art[1]+marg] for art in arts.T]))
            ext_arts_list.append(ext_arts)
    if len(ext_arts_list)==0: return np.array([[],[]])
    fused_arts = merge_overlaps(np.hstack(ext_arts_list))
    if 'mindist' in kwargs:
        starts,stops = fused_arts[0],fused_arts[1]
        over_idx = np.where((starts[1:]-stops[:-1])>kwargs['mindist'])[0]
        ext_fused = np.vstack([np.r_[starts[0],starts[over_idx+1]],np.r_[stops[over_idx],stops[-1]]])
        return ext_fused
    else:return fused_arts


def merge_overlaps(mymat):
    mymat = np.sort(mymat,axis=1)
    no_insiders = remove_insiders(mymat)
    no_overlaps = fuse_chains(no_insiders)
    return no_overlaps


def remove_insiders(mymat):
    starts,stops = mymat[:]
    insider_idx = np.where((starts[1:]<=stops[:-1]) & (stops[1:]<=stops[:-1]))[0]
    newstarts = np.delete(starts, insider_idx+1)
    newstops =  np.delete(stops, insider_idx+1) 
    return np.vstack([newstarts,newstops])
    
def fuse_chains(mymat):
    starts,stops = mymat[:]
    inds = np.where((stops[:-1]-starts[1:])<0)[0]
    return np.vstack([np.r_[starts[0],starts[inds+1]],np.r_[stops[inds],stops[-1]]])


def mask_data(data,startstop_pts):

    startstop_pts = startstop_pts.clip(0)#for negative point values, set 0!
    #create mask
    artmat = np.zeros(len(data))
    tvec = np.arange(len(data))
    for artstart,artstop in zip(startstop_pts[0],startstop_pts[1]):
    
        artmat[(tvec>artstart) & (tvec<artstop)]=1
    
    
    if np.max(artmat)==0: return data # if only negative values were in startstop_pts: return data without mask added
    else: return np.ma.array(data,mask=(artmat==1))# mask array and return       


#------------------------------------------------------------------------------ 
#PLOTTING

def plot_artifacts2(recObj,arttimes_listed,**kwargs):
    from scipy.stats import scoreatpercentile
    from matplotlib.pyplot import subplots
    from ea_management import Period
    
   
    #figure-setup
    pandur = kwargs['pandur'] if 'pandur' in kwargs else 40.
    border_large = kwargs['border_large'] if 'border_large' in kwargs else 5.
    border_small = kwargs['border_small'] if 'border_small' in kwargs else 2.
    pansPerFig = kwargs['pansPerFig'] if 'pansPerFig' in kwargs else 8
    p_h = kwargs['p_h'] if 'p_h' in kwargs else 1.5
    hspacing = kwargs['hspacing'] if 'hspacing' in kwargs else 0.4
    fwidth = kwargs['fwidth'] if 'fwidth' in kwargs else 16
    lr_marg = kwargs['lr_marg'] if 'lr_marg' in kwargs else 0.8
    
    t_h =0.4
    b_h = 0.4
    
    yAmp = recObj.raw_data.max()-recObj.raw_data.min()
    y1 = scoreatpercentile(recObj.raw_data,99.95)
    
    fheight = pansPerFig*p_h+(pansPerFig-1)*hspacing + t_h + b_h   
    
    pify = lambda x: np.int(x*recObj.sr)
    getT = lambda xarray: np.linspace(0.,len(xarray)/recObj.sr,len(xarray))
    
    ybar,xbar,xext = recObj.raw_data.min()+0.2*yAmp, pandur-border_large, border_large
    def draw_scalebar(myax):
        myax.plot([xbar,xbar+xext],[ybar,ybar],color='k',linewidth=4)
        myax.text(xbar+0.5*xext,ybar+0.05*yAmp,'%1.1f s'%(border_large),fontsize=11,ha='center',va='bottom')
        
        
    
    artobjs = []
    for tpts in arttimes_listed:
        if len(tpts) == 1: start,stop,atype = tpts[0]-border_small,tpts[0]+border_small,'small'
        elif len(tpts) == 2: start,stop,atype = tpts[0]-border_large,tpts[1]+border_large,'large'
        P = Period(start,stop)
        setattr(P,'tpts',tpts)
        setattr(P,'type',atype)
        artobjs.append(P)
    
    smalls = [A for A in artobjs if A.type == 'small']
    larges = [A for A in artobjs if A.type == 'large']
    
    smallsmerge = np.hstack([recObj.raw_data[pify(A.start):pify(A.stop)] for A in smalls]) if len(smalls)>0 else np.array([])
    
    npans_small = np.int(np.ceil(len(smalls)*2*border_small/pandur))
    npans_large = np.int(np.sum([np.ceil(A.dur/pandur) for A in larges]))
    npans = npans_small+npans_large
    
    nfigs = np.int(np.ceil(npans/np.float(pansPerFig)))
    
    arrlist,flist = [],[]
    for ff in range(nfigs):
        f, axarr = subplots(pansPerFig,1, figsize=(fwidth,fheight),facecolor='w')
        f.subplots_adjust(left = lr_marg/fwidth,right=1.-lr_marg/fwidth,bottom = b_h/fheight,top=1.-t_h/fheight,\
                              hspace=hspacing)
        arrlist.append(axarr)
        flist.append(f)
        
    aa = 0 
    axcounter = 0
    pansInFig = 0
    for ii in range(npans_small):
        
        if axcounter == pansPerFig: 
           aa+=1
           axcounter = 0         
        ax = arrlist[aa][axcounter]
        ax.plot(getT(smallsmerge),smallsmerge,color='k')
        for pp,P in enumerate(smalls): 
            artpt = border_small*(1+2*pp)
            ax.axvline(pp*2*border_small,color='w',linewidth=8)
            ax.axvline(artpt,color='r',linewidth=1, linestyle='--')
            ax.text(artpt,y1,' %1.1f s'%(P.tpts),ha='left',va='top',fontsize=11,color='r')
        ax.set_xlim([ii*pandur,(ii+1)*pandur])
        if axcounter ==0:draw_scalebar(ax)
        axcounter +=1
        pansInFig +=1
    for pp,A in enumerate(larges):
        npansA = np.int(np.ceil(A.dur/pandur))
        for ii in range(npansA):
            if axcounter == pansPerFig: 
                aa+=1
                axcounter = 0 
            ax = arrlist[aa][axcounter]
            data = recObj.raw_data[pify(A.start):pify(A.stop)]
            ax.plot(getT(data),data,color='k')
            tlim = [border_large,getT(data).max()-border_large]
            ax.hlines(y1,tlim[0],tlim[1],color='r',linewidth=3,alpha=0.5)  
            if ii==0: ax.text(border_large,y1,'[%d, %d] s '%(A.tpts[0],A.tpts[1]),ha='right',va='top',fontsize=11,color='r')    
            ax.set_xlim([ii*pandur,(ii+1)*pandur])
            if axcounter ==0:draw_scalebar(ax)
            axcounter +=1
            pansInFig +=1
            
        
    for f in flist:
        for ax in f.get_axes():
            if np.str(ax.dataLim.xmin) == '-inf':ax.set_axis_off()#remove empty axes
            else:
                ax.set_ylim([recObj.raw_data.min(),recObj.raw_data.max()])
                ax.set_ylabel('mV')
                ax.axes.get_xaxis().set_visible(False)
                for pos in ['top','bottom','right']:ax.spines[pos].set_visible(False)
                ax.yaxis.set_ticks_position('left')
    return flist


def plot_artifacts(data,artdict,sr=500.,**kwargs):
    import matplotlib.pyplot as plt
    from numpy import array
    
    
    rcdef = plt.rcParams.copy()
    newparams = {'axes.labelsize': 16, 'axes.labelweight':'bold','ytick.labelsize': 15, 'xtick.labelsize': 15}
    plt.rcParams.update(rcdef)# Before updating, we reset rcParams to its default again, just in case
    plt.rcParams.update(newparams)
    
    legdict = {}
    tvec = np.linspace(0.,len(data)/sr,len(data))
    #unpack variables
    for key in list(artdict.keys()): 
        #print key, artdict[key]
        if not np.size(artdict[key])==0: 
            if type(artdict[key])==float: exec('%s = %s' % (key,artdict[key]))
            else: exec('%s = np.array(%s)' % (key,list(artdict[key])))
    
    f = plt.figure(figsize=(16,5),facecolor='w')
    f.subplots_adjust(left=0.07,right=0.93,bottom=0.15,top=0.85)
    ax = f.add_subplot(111) 
    
    ax.plot(tvec,zscore(data),'k',lw=2)
    
    ax2 = ax.twinx()
    if 'rms_data' in locals(): ax2.plot(tvec,rms_data,'grey',lw=2)
    if 'crosspts' in locals():ax2.hlines(np.ones((crosspts.shape[1]))*high_thresh,crosspts[0]/sr,crosspts[1]/sr,color='b',linewidth=6)
    if ('cross_int' in locals()) and ('roi' in locals()): ax2.hlines(cross_int/10.,roi[0]/sr,roi[1]/sr,color='FireBrick',linewidth=6)
    if 'roishort' in locals(): ax2.hlines(np.ones((roishort.shape[1]))*low_thresh,roishort[0]/sr,roishort[1]/sr,color='#EDC9AF',linewidth=12)

    if 'bliptimes' in kwargs:ax.vlines(kwargs['bliptimes'],5.,6.,color='ForestGreen',linewidth=3)

    if 'high_thresh' in locals(): ax2.hlines(high_thresh,0.,len(data)/sr,color='b',linestyle='-')
    if 'low_thresh' in locals(): ax2.hlines(low_thresh,0.,len(data)/sr,color='FireBrick',linestyle='-')
    
    if 'int_thresh' in locals(): ax2.hlines(int_thresh*sr/10.,0.,len(data)/sr,color='FireBrick',linestyle='--')
    
    if 'arts' in locals():
        for artstart,artstop in zip(arts[0]/sr,arts[1]/sr): ax2.axvspan(artstart,artstop,color='FireBrick',alpha=0.3,zorder=2)
        #print 'heho.'
        ax.vlines(0.,-100,-90.,color='FireBrick',alpha=0.3,linewidth= 10,zorder=4,label='arts')#just to get a legend label!
    # now for the saturation artefacts
    if 'bigSnips' in locals():
        for snipstart,snipstop in zip(bigSnips[0]/sr,bigSnips[1]/sr): ax.axvspan(snipstart,snipstop,color='grey',edgecolor='none',alpha=0.2,zorder=1)
    if 'teeth' in locals(): ax.vlines(teeth/sr,-16.,16,color='DarkOrange',linewidth=4,zorder=3,label='teeth')#DEB887#7E3517

    if 'stumps' in locals():ax.vlines(stumps/sr,-16.,16,color='DarkViolet',linewidth=2,zorder=4,label='stumps')#'#C58917'

    if 'sats' in locals(): ax.vlines(sats/sr,-16.,16,color='#E2A76F',linewidth=2,zorder=3)
    
    if 'fused_arts' in locals():
        for artstart,artstop in zip(fused_arts[0]/sr,fused_arts[1]/sr): ax.axvspan(artstart,artstop,color='#FFDFDC',zorder=1)
    if 'cleans' in locals():
        for cleanstart,cleanstop in zip(cleans[0]/sr,cleans[1]/sr): ax.axvspan(cleanstart,cleanstop,color='#C2E1E1',zorder=2,alpha=0.5)
        ax.vlines(0.,-100,-90.,color='#C2E1E1',alpha=0.5,linewidth= 10,zorder=4,label='clean')
    
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels)
    ax2.add_artist(leg)
    ax.legend = None 
    
    ax.set_ylim([-16,16])
    ax2.set_ylim([-20,17])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [scores]')
    ax2.set_ylabel('Running RMS',color='grey',rotation = -90)
    if 'tbound' in kwargs: ax.set_xlim(tbound)
    
    plt.rcParams.update(rcdef)
    
    return f


def plot_artifact_snips(data,artdict,sr=500.,maxtime=20,margint=[5.,5.],nrows=5):

    import matplotlib.pyplot as plt
    from numpy import array
    
    
    rcdef = plt.rcParams.copy()
    newparams = {'axes.labelsize': 16, 'axes.labelweight':'bold','ytick.labelsize': 15, 'xtick.labelsize': 15}
    plt.rcParams.update(rcdef)# Before updating, we reset rcParams to its default again, just in case
    plt.rcParams.update(newparams)


    #stumps = artdict['stumps']
    #teeth = artdict['teeth']
    
    for key in list(artdict.keys()): 
        #print key, artdict[key]
        #print np.shape(artdict[key])
        if not np.size(artdict[key])==0 and not key=='bigSnips': 
            if type(artdict[key])==float: exec('%s = %s' % (key,artdict[key]))
            else: exec('%s = np.array(%s)' % (key,list(artdict[key].astype(np.float))))
    #line for debugging
    arts = artdict['arts']
    print('Shape of arts:', arts.shape)
    
    rms_data = zscore(rms_data)
    
    xstart,xlen,ystart,ylen = 0.07,0.9,0.78,0.14
    myylim = [-8,12]
    
    
    nfigs = int(np.ceil(arts.shape[1]/float(nrows)))

    tvec = np.linspace(0.,len(data)/sr,len(data))    

    
    figlist = []
    aa=0
    for ff in range(nfigs):
        f = plt.figure(figsize=(16,12),facecolor='w')
        #f.subplots_adjust(left=0.05,right=0.95,bottom=0.15,top=0.9)
        for ii in range(nrows):
            if aa<arts.shape[1]:
                pstart,pstop = int(arts[0,aa]-margint[0]*sr),int(arts[1,aa]+margint[1]*sr)
                pstart,pstop = np.max([pstart,0]),np.min([pstop,len(data)])
                lenfrac = ((pstop-pstart)/sr)/maxtime
                if lenfrac>1.:
                    lenfrac,pstart,pstop = 1.,arts[0,aa],arts[1,aa]
                    if ((pstop-pstart)/sr)>maxtime:pstart,pstop = int(pstart),pstop+maxtime*sr
                
                ax = f.add_axes([xstart,ystart-0.18*ii,xlen*lenfrac,ylen])
                ax.plot(tvec[pstart:pstop],data[pstart:pstop],'k',lw=2)
                ax2 = ax.twinx()
                ax2.plot(tvec[pstart:pstop],rms_data[pstart:pstop],'grey',lw=2)
                ax2.hlines(high_thresh,pstart/sr,pstop/sr,color='FireBrick',linestyle='-')
                ax2.hlines(cross_int[(roi[1]>=pstart) & (roi[0]<=pstop)]/10.,roi[0][(roi[1]>=pstart) & (roi[0]<=pstop)]/sr,roi[1][(roi[1]>=pstart) & (roi[0]<=pstop)]/sr,color='FireBrick',linewidth=6)
                ax2.hlines(low_thresh,pstart/sr,pstop/sr,color='b',linestyle='--')
                ax2.hlines(int_thresh*sr/10.,pstart/sr,pstop/sr,color='FireBrick',linestyle='--')
                
                ax.axvspan(pstart/sr,arts[0,aa]/sr,color='grey',alpha=0.2)
                ax.axvspan(arts[1,aa]/sr,pstop/sr,color='grey',alpha=0.2)
                ax.set_xlim([pstart/sr,pstop/sr])
                if 'stumps' in locals():
                    if len(stumps[(stumps>=pstart)&(stumps<=pstop)])>0: \
                    ax.vlines(stumps[(stumps>=pstart)&(stumps<=pstop)]/sr,myylim[0],myylim[1],\
                              color='DarkViolet',linewidth=5,alpha=0.2,zorder=1)
                if 'teeth' in locals():
                    if len(teeth[(teeth>=pstart)&(teeth<=pstop)])>0:\
                    ax.vlines(teeth[(teeth>=pstart)&(teeth<=pstop)]/sr,myylim[0],myylim[1],\
                              color='DarkOrange',linewidth=5,alpha=0.5,zorder=1)
                ax.set_ylim(myylim)
                ax2.set_ylim([-17,12])
                ax2.set_yticks([0.,5.,10.])
                if ii==nrows-1:ax.set_xlabel('Time [s]')
                if aa==arts.shape[1]-1:ax.set_xlabel('Time [s]')
                ax.set_ylabel('mV')
                ax2.set_ylabel('RMS [z]',color='grey',rotation=-90, labelpad=20)
                aa+=1
        figlist.append(f)
    plt.rcParams.update(rcdef)
    return figlist

def plot_saturation_events(data,events,sr=500.,plotint=[1.,1.],axdim=(4,8),**kwargs):
    from matplotlib.pyplot import figure
    
    std_data,mean_data = np.std(data),np.mean(data)
    data = zscore(data)
    eventtags =  kwargs['eventtags'] if 'eventtags' in kwargs else None
    
    npanels = axdim[0]*axdim[1]
    nfigs = int(np.ceil(len(events)/float(npanels)))
    figlist = []
    
    myylim = [-20.,15.]
    x_anch,y_anch,xlen,ylen = 0.05*sr,13,plotint[0]*sr,-5.
    ee=0
    for ff in range(nfigs):
        f = figure(figsize=(16,10),facecolor='w')
        f.subplots_adjust(left=0.03,right=0.97,bottom=0.08)
        for ii in range(npanels):
            if ee<len(events):
                ax = f.add_subplot(axdim[0],axdim[1],ii+1)
                ax.plot(data[events[ee]-plotint[0]*sr:events[ee]+plotint[1]*sr],'k',lw=2)
                ax.vlines(sr*plotint[0],myylim[0],myylim[1],color='grey',linewidth=2,linestyle='--')
                ax.hlines(0,0.,np.sum(plotint)*sr,color='grey',linewidth=2,linestyle='-',alpha=0.5,zorder=1)
                ax.text(0.7,0.9,'{0}s'.format(int(events[ee]/sr)),color='FireBrick',transform=ax.transAxes)
                if eventtags: ax.text(0.87,0.05,eventtags[ee].upper(),fontsize=16,fontweight='bold',alpha=0.7,color='grey',transform=ax.transAxes)
                ax.set_ylim(myylim)
                ax.set_axis_off()
                if ii==0:
                    ax.plot([x_anch,x_anch,x_anch+xlen],[y_anch+ylen,y_anch,y_anch],'k',lw=3)
                    ax.text(x_anch-0.25*sr,y_anch+0.5*ylen,'5 z',fontsize=13,fontweight='bold',rotation='vertical',va='center',ha='right')
                    ax.text(x_anch - 0.03*sr,y_anch+0.5*ylen,'%.1f mV' %(std_data*5),fontsize=13,fontweight='bold',rotation='vertical',va='center',ha='right')
                    ax.text(x_anch+0.5*xlen,y_anch+0.8,'1 s',fontsize=13,fontweight='bold',ha='center',va='bottom')
                    
                    
                ee+=1
            else: pass
        figlist.append(f)
    return figlist





#------------------------------------------------------------------------------ 
# WRITE AND READ ARTDICTS AS TXT-FILES
# save artifacts as txt files

def saveArtdict_txt(artdict,filename,decpts = 3):
    file = open(filename,'w')
    
    art_tuples = [(round(aa,decpts),round(bb,decpts)) for aa,bb in artdict['arts'].T]
    sats = artdict['sats']
    #print art_tuples
    
    file.write('(artstart,artstop)\n')
    for ii in art_tuples:
        file.write('{0}\n'.format(ii))
    file.write('\n')
    file.write('saturation artifacts\n')
    for ii in sats:
        file.write('{0}\n'.format(round(ii,decpts)))
    file.close()


def readArtdict_txt(filename):
    
    #collect lines in list stripped of /n that are not just empty
    lines = [line.strip() for line in open(filename,'r') if not line.strip()=='']
    
    #find indices in the list of lines where artifact and saturation artifact data is contained (_b: first index containing, _e: last index containing)
    art_b,art_e = lines.index('(artstart,artstop)')+1,lines.index('saturation artifacts')-1
    sat_b,sat_e = lines.index('saturation artifacts')+1,len(lines)-1
    
    #transform strings and strings of touples into arrays
    if art_e >= art_b: arts = np.vstack([[float(lines[ii][1:-1].split(',')[0]),float(lines[ii][1:-1].split(',')[1])] for ii in np.arange(art_b,art_e+1)]).T
    else: arts = np.array([[],[]])
    
    if sat_e >= sat_b: sats = np.array([float(lines[ii]) for  ii in np.arange(sat_b,sat_e+1)])
    else: sats = np.array([])
    
    artdict = {}
    artdict['arts'],artdict['sats'] = arts,sats
    return artdict


#------------------------------------------------------------------------------ 
#HELPER FUNCTIONS
def find_crosspoints(trace,thresh):
    aboves = np.where(trace>=thresh)[0]
    if len(aboves)>0:
        starts = np.r_[aboves[0],aboves[np.where(np.diff(aboves)>1)[0]+1]]
        stops = np.r_[aboves[np.where(np.diff(aboves)>1)[0]],aboves[-1]]
        return np.vstack([starts,stops])
    else:
        return np.array([[],[]])

