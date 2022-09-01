#blipsorting like spikesorting

from __future__ import division
from __future__ import print_function
import numpy as np


from core.helpers import savitzky_golay


def get_sparseAndDoubles(data,bliptimes,sr=500.,sgParams = [21,2],pol='neg',**kwargs):
    
    if pol=='mix':mixpol=True
    else: mixpol = False
    
    blipWin = kwargs['blipWin'] if 'blipWin' in kwargs else [0.1,0.2]
    nblip_limit = kwargs['nblip_limit'] if 'nblip_limit' in kwargs else [3.,4.,4.]
    intWidth = kwargs['intWidth'] if 'intWidth' in kwargs else [1.8,2.5,2.]
    maxDoubleDist = kwargs['maxDoubleDist'] if 'maxDoubleDist' in kwargs else 0.18 
    trace_margin = kwargs['trace_margin'] if 'trace_margin' in kwargs else 1.#blips closer to the end will be excluded
        #because it is impossible to know whether they are part of a group, also cutout 
        #windows might pose a problem
    t_offset = kwargs['t_offset'] if 't_offset' in kwargs else 0.
                                                

    
    polarity = lambda snip: np.sign(np.abs(snip.max())-np.abs(snip.min()))#just need if mixpol==True
    cr = 0.02
    
    sgTrace = savitzky_golay(data,sgParams[0],sgParams[1])#better work on sg-filtered data
    
    #fish out blips that ride on flanks of bigger blips
    try:
        flankRiders = find_fpsAtFlanks(bliptimes,data,sgTrace,sr=sr,blipWin = blipWin)
        #print flankRiders
    except:
        flankRiders = np.array([])
    
    #get blips that are not part of a larger group of blips and remove double detections among them
    #repeat until no more double-detections are found
    doubles = np.array([1])
    if len(flankRiders)==0:doubleSparse = np.array([])
    else: doubleSparse = flankRiders
    mybliptimes = bliptimes[:]
    while len(doubles)>0:
        #print 'run'
        sparseBlips,groupBlips = get_sparseBlips(mybliptimes,intWidth=intWidth,nblip_limit=nblip_limit,verbose=True)
        #remove double detections
        if not mixpol == True:
            newSparse,doubles = remove_doubleDetected(sparseBlips,sgTrace,maxDist=maxDoubleDist\
                                    ,blipWin=blipWin,sr=sr,verbose=True)
        else:
            sparse_pos = np.array([blip for blip in sparseBlips if polarity(data[int((blip-cr)*sr):int((blip+cr)*sr)])==1.])
            newPos,doublePos = remove_doubleDetected(sparse_pos,sgTrace*-1,maxDist=maxDoubleDist\
                                    ,blipWin=blipWin,sr=sr,verbose=True)
            
            sparse_neg = np.array([blip for blip in sparseBlips if not blip in sparse_pos])
            newNeg,doubleNeg = remove_doubleDetected(sparse_neg,sgTrace,maxDist=maxDoubleDist\
                                    ,blipWin=blipWin,sr=sr,verbose=True)
            
            newSparse,doubles = np.sort(np.r_[newNeg,newPos]),np.sort(np.r_[doubleNeg,doublePos])
            
        mybliptimes = np.array([blip for blip in mybliptimes if not blip in doubleSparse])
        doubleSparse = np.append(doubleSparse,doubles)

    #remove blips too close (trace_margin) to the ends of the trace
    sparseBlips = sparseBlips[np.nonzero((sparseBlips<len(data)/sr-trace_margin) & (sparseBlips>t_offset+trace_margin))]
    
    return sparseBlips,np.unique(np.r_[doubleSparse,flankRiders])

   
def find_fpsAtFlanks(bliptimes,data,filtDat,sr=500.,blipWin = [0.1,0.2]):
    marg1,marg2 = int(blipWin[0]*sr),int(blipWin[1]*sr)
    largeAmp = find_crosspoints(data*-1,4.5)
    #print largeAmp.shape,largeAmp
    largeAmp = np.transpose(np.vstack([[start,stop] for start,stop in zip(largeAmp[0],largeAmp[1])\
                          if not np.max(data[start-2*marg1:stop+2*marg2])>=7.]))#make sure you are not in a seizure
    
    #print largeAmp.shape,largeAmp
    flankRiders = []
    for snipstart,snipstop in zip(largeAmp[0]-marg1,largeAmp[1]+marg2):
        mydat = filtDat[snipstart:snipstop]
        tvec = np.linspace(snipstart/sr,snipstop/sr,len(mydat))
        myblips = bliptimes[(bliptimes<=snipstop/sr) & (bliptimes>=snipstart/sr)]
        mintimes = tvec[np.r_[False, mydat[1:] < mydat[:-1]] & np.r_[mydat[:-1] < mydat[1:], False]]
        blipmins = np.array([mintimes[np.argmin(np.abs(mintimes-blip))]for blip in myblips])
        trueMinBlips = [myblips[np.argmin(np.abs(blipmin-myblips))] for blipmin in np.unique(blipmins)]
        flankies = [blip for blip in myblips if not blip in trueMinBlips \
                    and not (snipstop/sr-blip)<=blipWin[1]/10. and not (blip - snipstart/sr)<=blipWin[0]/10.]
        flankRiders.append(flankies)
    return np.hstack(flankRiders)

def get_sparseBlips(bliptimes,intWidth=[3.,4.,2.],nblip_limit=[3.,4.,4.],verbose=False,strictBef=True):
    
    nBefore_lim,nAfter_lim,nAround_lim =nblip_limit 
    beforeInt,afterInt,aroundInt= intWidth
    

    nBefores = np.array([len(bliptimes[(bliptimes<blip) & (bliptimes>=blip-beforeInt)]) for blip in bliptimes])
    nAfters = np.array([len(bliptimes[(bliptimes>blip) & (bliptimes<=blip+afterInt)]) for blip in bliptimes])
    nArounds = np.array([len(bliptimes[(bliptimes>=blip-aroundInt) & (bliptimes<=blip+aroundInt)])-1 for blip in bliptimes])
    
    
    sparseBlips = np.array([ blip for ii,blip in enumerate(bliptimes) if (nBefores[ii]<=nBefore_lim) \
                            and (nAfters[ii]<=nAfter_lim) and (nArounds[ii]<=nAround_lim)])

    if strictBef:
        strictBefInt = 1.
        nBefores_strict = np.array([len(bliptimes[(bliptimes<blip) & (bliptimes>=blip-strictBefInt)]) for blip in bliptimes])
        sparseBlips2 = np.array([ blip for ii,blip in enumerate(bliptimes) if (nBefores_strict[ii]==0)])
        sparseBlips = np.unique(np.r_[sparseBlips,sparseBlips2])
        
    if verbose:
        groupBlips = np.array([blip for blip in bliptimes if not blip in sparseBlips])
        return sparseBlips,groupBlips      
    else:
        return sparseBlips
    
def remove_doubleDetected(bliptimes,datatrace,maxDist=0.15,blipWin=[0.05,0.2],sr=500.,verbose=False):

    intBefore,intAfter = blipWin
    firstInPair = bliptimes[np.r_[np.diff(bliptimes)<maxDist,False]]
    nextInPair = bliptimes[np.where(np.diff(bliptimes)<maxDist)[0]+1]
    
    true_blips,false_blips = [],[]
    for first,next in zip(firstInPair,nextInPair):
        pstart,pstop = int((first-intBefore)*sr),int((next+intAfter)*sr)
        mintime = (pstart+np.argmin(datatrace[pstart:pstop]))/sr
        pair = [first,next]
        
        
        #         pairdist = np.subtract(mintime,pair)
        #         if np.sum(np.sign(pairdist))<=0:#0 if opposite signs (thats one before min and one after, before wins) and -2 if both after (then nearer wins)
        #             trueBlip,falseBlip = pair[np.argmax(pairdist)],pair[np.argmin(pairdist)]
        #         else:#==2, thats both are before mintime
        #             trueBlip,falseBlip = pair[np.argmin(pairdist)],pair[np.argmax(pairdist)]
        pairdist = np.abs(np.subtract(mintime,pair))
        trueBlip,falseBlip = pair[np.argmin(pairdist)],pair[np.argmax(pairdist)]
        true_blips.append(trueBlip)
        false_blips.append(falseBlip)
    
    all_true = np.sort(np.r_[true_blips,bliptimes[np.r_[np.diff(bliptimes)>=maxDist,False]]])
    
    if verbose == False:
        return all_true
    else:
        return all_true, np.array(false_blips)
    
def get_waveformSnippets(btimes,data,sr=500.,minwin=[0.15,0.1],blipint = [0.1,0.15]):
    btimes = np.array([btime for btime in btimes if np.logical_and(btime<len(data)/sr-blipint[1],btime>blipint[0])])
    mintimes = np.array([blip/sr-minwin[0]+np.argmin(data[np.int(blip-minwin[0]*sr):np.int(blip+minwin[1]*sr)])/sr for blip in btimes*sr])
    snipdict = {snipid: data[int(blip)-int(blipint[0]*sr):int(blip)+int(blipint[1]*sr)] for snipid,blip in enumerate(mintimes*sr)}
    snippts = np.int(np.sum(blipint)*sr)
    '''print 'snipdictlen',len(snipdict)
    for key in snipdict.keys():
        print key,snipdict[key].shape'''
    snips = np.vstack([snipdict[key] for key in list(snipdict.keys()) if len(snipdict[key])])#==snippts to intercept too long snippets
    return snips

def pca(snips,ncomps=2):
    K = np.cov(snips.T)
    evals,evecs=np.linalg.eig(K)
    order=np.argsort(evals)[::-1]
    evecs=np.real(evecs[:,order])
    evals=np.abs(evals[order])
    scores= np.dot(evecs[:,:ncomps].T,snips.T)
    scores = scores/np.sqrt(evals[:ncomps, np.newaxis])
    return scores

def plot_components(comps,inds=[],plotax=[]):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    #inds = kwargs['clinds'] if kwargs.has_key('clinds') else None
    #colorlist = ['MediumBlue','FireBrick','ForestGreen','SkyBlue','DarkViolet','DarkOrange']
    colorlist = get_cluster_colors(nclust=len(np.unique(inds)))
    rcdef = plt.rcParams.copy()
    newparams = {'axes.labelsize': 16, 'axes.labelweight':'bold','ytick.labelsize': 15, 'xtick.labelsize': 15}
    plt.rcParams.update(rcdef)# Before updating, we reset rcParams to its default again, just in case
    plt.rcParams.update(newparams)
    
    if len(plotax)==0: f = plt.figure(facecolor='w')
    else: ax = plotax[0]
    
    if comps.shape[0]==2:
        ax = f.add_subplot(111)
        if len(inds)>0:
            for cln in np.unique(inds):
                mycl = comps.T[inds==cln]
                ax.plot(mycl[:,0],mycl[:,1],colorlist[cln],marker='o',linestyle='')
        else: ax.scatter(comps[0],comps[1],s=80, c='k',marker='o')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')

    elif comps.shape[0]==3:
        ax = Axes3D(f)
        if len(inds)>0:
            for cln in np.unique(inds):
                mycl = comps.T[inds==cln]
                ax.plot(mycl[:,0],mycl[:,1],mycl[:,2],colorlist[cln],marker='o',linestyle='')
        else:ax.scatter(comps[0], comps[1], comps[2], c='k', marker='o')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3',rotation=-90)
        
    plt.rcParams.update(rcdef)
    
    if len(plotax)==0: return f


def cluster(comps,nclust=3,flavour='gmm'):
    #from scikits.learn import cluster as skcluster
    #from scikits.learn import mixture
    
    gmm_failure = False
    
    try:
        if flavour=='gmm':
            from sklearn import mixture
            #Gaussian Mixture Model
            try: clf = mixture.GaussianMixture(n_components=nclust, covariance_type='full')#if you dont set random_state to a specific seed/int 
                                                                         #this will lead to slighthly different results each run
            except: clf = mixture.GMM(n_components=nclust, covariance_type='full')
                                                                            
            clf.fit(comps.T)
            clidx = clf.predict(comps.T)
            return clidx
    except:
        gmm_failure = True
        print ('GMM failed, calculating kmeans instead - check your sklearn package')
        
    if flavour=='kmeans' or gmm_failure:
        n_dim = comps.shape[0]
        centers = np.random.rand(nclust, n_dim)
        centers_new = np.random.rand(nclust, n_dim)
        partition = np.zeros(comps.shape[1], dtype=np.int)
        while not (centers_new == centers).all():
            centers = centers_new.copy()
            distances = (centers[:,np.newaxis,:] - comps.T)
            distances *= distances
            distances = distances.sum(axis=2)
            partition = distances.argmin(axis=0)

            for i in range(nclust):
                if np.sum(partition==i)>0:
                    centers_new[i, :] = comps.T[partition==i, :].mean(0)
        return partition

def plot_overlay(snips,inds,cutwin=[0.1,0.2],y_lim=[-5.,5.],axlist=[],polfac=1,**kwargs):
    from matplotlib.ticker import MaxNLocator

    import matplotlib.pyplot as plt
    
    
    if polfac ==-1: y_lim = [-1*y_lim[1],-1*y_lim[0]]
    #colorlist = ['MediumBlue','FireBrick','ForestGreen','SkyBlue','DarkViolet','DarkOrange']
    colorlist = get_cluster_colors(nclust=len(np.unique(inds)),**kwargs)
    snipt = np.linspace(-cutwin[0],cutwin[1],snips.shape[1])
    nclust = len(np.unique(inds))
    if len(axlist)==0:
        f = plt.figure(facecolor='w',figsize=(3.5*nclust,3.5))
        f.subplots_adjust(left=0.07,right=0.98,bottom=0.15)
    for cln in range(nclust):
        if len(axlist)==0: ax = f.add_subplot(1,nclust,cln+1)
        else: ax = axlist[cln]
        mysnips = snips[inds==cln]
        meansnip = np.mean(mysnips,axis=0)
        if polfac == 1:
            minpt = np.argmin(meansnip)
            ptp = np.max(meansnip[minpt:])-meansnip[minpt]
        else:
            maxpt = np.argmax(meansnip)
            ptp = meansnip[maxpt] - np.min(meansnip[maxpt:])
        for mysnip in mysnips:
            ax.plot(snipt,mysnip,color='grey')
        ax.plot(snipt,meansnip,linewidth=4,color=colorlist[cln])
        ax.text(0.98,0.95,'Clust {0}'.format(str(cln+1)),fontsize=16,fontweight='bold',color= colorlist[cln],transform=ax.transAxes,ha='right',va='top')
        ax.text(0.98,0.05,'PTP {0}'.format(np.around(ptp,1)),fontsize=15,color= colorlist[cln],transform=ax.transAxes,ha='right',va='bottom')
        
        
        ax.set_xticks([-0.2,-0.1,-0.,0.1,0.2,0.3,0.4,0.5])
        ax.set_yticks(np.arange(-8,8,1))
 
        ax.set_xlim([-cutwin[0],cutwin[1]])
        ax.set_xlabel('Time [s]')            
        ax.set_ylim(y_lim)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if cln==0: ax.set_ylabel('mV')
        else: ax.set_yticklabels([''])
    if len(axlist)==0: return f

def plotOverlay_allClusts(fpmethod,pca_dict,data,sr=500.,minwin=[0.1,0.1],\
                          cutwin=[0.1,0.2],quiet=True,**kwargs):
    from matplotlib.pyplot import figure
    import matplotlib.gridspec as gridspec
    
    pol = kwargs['pol'] if 'pol' in kwargs else 'NA'
    
    pca_selection = read_PCA_dict(fpmethod,pca_dict,quiet=quiet)
    
    btimes = pca_selection['blips']
    clinds = pca_selection['clinds']
    noiseclust = pca_selection['noiseclust']
    ncomp,nclust = pca_selection['ncomp'],pca_selection['nclust']
    
    ncols = np.max(nclust)#max works also when list contains single element
    
    if len(btimes)==2: 
        mixpol = True
        nrows = 2
        figD,fbottom,ftop = (ncols*2.5+2,5),0.12,0.92#that is mixed polarity
        
    else: 
        mixpol = False 
        nrows=1
        figD,fbottom,ftop = (ncols*2.5+2,3),0.2,0.85
        
    
    
    f = figure(facecolor='w',figsize=figD)
    f.subplots_adjust(left=0.08,bottom=fbottom,right=0.98,top=ftop)
    
    gsMain = gridspec.GridSpec(nrows, ncols)
    for row in range(nrows):
        
        if row==1 or pol=='pos': pol_factor =-1
        else: pol_factor = 1
        
        snips = get_waveformSnippets(btimes[row],data*pol_factor,sr=sr,minwin=minwin,blipint = cutwin)

        axlist = []
        for cc in range(nclust[row]):
            ax = f.add_subplot(gsMain[row,cc])
            ax.tick_params(top=False, right=False)
            axlist.append(ax)
        plot_overlay(snips,clinds[row],cutwin=cutwin,y_lim=[-12.*data.std(),6*data.std()],axlist=axlist,\
                              noiseclusts=np.array(noiseclust[row])-1)        
                
        if mixpol and row==0:
            for myax in f.get_axes(): 
                myax.set_xlabel('')
                myax.set_ylabel('')
                myax.set_xticks([])
        if mixpol and row==1:
            myax = f.get_axes()[nclust[0]]
            myax.set_yticklabels(myax.get_yticks()*-1)
    return f


def plot_clustInData(data,sparseBlips,inds,sr=500.,**kwargs):
    import matplotlib.pyplot as plt
    from scipy.stats import zscore
    
    #------------------------------------------ if kwargs.has_key('cleantimes'):
        #------------------------------------- cleantimes = kwargs['cleantimes']
        # oldtrues = np.array([blip for blip in sparseBlips if blip in cleantimes])
        # oldfps = np.array([blip for blip in sparseBlips if blip not in cleantimes])
    
    #colorlist = ['MediumBlue','FireBrick','ForestGreen','SkyBlue','DarkViolet','DarkOrange']
    
    if type(inds) == list:
        nclusts_neg,nclusts_pos = len(np.unique(inds[0])),len(np.unique(inds[1]))
        nclusts = [nclusts_neg,nclusts_pos]#first negative then positive
        colorlists = [get_cluster_colors(nclust=nclusts_neg),get_cluster_colors(nclust=nclusts_pos)]
        polord = ['-','+']
    else : 
        nclust = len(np.unique(inds))
        colorlist = get_cluster_colors(nclust=nclust)
    
    
    y1,y2 = -20.,10.
    tvec  = np.linspace(0.,len(data)/sr,len(data))
    f = plt.figure(figsize=(16,4),facecolor='w')
    f.subplots_adjust(left=0.05,right=0.96,bottom=0.15)
    ax = f.add_subplot(111)
    ax.plot(tvec,zscore(data),'k',lw=2,zorder=5)
    if type(inds) == list:
        xx_pp = [[y1,-5.],[5.,y2]]#neg vline range and positive vline range
        for pp in [1,0]:#there are two polarities, for aestetical reasons we plot pos. first
            sparse,clinds,xx = sparseBlips[pp],inds[pp],xx_pp[pp]
            for cln in range(nclusts[pp]):
                ax.vlines(sparse[clinds==cln],xx[0],xx[1],linewidth=3,color=colorlists[pp][cln],alpha=0.8,zorder=3,\
                          label=polord[pp]+'cl {0}'.format(str(cln+1)))
    else:
        for cln in range(nclust):
            ax.vlines(sparseBlips[inds==cln],y1,10.,linewidth=3,color=colorlist[cln],alpha=0.8,zorder=3,label='cl {0}'.format(str(cln+1)))
    if 'groupblips' in kwargs:
        ax.vlines(kwargs['groupblips'],y1,y2,linewidth=9,color='grey',alpha=0.2,zorder=1,label='group')
        #try:ax.vlines(oldfps,-20,15,linewidth=9,color='r',alpha=0.2,zorder=2,label='fp_old')
        #except: print 'no old fps'
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels)
    
    ax.set_ylim([y1,y2])
    
    ax2 = ax.twinx()
    ax2.set_ylim([y1*data.std()-data.mean(),y2*data.std()-data.mean()])
    
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('z-Voltage')
    ax2.set_ylabel('mV',rotation=-90,labelpad=15)
    return f


def get_cluster_colors(nclust=3,**kwargs):

    noiseclusts = kwargs['noiseclusts'] if 'noiseclusts' in kwargs else [nclust-1]
    noisecolor = kwargs['noisecolor'] if 'noisecolor' in kwargs else 'FireBrick'
    
    allcolors = ['MediumBlue','ForestGreen','#FBB117','#92C7C7','DeepPink']
    
    return [col if ii not in noiseclusts else 'FireBrick' for ii,col in enumerate(allcolors) ][:nclust]



def order_clustIndsPTP(snips,inds,minpt=50):
    ptp_meansnip = lambda myind: np.max(np.mean(snips[inds==myind],axis=0)[minpt:])-np.mean(snips[inds==myind],axis=0)[minpt]

    ptp_order = np.argsort(np.array([ptp_meansnip(cln) for cln in np.unique(inds)]))[::-1]
    newinds = np.zeros((inds.shape))

    for rankord,oldord in enumerate(ptp_order):
        newinds[inds==oldord] = rankord
    
    return newinds.astype(int)    

def sortBlips(bliptimes,data,sr=500.,ncomps=3,nclust=4,minsearchwin = [0.1,0.1],cutwin = [0.1,0.2]):    
    
    #get waveforms
    snips = get_waveformSnippets(bliptimes,data,sr=sr,minwin=minsearchwin,blipint = cutwin)

    #extract features
    comps = pca(snips,ncomps=ncomps)

    #cluster
    clinds = cluster(comps,nclust=nclust,flavour='gmm')
    clinds = order_clustIndsPTP(snips,clinds,minpt=int(sr*cutwin[0]))#clind 0 has highest average PTP amplitude, clind.max() has lowest
    return clinds


def eval_PCAPerformance(visstarts,visstops,visblips,cleanalg,dirtyalg,tolerance=0.15):
    
    eval_dict = {key: [] for key in ['tp','fp','fn','tn']}
    
    for snipstart,snipstop in zip(visstarts,visstops):
        visblip_snip = visblips[(visblips>=snipstart)&(visblips<=snipstop)]
        truealg_snip = cleanalg[(cleanalg>=snipstart)&(cleanalg<=snipstop)]
        falsealg_snip = dirtyalg[(dirtyalg>=snipstart)&(dirtyalg<=snipstop)]
    
    
        tp,fp = [],[]
        
        for ta in truealg_snip:
            matchblip = visblip_snip[(visblip_snip>(ta-tolerance))& (visblip_snip<(ta+tolerance))]
            if len(matchblip)>0: eval_dict['tp'].append(matchblip[0])
            else: eval_dict['fp'].append(ta)

        tn,fn = [],[]
        
        for fa in falsealg_snip:
            matchblip = visblip_snip[(visblip_snip>(fa-tolerance))& (visblip_snip<(fa+tolerance))]
            if len(matchblip)>0: eval_dict['fn'].append(matchblip[0])
            else: eval_dict['tn'].append(fa)
            

    return eval_dict





def removeFPs_byPCA(bliptimes,data,sr=500.,sgParams=[21,2],ncomps=3,nclust=4,minsearchwin=[0.1,0.1],\
                    cutwin=[0.1,0.2],printLoop=True,outverbose=True,no_doubles=True,pol='neg',**kwargs):
    
    '''last cluster is tentatively interpreted as noise to get all sparse blips
    no_doubles: view detected double-blips actually as sparse: recommended! - they will get
    removed eventually anyway!'''
    
    t_offset = kwargs['t_offset'] if 't_offset' in kwargs else 0.
    
    if pol == 'pos': data = data*-1.
    polarity = lambda snip: np.sign(np.abs(snip.max())-np.abs(snip.min()))#only used if mixpol==True
    cr = 0.02 # range in which before and after blip polarity will be evaluated in case of mixpol==True

    dirties = np.array([])
    doubles = np.array([])
    dirtydiff = 1
    
    protectedEndblips = np.array([blip for blip in bliptimes if len(data)/sr-blip<np.sum(cutwin)])
    bliptimes = np.array([blip for blip in bliptimes if not blip in protectedEndblips])
    #print protectedEndblips,len(bliptimes)
    
    counter = 0
    while dirtydiff>0:
        #find group and sparse blips
        sparseBlips,doubleDetections = get_sparseAndDoubles(data,bliptimes,sr=sr,sgParams = sgParams,pol=pol,t_offset=t_offset)
        if no_doubles: sparseBlips,doubleDetections = np.r_[sparseBlips,doubleDetections],np.array([])
        groupblips = np.array([blip for blip in bliptimes if not blip in sparseBlips and not blip in doubleDetections])
        
        if counter==0: mysparse = sparseBlips[:]
        else: mysparse = np.r_[dirties,sparseBlips]
        
        
        
        if not pol=='mix':#run blipsort
            clinds = sortBlips(mysparse,data,sr=sr,ncomps=ncomps,nclust=nclust,minsearchwin = minsearchwin,cutwin = cutwin)
            dirtysparse = mysparse[clinds == np.max(clinds)]

        else:#cluster pos and neg trace separately
            mysparse_pos = np.array([blip for blip in mysparse if polarity(data[np.int((blip-cr)*sr):np.int((blip+cr)*sr)])==1.])
            mysparse_neg = np.array([blip for blip in mysparse if not blip in mysparse_pos])
            
            clinds_pos = sortBlips(mysparse_pos,data*-1,sr=sr,ncomps=ncomps,nclust=nclust,minsearchwin = minsearchwin,cutwin = cutwin)
            clinds_neg = sortBlips(mysparse_neg,data,sr=sr,ncomps=ncomps,nclust=nclust,minsearchwin = minsearchwin,cutwin = cutwin)
            
            dirtysparse = np.sort(np.r_[mysparse_pos[clinds_pos == np.max(clinds_pos)],mysparse_neg[clinds_neg == np.max(clinds_neg)]])
        
        dirtydiff = len(dirtysparse) - len(dirties)
        #print 'dirtydiff', dirtydiff
            
        
        if counter==0: dirties, doubles = dirtysparse[:], doubleDetections[:]
        else: dirties,doubles = np.unique(np.r_[dirties,dirtysparse]),np.unique(np.r_[doubles,doubleDetections])
                
        if printLoop: print('Loop {0} \t #(FP) {1} \t #(new FPs) {2}'.format(counter,len(dirtysparse),dirtydiff))
        
        cleansparse = np.array([blip for blip in mysparse if not blip in dirties])
        bliptimes = np.sort(np.r_[cleansparse,groupblips])
        
        counter += 1
    
    bliptimes = np.r_[bliptimes,protectedEndblips]
    if outverbose:
        outdict = {}
        outdict['cleansparse'] = cleansparse
        outdict['allcleans'], outdict['alldirties'] = bliptimes,np.sort(np.r_[doubles,dirties])
        outdict['dirties'] = dirties
        outdict['doubles'] = doubles
        outdict['groupblips'] = np.r_[groupblips,protectedEndblips]
        
        if not pol=='mix':        
            outdict['allsparse'] = mysparse
            outdict['clusterid_sparse'] = clinds
        else:
            outdict['allsparse'] = [mysparse_neg,mysparse_pos]
            outdict['clusterid_sparse'] = [clinds_neg,clinds_pos]
            outdict['mix_order'] = ['neg','pos']
        
        return outdict
        
        
    else:
        return bliptimes,dirties,doubles
    




def identifyDoubletteClusters(btimes,clinds,data,sr=500.,minwin=[0.1,0.1],cutwin=[0.1,0.2]):
    '''btimes is here the sparse blips'''
    
    snips = get_waveformSnippets(btimes,data,sr=sr,minwin=minwin,blipint = cutwin)        

    snipt = np.linspace(-cutwin[0],cutwin[1],snips.shape[1])
    nclust = len(np.unique(clinds))
    falseClustIds = []
    for cln in range(nclust):
        mysnips = snips[clinds==cln]
        meansnip = np.mean(mysnips,axis=0)
        minpt = np.argmin(meansnip)
        if minpt < cutwin[0]*sr-2: falseClustIds.append(cln)
    return falseClustIds


def get_pca_dict(data,bliptimes,ncomp_list,nclust_list,sr=500,sgParams=[21,2],minsearchwin=[0.1,0.1],\
                 cutwin=[0.1,0.2],visoffset_s=600.,quiet = True,pol='neg',**kwargs):
    '''
    for each specified number of components, do a pca and cluster for each number of clusters specified
    output: at each pca_dict[ncomps][nclust], bliptimes and cluster ids, is tuple when polarity is mixed
    
    '''
    t_offset = kwargs['t_offset'] if 't_offset' in kwargs else 0.
    pca_dict= {}
    
    if 'visdict' in kwargs:
        visdict = kwargs['visdict']
        visblips = np.hstack([val[0] for val in list(visdict.values())])/sr-visoffset_s
        visstarts = np.sort(np.array(list(visdict.keys()))/sr-visoffset_s)
        tolerance = 0.15
        visdur=20.
    
    for ncomps in ncomp_list:
        pca_dict[ncomps] = {}
        for nclust in nclust_list:
            if not quiet: print('#(comps) {0} \t #(clust) {1}'.format(ncomps,nclust))
            fpr_dict = removeFPs_byPCA(bliptimes,data,sr=sr,sgParams=sgParams,\
                ncomps=ncomps,nclust=nclust,minsearchwin=minsearchwin,cutwin=cutwin,\
                printLoop=not quiet,pol=pol,t_offset=t_offset)
            cleansparse,falsesparse,double_detections = fpr_dict['cleansparse'],fpr_dict['dirties'],fpr_dict['doubles']
            dirtysparse = np.r_[falsesparse,double_detections]
            
            if 'visdict' in kwargs:
                statsdict = eval_PCAPerformance(visstarts,visstarts+visdur,visblips,cleansparse,dirtysparse,tolerance=tolerance)
                
                if not len(statsdict['fp'])+len(statsdict['tp']) == 0:
                    tp_ratio = len(statsdict['tp'])/float(len(statsdict['fp'])+len(statsdict['tp']))
                else:
                    tp_ratio = 1.
                    
                if not len(statsdict['tn'])+len(statsdict['fn']) ==0:#avoid dividing by zero
                    tn_ratio = len(statsdict['tn'])/float(len(statsdict['tn'])+len(statsdict['fn']))
                else:
                    tn_ratio = 1.
                    
                pca_dict[ncomps][nclust] = fpr_dict['allsparse'],fpr_dict['clusterid_sparse'],(tp_ratio,tn_ratio)
            
            else:
                pca_dict[ncomps][nclust] = fpr_dict['allsparse'],fpr_dict['clusterid_sparse']
    
            if not quiet: print('##########################')
    return pca_dict


def read_PCA_dict(fpmethod,pca_dict,quiet=True):
    '''
    fpmethod can have three possible formats - please stick to them rigorously!:
        (1): 'cocl(ncomp,nclust)', eg.'cocl(3,4)', then the noise cluster is by default 
                the last cluster (eg. the 4th)
        (2): 'cocl(ncomp,nclust,[noiseclust1,noisclust2,...])', noiseclust1 etc are the
                clusters you would like to be seen as noise, eg. 'cocl(3,5,[2,4,5])', then 
                the 2nd,4th and 5th cluster would be interpreted as noise
                the list of noise clusters can also contain only one element (eg. cocl(3,5,[2]))
        (3): 'cocl_neg(ncomp_neg,nclust_neg,[nc1_neg,nc2_neg,...])/cocl_pos(ncomp_pos,nclust_pos,[nc1_pos,nc2_pos,...])'
                this is to be used when you have mixed polarity and do not want to have
                the same parameters for positive and negative blips
        --if you do not want to remove any cluster, enter a number higher than nclust in the
            list of noiseclusters, eg. 'cocl(3,4,[9])'
    '''
    if quiet ==False: print('FPmethod',fpmethod)

    if np.size(list(pca_dict['pca_dict'].values())[0].values()[0])==4:mixpol = True #4 because it is a tuple with 4 arrays
    else: mixpol = False
    
    if fpmethod.count('cocl')==1:#no different evaluation for positive and negative blips
        if '[' in fpmethod: ncomp,nclust,noise_clust = eval(fpmethod[5:-1])
        else: 
            ncomp,nclust = eval(fpmethod[5:-1])
            noise_clust = [nclust]
        
        if mixpol:
            ncomp_neg = ncomp_pos = int(ncomp)
            nclust_neg = nclust_pos = int(nclust)
            noise_neg = noise_pos = noise_clust

        
        
    if fpmethod.count('cocl')==2:#this means different fp removal strategy for pos and neg polarity
        ncomp_neg,nclust_neg,noise_neg = eval(fpmethod[8:fpmethod.index('cocl_pos')-1])
        ncomp_pos,nclust_pos,noise_pos = eval(fpmethod[fpmethod.index('cocl_pos')+8:])
        fpm_neg,fpm_pos = fpmethod.split('/')
    
    
    
    if mixpol:
        
        blips_npol = pca_dict['pca_dict'][ncomp_neg][nclust_neg][0][0]#0 beause its negative
        inds_npol = pca_dict['pca_dict'][ncomp_neg][nclust_neg][1][0]
        
        blips_ppol = pca_dict['pca_dict'][ncomp_pos][nclust_pos][0][1]#1 beause its positive
        inds_ppol = pca_dict['pca_dict'][ncomp_pos][nclust_pos][1][1]
        
        if quiet ==False: print('Mixed Polarity',\
                        [ncomp_neg,ncomp_pos],[nclust_neg,nclust_pos],[noise_neg,noise_pos])
        
        return {'blips':[blips_npol,blips_ppol],'clinds':[inds_npol,inds_ppol],\
                'order':['neg','pos'],'noiseclust':[noise_neg,noise_pos],\
                'ncomp':[ncomp_neg,ncomp_pos],'nclust':[nclust_neg,nclust_pos]}
        
    else:#positive or negative polarity only
        if quiet == False: print('Single Polarity',ncomp,nclust,noise_clust)
        sparse_blips,inds = pca_dict['pca_dict'][ncomp][nclust]
       
        return {'blips':[sparse_blips],'clinds':[inds],'order':['NA'],\
                'noiseclust':[noise_clust],'ncomp':[ncomp],'nclust':[nclust]}
    


def separate_cleanAndNoise(pca_selection,bliptimes):
    
    if len(pca_selection['blips']) == 2: #that is in case of mixed polarity
        blips_npol,blips_ppol = pca_selection['blips']
        inds_npol,inds_ppol = pca_selection['clinds']
        noise_neg,noise_pos = pca_selection['noiseclust']
        
        dirty_neg = blips_npol[np.in1d(inds_npol,np.array(noise_neg)-1)]
        dirty_pos = blips_ppol[np.in1d(inds_ppol,np.array(noise_pos)-1)]
        
        dirties = np.r_[dirty_pos,dirty_neg]
        cleanblips = np.array([blip for blip in bliptimes if not blip in dirties])
        
    else:#either pos or neg polarity
        sparseblips = pca_selection['blips'][0]#0 because it is an one element list
        clinds = pca_selection['clinds'][0]
        noise_clust = pca_selection['noiseclust'][0]

        dirties = sparseblips[np.in1d(clinds,np.array(noise_clust)-1)] 
        cleanblips = np.array([blip for blip in bliptimes if not blip in dirties])
        
    return cleanblips,dirties



def plotCompare_nclust(nclust_dict,data,sr=500.,minsearchwin=[0.1,0.1],cutwin=[0.1,0.2],pol='neg',**kwargs):
    from matplotlib.pyplot import figure
    
    pan_h = kwargs['pan_h'] if 'pan_h' in kwargs else 2.5
    pan_w = kwargs['pan_w'] if 'pan_w' in kwargs else 3.
    wspace = 0.1
    hspace = 0.1
    t_h = 1.
    b_h = 1.
    r_marg = 0.5
    
    if len(nclust_dict[list(nclust_dict.keys())[0]]) == 3:
        #include statistics plots
        barstart,barwidth = 1,0.5 # parameters for barplot
        leftclustplot = pan_w/2.
    elif len(nclust_dict[list(nclust_dict.keys())[0]]) == 2:
        leftclustplot = 1.
        
    nclusts = list(nclust_dict.keys())#list of clusters used
    
    #setting up figure dimensions
    nrows = len(nclust_dict)
    ncols = np.max(nclusts)
    
    fheight = nrows*pan_h + t_h + b_h + (nrows-1)*hspace
    fwidth = ncols*pan_w + r_marg +leftclustplot +(ncols-1)*wspace
    
     
    def exec_figure(nclust_dict,polfac=1):

        f = figure(facecolor='w',figsize=(fwidth,fheight))
        
        f.subplots_adjust(left=leftclustplot/fwidth,top=1.-t_h/fheight,bottom=b_h/fheight,right=1.-r_marg/fwidth)
        for nclust in nclusts:
            
            #PLOT CLUSTERS: lower triganular plot one row: clusters identified by pca for condition #(clust) = nclust
            #prepare axis for the different cluster panels for one specified number of clusters
            axlist = []
            for ii in range(nclust):
               
                ax = f.add_subplot(len(nclusts),np.max(nclusts),(nclust-np.min(nclusts))*np.max(nclusts)+1+ii)
                ax.tick_params(top=False, right=False) #turn of tickmarks on top and right
        
                axlist.append(ax)
            
            snips = polfac*get_waveformSnippets(nclust_dict[nclust][0],data*polfac,sr=sr,\
                                         minwin=minsearchwin,blipint = cutwin)#outside polfac\ to make them return
            inds = nclust_dict[nclust][1]
            plot_overlay(snips,inds,cutwin=cutwin,y_lim=[-12.*data.std(),6.*data.std()],axlist=axlist,polfac=polfac)
        
            if not nclust == np.max(nclusts):
                for ax in axlist:
                    ax.set_xlabel('')
                    ax.set_xticklabels([''])
                    
            #BAR-PLOT on the left
            #the statistics for visual identfied blips (if present) 
            if len (nclust_dict[nclust])>2:
                tp_ratio,tn_ratio = nclust_dict[nclust][2]
    
                #define axis postions and create
                firstax = axlist[0]
                axpos = firstax.get_position()
                boxax = f.add_axes([0.065,axpos.y0,0.05,axpos.y1-axpos.y0])
    
            
                #plot bars as percentage
                
                boxax.bar(barstart, tp_ratio*100., barwidth, color='ForestGreen',edgecolor=None,alpha=0.25)
                boxax.bar(barstart+barwidth+barwidth/2, tn_ratio*100., barwidth, color='FireBrick',edgecolor=None,alpha=0.25)
            
                boxax.spines['bottom'].set_color('none')#remove bottom axis
                boxax.spines['left'].set_position(('axes', -0.1))#move it a bit to the left
                boxax.tick_params(axis='y',direction='out', top=False, right=False, length=10, pad=12, width=1, labelsize='medium')
                boxax.set_xticks([])
                boxax.set_ylabel('Compliance Score',fontsize=14,fontweight='normal')
                boxax.set_ylim([0.,100.])
                boxax.set_xlim([barstart,barstart+2*barwidth+barwidth/2.+0.1*barwidth])
                
                #write down what bars show on first occurence
                if nclust == np.min(nclusts):
                    boxax.text(barstart+barwidth/2,10,'tp/(tp+fp)',rotation=90,va='bottom',ha='center')
                    boxax.text(barstart+barwidth+0.5,10,'tn/(tn+fn)',rotation=90,va='bottom',ha='center')
        
        #remove top and right axis for all panels
        for myax in f.get_axes():
            myax.spines['right'].set_color('none')
            myax.spines['top'].set_color('none')
        return f
    
    
    if  pol == 'mix':
        clustdict1 = {nclust:(nclust_dict[nclust][0][0],nclust_dict[nclust][1][0]) \
            for nclust in list(nclust_dict.keys())} 
        clustdict2 = {nclust:(nclust_dict[nclust][0][1],nclust_dict[nclust][1][1]) \
            for nclust in list(nclust_dict.keys())} 
        figs = [exec_figure(clustdict1,polfac=1), exec_figure(clustdict2,polfac=-1)] 
        return figs
    
    elif pol == 'neg': return exec_figure(nclust_dict,polfac=1)
    elif pol=='pos': return exec_figure(nclust_dict,polfac=-1)



def plot_fpExamples(cleantimes,noisetimes,doubletimes,data,snipdur=20.,sr=500., **kwargs):
    from matplotlib.pyplot import figure
    import matplotlib.collections as collections
    from scipy.stats import zscore
    

    data_mean,data_std = np.mean(data),np.std(data)
    data = zscore(data)

    cutwin = kwargs['cutwin'] if 'cutwin' in kwargs else [0.1,0.1]
    t_offset = kwargs['t_offset'] if 't_offset' in kwargs else  0.
    n_panels = kwargs['n_panels'] if 'n_panels' in kwargs else 5
    starts = kwargs['starts'] if 'starts' in kwargs else \
        np.sort(np.random.rand(n_panels)*(len(data)/sr-t_offset))+t_offset
    hw = (cutwin[0]+cutwin[1])/2.


    f = figure(facecolor='w',figsize=(16,10))
    f.subplots_adjust(left=0.05,right=0.96,top=0.92,bottom=0.08)
    f.text(0.01,0.99,'accepted EDs',color='b',fontsize=15,ha='left',va='top')
    f.text(0.01,0.96,'false positives',color='r',fontsize=15,ha='left',va='top')
    f.text(0.99,0.99,'artifacts',fontweight='bold',color='khaki',fontsize=15,ha='right',va='top')
    f.text(0.99,0.96,'candidates for FP-detection',fontweight='bold',color='grey',fontsize=15,ha='right',va='top')
    
    
    tvec = np.linspace(0,len(data)/sr,len(data))
    
    for ii,start in enumerate(starts):
        ax = f.add_subplot(5,1,1+ii)
        tsnip = tvec[(tvec>=start)& (tvec<=start+snipdur)]
        ttrue = (tvec>=start) & (tvec<=start+snipdur)
        ax.plot(tsnip,data[ttrue],'k',lw=2)
        for events,col in zip([cleantimes,noisetimes,doubletimes],['b','r','DarkOrange']):
            try:
                ax.vlines(events[(events>=start)&(events<=start+20.)],9.5,11.5,color=col,linewidth=2)
            except:
                pass
        if 'boxblips' in kwargs:
            boxblips = kwargs['boxblips']
            boxBlips = boxblips[(boxblips>=start)&(boxblips<=start+20)]
            boxarray = convolve_box(tsnip,(boxBlips-cutwin[0])+hw,hw)
            collection = collections.BrokenBarHCollection.span_where(tsnip, ymin=-13, ymax=15.\
                        , where=boxarray>0, facecolor='grey', alpha=0.3,edgecolor='none')
            ax.add_collection(collection)
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
        
    return f

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#ACCESSORIES

def find_crosspoints(trace,thresh):
    aboves = np.where(trace>=thresh)[0]
    if len(aboves)>0:
        starts = np.r_[aboves[0],aboves[np.where(np.diff(aboves)>1)[0]+1]]
        stops = np.r_[aboves[np.where(np.diff(aboves)>1)[0]],aboves[-1]]
        return np.vstack([starts,stops])
    else:
        return np.array([[],[]])

def get_methdict_blipsort(localvars):
    from time import ctime
    
    date = ctime()
    localvars['date'] = date
    localvars['analysed'] = 'sparse blips only'
    
    
    varlist = ['nclust_list','ncomp_list','minsearchwin','cutwin','sr','sg_Params','date']
    
    methdict = {k: v for (k,v) in list(localvars.items()) if k in varlist}
    return methdict


def convolve_box(tvec,events,width):
    boxarray = np.zeros((len(tvec)))
    for event in events:
        boxarray[(tvec>(event-width))&(tvec<(event+width))]+=1
    return  boxarray



#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#YUNKJARD
def removeFPs_byPCA_DEPRECATED(bliptimes,data,sr=500.,sgParams=[21,2],ncomps=3,nclust=4,minsearchwin=[0.1,0.1],\
                    cutwin=[0.1,0.2],printLoop=True,outverbose=True,pol='neg',**kwargs):

    noiseclusters = list(np.array(kwargs['noiseclusters'])-1) if 'noiseclusters' in kwargs else list(np.array([nclust])-1)
    
    #print 'noiseclusters', noiseclusters
    
    if pol == 'pos': data = data*-1.
    polarity = lambda snip: np.sign(np.abs(snip.max())-np.abs(snip.min()))#only used if mixpol==True
    cr = 0.02 # range in which before and after blip polarity will be evaluated in case of mixpol==True

    dirties = np.array([])
    doubles = np.array([])
    dirtydiff = 1
    
    counter = 0
    while dirtydiff>0:
        #find group and sparse blips
        sparseBlips,doubleDetections = get_sparseAndDoubles(data,bliptimes,sr=sr,sgParams = sgParams,pol=pol)
        groupblips = np.array([blip for blip in bliptimes if not blip in sparseBlips and not blip in doubleDetections])
        
        if counter==0: mysparse = sparseBlips[:]
        else: mysparse = np.r_[dirties,sparseBlips]
        
        
        
        if not pol=='mix':#run blipsort
            clinds = sortBlips(mysparse,data,sr=sr,ncomps=ncomps,nclust=nclust,minsearchwin = minsearchwin,cutwin = cutwin)
            dirtysparse = mysparse[np.in1d(clinds,np.array(noiseclusters))]

        else:#cluster pos and neg trace separately
 
            mysparse_pos = np.array([blip for blip in mysparse if polarity(data[(blip-cr)*sr:(blip+cr)*sr])==1.])
            mysparse_neg = np.array([blip for blip in mysparse if not blip in mysparse_pos])
            
            if type(ncomps)== np.int: 
                if counter==0: print('mix - same components & clusters',ncomps,nclust)
                clinds_pos = sortBlips(mysparse_pos,data*-1,sr=sr,ncomps=ncomps,nclust=nclust,minsearchwin = minsearchwin,cutwin = cutwin)
                clinds_neg = sortBlips(mysparse_neg,data,sr=sr,ncomps=ncomps,nclust=nclust,minsearchwin = minsearchwin,cutwin = cutwin)
                
            else:
                if counter==0: 
                    print('mix - neg/pos components',ncomps[0],ncomps[1])
                    print('mix - neg/pos clusters',nclust[0],nclust[1])
                clinds_pos = sortBlips(mysparse_pos,data*-1,sr=sr,ncomps=ncomps[1],nclust=nclust[1],minsearchwin = minsearchwin,cutwin = cutwin)
                clinds_neg = sortBlips(mysparse_neg,data,sr=sr,ncomps=ncomps[0],nclust=nclust[0],minsearchwin = minsearchwin,cutwin = cutwin)
           
            if type(noiseclusters[0])==list:     
                if counter==0: print('mix - different noise clusters neg/pos', noiseclusters[0],noiseclusters[1])
                dirty_pos = mysparse_pos[np.in1d(clinds_pos,np.array(noiseclusters[1]))]
                dirty_neg = mysparse_neg[np.in1d(clinds_neg,np.array(noiseclusters[0]))]
            else:
                if counter==0: print('mix - same noiseclusters')
                dirty_pos = mysparse_pos[np.in1d(clinds_pos,np.array(noiseclusters))]
                dirty_neg = mysparse_neg[np.in1d(clinds_neg,np.array(noiseclusters))]
        
            dirtysparse = np.sort(np.r_[dirty_pos,dirty_neg])
        
        dirtydiff = len(dirtysparse) - len(dirties)
        #print 'dirtydiff', dirtydiff
            
        
        if counter==0: dirties, doubles = dirtysparse[:], doubleDetections[:]
        else: dirties,doubles = np.unique(np.r_[dirties,dirtysparse]),np.unique(np.r_[doubles,doubleDetections])
                
        if printLoop: print('Loop {0} \t #(FP) {1} \t #(new FPs) {2}'.format(counter,len(dirtysparse),dirtydiff))
        
        cleansparse = np.array([blip for blip in mysparse if not blip in dirties])
        bliptimes = np.sort(np.r_[cleansparse,groupblips])
        
        counter += 1

    if outverbose:
        outdict = {}
        outdict['cleansparse'] = cleansparse
        outdict['allcleans'], outdict['alldirties'] = bliptimes,np.sort(np.r_[doubles,dirties])
        outdict['dirties'] = dirties
        outdict['doubles'] = doubles
        outdict['groupblips'] = groupblips
        
        if not pol=='mix':        
            outdict['allsparse'] = mysparse
            outdict['clusterid_sparse'] = clinds
        else:
            outdict['allsparse'] = [mysparse_neg,mysparse_pos]
            outdict['clusterid_sparse'] = [clinds_neg,clinds_pos]
            outdict['mix_order'] = ['neg','pos']
        
        return outdict
        
        
    else:
        return bliptimes,dirties,doubles
    
    
def plotOverlay_allClustsOLD(btimes,clinds,data,sr=500.,minwin=[0.1,0.1],cutwin=[0.1,0.2]):
    from matplotlib.pyplot import figure
    
    if type(btimes)==list: nrows,figD,fbottom,ftop =2,(14,5),0.12,0.92
    else: btimes,clinds,nrows,figD,fbottom,ftop = [btimes],[clinds],1,(14,3),0.2,0.85
    
    nclust = len(np.unique(clinds[0]))
    
    f = figure(facecolor='w',figsize=figD)
    f.subplots_adjust(left=0.08,bottom=fbottom,right=0.98,top=ftop)
    for jj in range(len(btimes)):
        axlist = []
        for ii in range(nclust):
            ax = f.add_subplot(nrows,nclust,jj*nclust+1+ii)
            ax.tick_params(top=False, right=False) #turn of tickmarks on top and right
        
            axlist.append(ax)
        
        if jj==1:pol_factor=-1.
        else: pol_factor=1.
        snips = get_waveformSnippets(btimes[jj],data*pol_factor,sr=sr,minwin=minwin,blipint = cutwin)        
        plot_overlay(snips,clinds[jj],cutwin=cutwin,y_lim=[-12.,6],axlist=axlist)
        if len(btimes)==2 and jj==0:
            for myax in f.get_axes(): 
                myax.set_xlabel('')
                myax.set_ylabel('')
                myax.set_xticks([])
        if len(btimes)==2 and jj==1:
            myax = f.get_axes()[nclust]
            myax.set_yticklabels(myax.get_yticks()*-1)
            
    return f