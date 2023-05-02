from __future__ import division
from __future__ import print_function

import os
import numpy as np
import logging
import matplotlib
import matplotlib.pyplot as plt

import time
import subprocess
import inspect
import sys
import socket,getpass
import yaml

from core import artisfaction,blipS,blipsort
import core.helpers as hf
import core.ea_management as eam
#import core.artisfaction as artisfaction
#import blipS
#import blipsort
#import helpers as hf
#from helpers import open_obj,save_object,createPath
#import ea_management as eam

logger = logging.getLogger(__name__)
#logger.disabled = True

if 'analysisConfig' in os.environ: configpath = os.environ['analysisConfig']
else: configpath = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace('core','config'),'configAnalysis.yml')


 
class Analysis(object):
    def __init__(self):
        self.defaults = {}
    
    @property
    def id(self):
        return type(self).__name__  
    
    @property
    def defaults(self):
        if not hasattr(self,'_defaults'):
           
            with open(configpath, 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
            self._defaults =  cfg[self.id]
        return self._defaults
    
    @defaults.setter
    def defaults(self,newdict):
        logger.debug('Setting new default dict.')
        self._defaults = newdict 
    
    def setparam(self,pname,val):
        if pname not in self.defaults:
            assert 0, '%s is not a parameter here, please choose from %s'%(pname,str(list(self.defaults.keys())))
        else:
            setattr(self,pname,val)
    
    def figsave(self,fhandle,figname,**kwargs):           
        if not hasattr(self,'fignames'):setattr(self,'fignames',[])
        fhandle.savefig(figname,dpi=self.mydpi,**kwargs)
        self.fignames += [figname]
        plt.close(fhandle)
        
    @property
    def infodict(self):
        if subprocess.call(["git", "branch"], stderr=subprocess.STDOUT, stdout=open(os.devnull, 'w')) != 0:
            githash = 'not_available'
            logger.debug('Not a git repo.')
        else:
            try:
                githash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            except:
                githash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).encode('utf-8')
        mydate,mytime = time.strftime('%Y-%m-%d'), time.strftime('%H:%M:%S')
        calldir =  os.path.realpath(inspect.getfile(inspect.currentframe())) 
        fhandler = [hand for hand in logging.getLogger().handlers if hasattr(hand,'baseFilename')][0] if len(logging.getLogger().handlers)>0 else 0
        logfilename = fhandler.baseFilename if not fhandler==0 else 'NA'#idx one goes to the filehandler
        idict =  {'CodeRevision':githash,'Date':mydate,'Time':mytime,'CodeFile':calldir,\
                'Class':self.id,'LogFile':logfilename,\
                'User':getpass.getuser(),'Host':socket.gethostname()}#'Function':sys._getframe(2).f_code.co_name,
        attrs = ['fignames','method','dependsOn']#'resultfile',
        keys = ['ResultFigure','Method','DependsOn']#'ResultFile'
        for attr,key in zip(attrs,keys):
            val = getattr(self,attr) if hasattr(self,attr) else 'NA'
            idict.update({key:val})
        return idict
        
    @property
    def methodsdict(self):
        #could also by default load and user etc
        return {key:getattr(self,key) for key in list(self.defaults.keys())}


    @property
    def savedict(self):
        return {'data': self.datadict, 'methods': self.methodsdict, 'info': self.infodict}
    
    def _save_resultsdict(self,recObj):
        '''results dict have three sub-dictionaries:
            'data', where the actual analysis results are
            'methods', here you find parameters
            'info': analysis-dates, code-revisions, function calls'''

        recObj._save_byGroup(self.savedict,self.groupkey)


    def make_done_button(self,fig,xanch=0.91,yanch=0.05,width=0.07,height=0.04):
        from matplotlib.widgets import Button
        self._finishedFigs = False
        self.sandexax = fig.add_axes([xanch,yanch,width,height])
        self.bsave = Button(self.sandexax, 'DONE',color='b',hovercolor='c') 
        self.bsave.label.set_color('w')
        self.bsave.label.set_fontweight('bold')
        self.bsave.on_clicked(self._close_figures)
    
    def _close_figures(self,event):
        for f in self.figures:
            plt.close(f)
            self._finishedFigs = True

def open_source(filename,**kwargs):

    fileext = os.path.splitext(filename)[-1]
    if  fileext == '.smr':
        from core.helpers import extract_smrViaNeo
        ddict = extract_smrViaNeo(filename,**kwargs)
    elif fileext == '.edf':
        from core.helpers import extract_edf
        ddict = extract_edf(filename,**kwargs)
    return ddict
    


class Preprocessing(Analysis):
    def __init__(self,data_before,sr_before,chanid='NA',moreinfo={},**kwargs):
        
        for key,val in list(self.defaults.items()):
            setattr(self,key,val) 
        #
        
        self.setparam('sr_before',sr_before)
        self.setparam('moreinfo',moreinfo)
        self.setparam('chanid',chanid)

        self.data_before = data_before 
        
        self.save_data = True
        self.method = 'Resampling from source file.'
        self.dependsOn = kwargs['sourcefile'] if 'sourcefile' in kwargs else 'NA'

 
    @eam.lazy_property
    def resampled(self):
        from core.helpers import smartwins,resample_portions
        if self.sr == self.sr_before: 
            logger.info('No resampling as sr_before matches sr: %1.2f'%(self.sr))
            return self.data_before
        else:
            logger.info('Resampling from %1.2f to %1.2f'%(self.sr_before,self.sr))
            minpts = np.int(self.mindur_snip*self.sr_before)
            winarray = smartwins(len(self.data_before),minpts,overfrac=self.overlap_frac)
            logger.debug('Start resampling')        
            resampled = resample_portions(self.data_before,winarray,self.sr_before,self.sr)
            logger.info('Done resampling')
            return resampled
    
    def plotcheck(self):
        from core.helpers import checkplot_resampling
        f = checkplot_resampling(self.data_before,self.resampled,self.sr_before,self.sr)
        return f
    
    def write_result(self,recObj,**kwargs):
        
        if self.save_data:
            #self.datadict = {'trace':self.resampled ,'sr':self.sr}
            if 'to_file' in kwargs: self.outfile = kwargs['to_file']
            else: self.outfile = str(recObj.rawfileH)
            with h5py.File(self.outfile,'w') as fdst:
                dgr = fdest.create_group('data')
                dgr.create_dataset('trace',data=self.resampled,dtype='f')
                dgr.attrs['sr'] = self.sr
                igr = fdest.create_group('info')
                for key,val in self.infodict.items():
                    igr.attrs[key] = val
                mgr = fdest.create_group('methods')
                mgr.attrs['sr'] = self.sr
            #hf.saveto_hdf5(self.savedict,self.outfile, overwrite_file=True, mergewithexisting=False)



    
class Polarity(Analysis):
    def __init__(self):
        for key,val in list(self.defaults.items()):
            setattr(self,key,val) 
        
        self.method = 'Checking polarity of discharges by visual ispection.'
        self.valid_polarities = ['neg','pos','mix']
        self.figures = []
    
    def set_polarity(self,recObj,pol,write=True,**kwargs):
        self.setparam('polarity',pol)

        recObj.polarity = pol
        if write:
            logger.info('Writing polarity: %s'%(pol))
            if 'to_file' in kwargs: filepath = kwargs['to_file']
            else: filepath = recObj._get_filepath('polarity')
            tf = open(filepath,'w')
            tf.write(pol)
            tf.write('\nchecked=%s'%(str(self.checked)))
            tf.close()
        
    
    def _checkfn(self,label):
        #print label
        #boollist= [label==pol for pol in self.valid_polarities]
        #self._checkbuttons()
        #self.set_polarity(label,self.recObj)
        bools = np.array([self.check.lines[ii][0].get_visible() for ii in range(len(self.valid_polarities))])
        if np.sum(bools)==1:
            self.checked_polarity = np.array(self.valid_polarities)[bools][0]
            logger.info('Clicking polarity: %s'%(self.checked_polarity))
        self.setparam('checked', True)
            
    def plot_and_pick(self,recObj,checkbox_on=True,save_and_exit_button=True):
        
        #plotting raw
        recObj.plot(['raw'])
        self.figures.append(plt.gcf())
        
        
        #plotting amplitudes
        self.plot_amps(recObj,checkbox_on=checkbox_on,save_and_exit_button=save_and_exit_button)
        plt.show()

    def plot_amps(self,recObj,checkbox_on=True,save_and_exit_button=True):
        from matplotlib.widgets import CheckButtons
        self.recObj=recObj
        f,ax = plt.subplots(1,1,facecolor='w')
        mybins = np.int(np.sqrt(len(recObj._raw))/3.)
        myhist,mybins = np.histogram(recObj._raw,mybins)
        yhist = myhist/float(len(recObj._raw))
        ymin,ymax = 10**-7,yhist.max()+10**-2
        ax.plot(mybins[:-1],yhist,'k',lw=2)
    
        ax.set_xlim([mybins.min(),mybins.max()])
        ax.set_yscale('log')
        ax.set_ylim(ymin,ymax)
        
        ztify  = lambda x: (x-np.mean(recObj._raw))/np.std(recObj._raw)
        zlim = [ztify(ax.get_xlim()[0]),ztify(ax.get_xlim()[1])]
        ax2 = ax.twiny()
        ax2.set_xlim(zlim)#np.floor(ztify(data).max()) 
        
        ax.set_xlabel('Amplitudes [mV]')
        ax2.set_xlabel('Zscore')
        ax.set_ylabel('Probability')
        
        if checkbox_on:
            #polarity checkboxes
            self.boollist= [self.polarity==pol for pol in self.valid_polarities] #to make the default polarity checked
            self.checked_polarity = str(self.polarity)
            self.checkax = f.add_axes([0.75, 0.75, 0.1, 0.15])        
            self.checkax.set_axis_off()
            self.check = CheckButtons(self.checkax, self.valid_polarities, self.boollist)
            self.check.on_clicked(self._checkfn)
        
        self.figures.append(f)
        if save_and_exit_button:self.make_done_button(f)
      
        plt.show()       
    

class ArtifactDetection(Analysis):
    def __init__(self):
        
        for key,val in list(self.defaults.items()):
            setattr(self,key,val) 
            
        for key,val in list(self.plot_dict.items()):
            setattr(self,key,val)
            
        self.rejectcolor = 'skyblue'
        self.acceptcolor = 'gold'
        self.artcolor = 'darkviolet'
        self.sartcolor = 'r'
        
        self.method = 'Semi-automatic artifact detection.'
        self.figures = []
    
    
    
    @property
    def plot_dict(self):
        return {'figformat': '.png',\
                'mydpi': 100}  
    
    def _plot_shortart(self,timept,mfac=1.):
        mypt = self.ax.plot(timept,(self.y1-0.1*self.yAmp)*mfac, ms=10,marker='o',mec=self.sartcolor, color=self.sartcolor,picker=5)
        return mypt[0]
    
    def _plot_longart(self,timepts,mfac=1.):
        myline = self.ax.plot(timepts,np.array([self.y1,self.y1])*mfac,linewidth=5,color=self.artcolor,picker=5)
        return myline[0]
    
    def get_artifacts(self,recObj):
        art_params = {attr:getattr(self,attr) for attr in ['window','high_thresh','low_thresh',\
                                                               'high_mindur','low_mindur','int_thresh']}
        sart_params = {attr:getattr(self,attr) for attr in ['mindur_stumps','mindur_teeth','maxdiff_stumps',\
                                                               'maxdiff_teeth','zlim']}
        arts = artisfaction.get_bigArtifacts(recObj.raw_data,sr=recObj.sr,**art_params)
        sarts = artisfaction.get_saturationArtifacts(recObj.raw_data, sr=recObj.sr,**sart_params)

        self.arts_long = arts/recObj.sr
        self.arts_short = sarts/recObj.sr

    def _onpick(self,event):
        thisline = event.artist
        self.ax.picked_object = thisline#to be able to delete it again
        xdata = thisline.get_xdata()
        mouse = event.mouseevent
        #print mouse
        if mouse.dblclick and mouse.button in [1,3]:
            if mouse.button==1: thisline.set_color(self.acceptcolor)
            elif mouse.button==3:thisline.set_color(self.rejectcolor)     
            if mouse.ydata>1.:#that is in the upper part where the auto-detected artifacts are shown
                self.artcount = self.artcount-1
                self.update_suptitle()
            self.fig.canvas.draw()      


    def _draw_line(self,startx):
        xy = plt.ginput(1)
        x = [startx,xy[0][0]]
        mylongart = self._plot_longart(x,mfac=-1.)
        mylongart.set_color(self.acceptcolor)
        self.ax.figure.canvas.draw()        

    def _onclick(self,event):
        if event.ydata <0.: 
            #print event       
            if event.dblclick:
                if event.button == 1:
                    self._draw_line(event.xdata) # here you click on the plot
                    
                elif event.button == 2:
                    myshortart = self._plot_shortart(event.xdata,mfac=-1.)
                    myshortart.set_color(self.acceptcolor)
                    self.ax.figure.canvas.draw()
                else:
                    pass 
    
    def _on_key(self,event):

        if event.key == 'd' and event.ydata<0.:#you can only delete the auto-drawn ones below the zero-line
            if self.ax.picked_object:
                self.ax.picked_object.remove()
                self.ax.picked_object = None
                self.ax.figure.canvas.draw()
    
    def _goto_neighbor(self,event):

        flavour = 'next' if event.inaxes==self.axnext else 'prev' if event.inaxes==self.axprev else 0
        
        myfn = np.min if flavour=='next' else np.max 
        
        if len(self.allartstarts)>0.:
            winw = self.ax.viewLim.width
           
            if hasattr(self,'current'):self.current[0].remove()
            
            if flavour == 'next': artstarts = self.allartstarts[self.allartstarts>=self.ax.viewLim.xmax]
            elif flavour =='prev': artstarts = self.allartstarts[self.allartstarts<=self.ax.viewLim.xmin]
            if len(artstarts)>0: nextart = myfn(artstarts)
            else: nextart = myfn(self.allartstarts)
            self.current = self.ax.plot(nextart,self.y1,'v',mfc='none',mec='b',mew=2,ms=10)
            self.ax.set_xlim([nextart-winw/2.,nextart+winw/2.])
            plt.draw()
        
        

    

    def update_suptitle(self):
        self.fig.suptitle('%d out of %d remaining to be checked'%(self.artcount,self.N_auto))
        
    def plotcheck_artifacts(self,recObj,save_and_exit_button=True):
        from matplotlib.widgets import Button
        from matplotlib.ticker import MultipleLocator
        from scipy.stats import scoreatpercentile
        alpha = 0.4
        self.recObj = recObj
        if not hasattr(self,'arts_short'): self.get_artifacts(recObj)
        
        self.N_auto = len(self.arts_short)+len(self.arts_long[0])
        self.artcount = np.int(self.N_auto)
        logger.info('N artifacts automatically detected: %d'%self.N_auto)
        
        self.y1 = scoreatpercentile(recObj.raw_data,99.95)
        self.yAmp = np.max(recObj.raw_data)-np.min(recObj.raw_data)
        
        recObj.plot(['raw'])
        ax, f = plt.gca(), plt.gcf()
        f.subplots_adjust(top=0.85)
        f.text(0.01,0.98,'Accept: Double Leftclick',color=self.acceptcolor,ha='left',va='top',fontweight='bold')
        f.text(0.99,0.98,'Reject: Double Rightclick',color=self.rejectcolor,ha='right',va='top',fontweight='bold')
        f.text(0.01,0.01,'Add singlet: Double Middleclick',color=self.acceptcolor,ha='left',va='bottom',fontweight='bold')
        f.text(0.99,0.01,'Add stretch: Double Leftclick + singleclick',color=self.acceptcolor,ha='right',va='bottom',fontweight='bold')
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_ylabel('mV')
        
        
        
        self.ax = ax 
        self.fig = f
        self.update_suptitle()
        
        for sart in self.arts_short: self._plot_shortart(sart)
        for art in self.arts_long.T: self._plot_longart(art)        
        self.fig.canvas.mpl_connect('pick_event', self._onpick)
        cid = self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        cid2 = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        #move to next artifact buttons
        self.axprev = self.fig.add_axes([0.2, 0.9, 0.05, 0.075])
        self.axnext = self.fig.add_axes([0.25, 0.9, 0.05, 0.075])
        self.bnext = Button(self.axnext, 'Next')
        self.bprev = Button(self.axprev, 'Prev')
        self.allartstarts = np.r_[self.arts_long[0],self.arts_short]
        self.bnext.on_clicked(self._goto_neighbor)
        self.bprev.on_clicked(self._goto_neighbor)
        
        self.figures.append(f)
        if save_and_exit_button:self.make_done_button(f,xanch=0.92,yanch=0.07,width=0.07,height=0.07)
        
        self.setparam('checked', True)
        plt.show()
        
    def harvest_artifacts(self):
        if not hasattr(self,'ax'):
            logger.error('Please check artifacts by clicking using plotcheck_artifacts')
        unchecked, accepted, rejected = [],[],[] 
        for obj in self.ax.get_children():
            if not hasattr(obj,'get_color'): pass
            elif obj.get_color() in [self.artcolor,self.sartcolor]:
                unchecked.append(obj.get_xdata())
            elif obj.get_color() == self.rejectcolor: 
                rejected.append(obj.get_xdata())
            elif obj.get_color() == self.acceptcolor: 
                accepted.append(obj.get_xdata())
            else: pass

        if len(unchecked) ==0: logger.info('All automatically deteced artifacts have been checked.')
        else: logger.warning('%d are not checked yet.'%(len(unchecked)))
        
        self.unchecked = unchecked
        self.rejected = rejected
        self.accepted = accepted
        
    @property
    def artdict(self):
        '''can only be used after harvest_artifacts'''
        sats = np.unique([item for item in self.accepted if np.size(item)==1])
        _arts =  [item for item in self.accepted if np.size(item)==2]
        arts =  np.vstack(_arts).T if len(_arts)>0 else np.array([])
        return {'arts':arts,'sats':sats}
    
    def write_artifacts(self,recObj,save_txt =True):
        if not hasattr(self,'accepted'):
            logger.info('Harvesting artifacts clicked in plotcheck_artifacts')
            self.harvest_artifacts()
            

        artisfaction.saveArtdict_txt(self.artdict,recObj.artifactfile,decpts = 2)
        
        if self.save_figs:
            self.fdir = os.path.join(recObj.figpath,recObj.id+'__artifacts')
            logger.info('Saving figures of artifacts: %s'%(self.fdir))            
            hf.createPath(self.fdir)

            if len(self.accepted)>0:
                flist = artisfaction.plot_artifacts2(recObj,self.accepted)
                for ff,fig in enumerate(flist):
                     fig.suptitle('Accepted Artifacts. %d/%d'%(ff+1,len(flist)))
                     figname = os.path.join(self.fdir,recObj.id+'__artifacts_%dOf%d'%(ff+1,len(flist))+self.figformat)
                     self.figsave(fig,figname)



class EdDetection(Analysis):
    def __init__(self):
        for key,val in list(self.defaults.items()):
            setattr(self,key,val)                    
        for key,val in list(self.plot_dict.items()):
            setattr(self,key,val)
            
        self.method = 'Detecting discharges based on spectrogram, then applying amplitude threshold to catch undetected.'
    
        self.figures = []
        
        
    @property
    def plot_dict(self):
        return {'figformat': '.png',\
                'mydpi': 100,\
                'snipdur': 40.,\
                'spectral_color':'b',\
                'amp_color':'r'}  

    
    def get_data(self):    
        dfile = self.recObj.rawfileH
        self.data = self.recObj.raw_data[np.int(self.offset*self.recObj.sr):]
        if self.consider_artifacts: self.mask_artifacts()
        
    def mask_artifacts(self):
        
        artifactfile = self.recObj.artifactfile

        if os.path.isfile(artifactfile):
            logger.info('Loading artifacts.')
            artdict = artisfaction.readArtdict_txt(artifactfile)
            self.arts_fused = artisfaction.fuse_artifacts([artdict['arts'], artdict['sats']], [self.marg_arts, self.marg_sats],\
                                                  mindist=self.mindist_arts)
            logger.debug('Number of artifact-regions: %d'%(self.arts_fused.shape[1]))
            if np.size(self.arts_fused)>0:
                logger.info('Masking data.')
                self.data = artisfaction.mask_data(self.data, (self.arts_fused - self.offset) * self.recObj.sr)  # data will stay unmasked if artifacts are out of range or simply not present
                
        else: 
            logger.info('Disreagarding artifacts.')
    
    def set_recObj(self,recObj): 
        self.recObj = recObj
        self.sr = recObj.sr
    
    
    @eam.lazy_property
    def avg_power(self):
        if not hasattr(self,'data'): self.get_data()

        self.retrieved_apower = False
        #check whether the resultsfile has an entry in the avg_power
        apowergroup ='/'+self.groupkey+'/data/apower'
        normfacgroup = '/'+self.groupkey+'/data/normfacs'
        exists_solid = np.min([hf.checkexists_hdf5(self.recObj.resultsfileH,mygroup) for mygroup in [normfacgroup,apowergroup]])
        exists_temp = np.min([os.path.isfile(self.recObj.resultsfileH.replace('.h5', '_Temp_%s.h5' %(dsname))) for dsname in ['apower','normfacs']])

        if exists_solid and self.retrieve_apower:
            logger.info('Loading averaged spectrogram from %s'%(apowergroup))
            with h5py.File(self.recObj.resultsfileH,'r') as hand:
                avg_power = hand[apowergroup][()]
                self.normfacs = hand[normfacgroup][()]
            #avg_power = hf.open_hdf5(self.recObj.resultsfileH,apowergroup)
            #self.normfacs = hf.open_hdf5(self.recObj.resultsfileH,normfacgroup)
            self.retrieved_apower = True

        elif exists_temp and self.retrieve_apower:
            logger.info('Reading from Temporary files')
            apower_path = self.recObj.resultsfileH.replace('.h5', '_Temp_%s.h5' %('apower'))
            normfacpath = self.recObj.resultsfileH.replace('.h5', '_Temp_%s.h5' %('normfacs'))
            with h5py.File(apower_path,'r') as hand: avg_power = hand['apower'][()]
            with h5py.File(normfacpath,'r') as hand: self.normfacs = hand['normfacs'][()]
            #avg_power = hf.open_hdf5(apower_path)
            #self.normfacs = hf.open_hdf5(normfacpath)

        else:
            import gc
            logger.info('Calculating spectrogram.')
            tstart = time.time()
              
            # calculate average power and save
            sp = blipS.get_spectrogram(self.data, self.window)    
            logger.info('Normalizing spectrogram, conf-int: %s'%(str(self.norm)))
            sp_norm,self.normfacs = blipS.dynamic_normalisation(sp, norm=self.norm,verbose=True)
            del sp
            logger.info('Averaging spectrogram, bins: %s Hz'%(str(self.avg_lim)))
            avg_power = blipS.average_spectralBins(sp_norm, sr=self.recObj.sr, avg_lim=self.avg_lim)
            del sp_norm
            logger.info('Time needed to calculate average spectrogram: %1.1f s'%(time.time()-tstart))
            if self.save_apower: 
                logger.info('Saving average power and normfactors')
                for dsname,obj in zip(['normfacs', 'apower'],[self.normfacs,avg_power]):
                    tempname = self.recObj.resultsfileH.replace('.h5','_Temp_%s.h5'%(dsname))
                    logger.info('... '+tempname)
                    with hdf.File(tempname,'w') as hand:
                        hand.create_dataset(dsname,dataset=obj,dtype='f')

                    #hf.save_hdf5(tempname, obj)
            gc.collect()
            
        logger.info('Z-scoring and padding average power.')
        #logger.info('len avg_power: %d, len data: %d'%(len(avg_power),len(self.data)))
        return blipS.preprocess_data(avg_power, len(self.data)) #self.avg_power =
    
    
    @property
    def threshx(self):
        return np.arange(self.thr_range[0], self.thr_range[1] + self.thr_res, self.thr_res)

    def get_spikes(self,thresh,avg_power):
        return blipS.zthreshdetect_blips(avg_power,sr=self.recObj.sr,thresh=thresh,pruneOn=self.prune_on,dtime=self.dtime)
        
    @eam.lazy_property
    def f_of_thresh(self):
        logger.info('Calculating #(discharges) as function of threshold')
        logger.debug('threshmin %1.2f | threshmax %1.2f'%(self.threshx[0], self.threshx[1]))
        if type(self.avg_power) == np.ma.core.MaskedArray:
            avg_power = self.avg_power[self.avg_power.mask==False].data
        else: avg_power = self.avg_power[:]
            
        return np.array([len(self.get_spikes(zthresh,avg_power)) for zthresh in self.threshx])

    @property
    def zthresh(self):
        if not hasattr(self,'_zthresh'): 
            if self.manthresh == None:            
                thr = blipS.getZthresh_derivMethod(self.f_of_thresh, self.threshx, peakperc=self.peakperc,\
                                                  mode=self.threshmode)
                logger.info('Using mode: %s; Calculated threshold is %1.2f'%(self.threshmode,thr))
            else:
                thr = np.float(self.manthresh)
                logger.info('Using manually set threshold: %1.2f'%(thr))
            self._zthresh = thr
            return thr
        else: return self._zthresh
        
    @zthresh.setter
    def zthresh(self,zthresh):
        logger.info('Setting zthresh: %1.2f'%(zthresh))
        self._zthresh =zthresh    
        #self.threshmode='man'
        #self.setparam('manthresh',zthresh) 

    
    @property
    def spikes_spectral(self):
        return self.get_spikes(self.zthresh,self.avg_power)    
    
    @property
    def spikes_amplitude(self):
        logger.info('Catching amplitude spikes. Polarity is %s'%(self.recObj.polarity))
        ampspikes = blipS.catch_undetectedBlips(self.spikes_spectral, self.data, sr=self.recObj.sr, \
                                    zthresh=self.amp_thresh, blip_margin=self.spike_margin, \
                                    dtime=self.dtime, pol=self.recObj.polarity)
        ratio = len(ampspikes)/np.float(len(self.spikes_spectral))
        logger.info('N_spectral: %d, N_amplitude: %d, ratio:%1.2f.'%(len(self.spikes_spectral),\
                                                                    len(ampspikes),ratio))
        return ampspikes


    def plot_fofthresh(self,ax=None):
        if ax == None:
            f, ax = plt.subplots(1,1, figsize=(6.,5.2),facecolor='w')
        if self.threshmode.count('man'):
            _ = blipS.getZthresh_derivMethod(self.f_of_thresh, self.threshx, peakperc=self.peakperc,\
                                                  mode=self.threshmode,plotax=ax,manthresh=self.zthresh)
        
        else:
            _ = blipS.getZthresh_derivMethod(self.f_of_thresh, self.threshx, peakperc=self.peakperc,\
                                                  mode=self.threshmode,plotax=ax)
        
    
    
    def _thrpt2d(self,val):
        return [self.threshx[np.argmin(np.abs(self.threshx-val))],\
                self.f_of_thresh[np.argmin(np.abs(self.threshx-val))]]
        
    def drawthreshline(self,ax,key):
        thr,mycol = self.threshdict[key]
        x,y = self._thrpt2d(thr)
        ax.axvline(x,color=mycol,linewidth=1.5,linestyle='--')
        myh = ax.plot(x,y,marker='o',color=mycol,ms=10,mew=0,picker=5,label=key)
        return myh[0]
    
    def drawgetSpectralticks(self,ax,key,**kwargs):
        y1,y2 = kwargs['yext'] if 'yext' in kwargs else [0.,1.]
        thr,mycol = self.threshdict[key]
        if 'pax' in kwargs:
            kwargs['pax'].axhline(thr,color=mycol,label=key,linewidth=1.5,linestyle='--')
        myspikes = self.get_spikes(thr,self.avg_power)+self.offset
        ax.vlines(myspikes,y1,y2,color=mycol,label=key,linewidth=2.)
    
    def _append_threshdict(self,lab,threshval):        
        newentry = {lab:[threshval,self._thrcolors[0]]}
        logger.debug('Updating threshdict: %s'%(str(newentry)))
        self.threshdict.update(newentry)
        self._thrcolors = self._thrcolors[1:]    
    
    def _onpick(self,event):
        thisobj = event.artist
        setattr(self._threshax,'picked_object',thisobj)#to be able to delete it again
  
    def _onclick(self,event):
        '''On doublerightclick a new threshold is created'''
        if event.dblclick and event.button == 3:
            mankeys = [key for key in list(self.threshdict.keys()) if key.count('man')] 
            newlab = 'man1' if len(mankeys)==0 else 'man%s'%(np.max([np.int(key.split('man')[-1]) for key in mankeys])+1)
            self._append_threshdict(newlab,event.xdata)
            self.draw_newpicks()
        else:pass 
    
    
    def _on_key(self,event):
        if event.key == 'd' and hasattr(self._threshax,'picked_object'):#you can only delete the auto-drawn ones below the zero-line
            lab = self._threshax.picked_object.get_label()
            if lab in list(self.threshdict.keys()):
                self._thrcolors = self._thrcolors +[self.threshdict[lab][1]]
                del self.threshdict[lab]
                
                self.undraw_oldpicks()
      
                self._threshax.picked_object = None
            

    
    def plotpick_thresholds(self,recObj,save_and_exit_button=True):
        from scipy.stats import scoreatpercentile
        from matplotlib.ticker import MultipleLocator
        self._thrcolors = ['Darkorange','Darkviolet','SaddleBrown','SteelBlue','OliveDrab','PaleVioletRed']
        self.threshdict = {}
        self._allpicklabels = []
        self.nMaxPickThresh = len(self._thrcolors) #maximal number of thresholds that can be shown in one plot
        self.pickcol = 'gold'
        diffcol = 'grey'
        if not hasattr(self,'recObj'): setattr(self,'recObj',recObj)   
        
        logger.debug('Setting up f_of_threshold plot for picking')
        f = plt.figure(facecolor='w')
        
        f.text(0.01,0.98,'Add thresh: Double Rightclick',color='k',ha='left',va='top',fontweight='bold')
        #f.text(0.99,0.98,'Pick thresh: Click+S',color=self.pickcol,ha='right',va='top',fontweight='bold')
        f.text(0.01,0.01,'Delete: Click+d',color='k',ha='left',va='bottom',fontweight='bold')
        f.subplots_adjust(right=0.83)
        
        ax2 = f.add_subplot(111)
        ax = f.add_subplot(111,sharex=ax2,frameon=False)
        #plotting slope
        dZ = np.diff(self.f_of_thresh)
        ax2.plot(self.threshx[:-1],dZ,color=diffcol,linewidth=1.5,alpha=0.8,label='_nolegend_')
        ax2.axhline(scoreatpercentile(dZ[dZ<0],100.-self.peakperc),linestyle='--',color=diffcol)
        
        ax2.set_ylim([dZ.min(),0])        
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.set_ylabel('diff(#)',color=diffcol,rotation=-90)
        
        ax.set_zorder(2.) #to make clicks go to here
        ax.plot(self.threshx,self.f_of_thresh,'k',linewidth=2,label='_nolegend_')        
        
        ax.set_xlim([np.min(self.threshx),np.max(self.threshx)])
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position('left')
        ax.set_ylabel('#(discharges)')
        ax.set_xlabel('threshold [z]')
        
        self.figures.append(f)
    
        logger.debug('Setting up plot showing threshold choices on data.')
        f2,axarr = plt.subplots(3,1,figsize=(16,7),facecolor='w',sharex=True, \
                                           gridspec_kw = {'height_ratios':[2,3,3]})
        f2.subplots_adjust(left=0.06,right=0.99,top=0.99,bottom=0.12)
        tax,pax,rax = axarr
        recObj.plot(['raw'],ax=rax)
        rax.yaxis.set_major_locator(MultipleLocator(1))
        rax.set_ylabel('mV',fontsize=11)
        
        pax = axarr[-2]
        tvec = np.linspace(0.,len(self.avg_power)/recObj.sr,len(self.avg_power))+self.offset
        pax.plot(tvec,self.avg_power,color='k',linewidth=2)
        pax.set_ylabel(r'$\Sigma$',fontsize=16)
        for myax in axarr: myax.set_xlim([recObj.start,recObj.stop])
        tax.set_ylim([0.,self.nMaxPickThresh])
        tax.set_axis_off()
        
        self.figures.append(f2)
        
        if save_and_exit_button:self.make_done_button(f,yanch=0.01,height=0.05)
        
        
        f.canvas.mpl_connect('pick_event', self._onpick)
        cid = f.canvas.mpl_connect('button_press_event', self._onclick)
        cid2 = f.canvas.mpl_connect('key_press_event', self._on_key)
        
        self._threshfig = f
        self._threshax = ax
        self._ax2 = ax2
        
        self._threshfigTrace = f2
        self._tax = tax
        self._pax = pax           
        
        
        logger.info('Filling plot with default thresholds...')
        mthresh,lthresh,_ = blipS.getZthresh_derivMethod(self.f_of_thresh, self.threshx, \
                                                    peakperc=self.peakperc,mode='bump',verbose=True) 
        logger.info('bump-thresh is: %1.2f, firstcross-thresh: %1.2f'%(mthresh,lthresh))
        
        self._append_threshdict('bump',mthresh)
        self._append_threshdict('firstcross',lthresh)        
        
        self.draw_newpicks()
        
        plt.show()
    
    

        
    def make_checklegend(self):
        from matplotlib.widgets import CheckButtons
        if hasattr(self,'checkax'): self.checkax.remove()
        objs = [obj for obj in self._threshax.get_children() if obj.get_label() in self.threshdict]
        self.labs = [str(obj.get_label()) for obj in objs]
        cols = [obj.get_color() for obj in objs]
        self.boollist= [self.threshmode==lab for lab in self.labs] #to make the default polarity checked
        self.checked_thresh = str(self.threshmode)
        height = 0.03+len(objs)*0.03
        if np.int(matplotlib.__version__[0])>=2: self.checkax = self._threshfig.add_axes([0.84, 0.75, 0.15, height], facecolor='w')
        else: self.checkax = self._threshfig.add_axes([0.84, 0.75, 0.15, height], axisbg='w')

        #self.checkax.set_axis_off()
        self.check = CheckButtons(self.checkax, self.labs, self.boollist)
        for ii,col in enumerate(cols): self.check.labels[ii].set_color(col)
        plt.draw()
        self.check.on_clicked(self._checkfn)
        
    def _checkfn(self,label):
        bools = np.array([self.check.lines[ii][0].get_visible() for ii in range(len(self.labs))])
        if np.sum(bools)==1:
            self.checked_lab = np.array(self.labs)[bools][0]
            logger.info('Picking thresh labeled: %s'%(self.checked_lab))    
            thr = self.threshdict[self.checked_lab][0]
            labname = 'man%1.2f'%(thr) if self.checked_lab.count('man') else str(self.checked_lab)
            logger.info('Setting new threshold threshmode: %s, zthresh: %1.2f'%(labname,thr))
            self.setparam('threshmode',labname)
            self.zthresh = np.float(thr)
            if self.checked_lab.count('man'):self.setparam('manthresh',thr) 
            self.setparam('selected_thresh', True)
                
        
    def draw_newpicks(self):
        oldlabs  = [obj.get_label() for obj in self._threshax.get_children() if obj.get_label() in self.threshdict]
        newlabs  = [lab for lab in list(self.threshdict.keys()) if not lab in oldlabs]
        self._allpicklabels += newlabs
        for lab in newlabs: self.drawthreshline(self._threshax,lab)
        #self._threshax.legend()
        self.make_checklegend()
        #filling gaps in deleted
        ymaxvals = np.array([obj.get_segments()[0][1][1] for obj in self._tax.get_children() \
                    if obj.get_label() in list(self.threshdict.keys())])
        if np.size(ymaxvals)>0:
            ystarts = ymaxvals[np.where(np.diff(ymaxvals)>1)[0]][:len(newlabs)]
            ystarts = np.r_[ystarts, np.arange(len(newlabs)-len(ystarts))+ymaxvals.max()]
        else:
            ystarts = np.arange(len(newlabs))
        
        ranks = np.argsort(np.array([self.threshdict[lab][0] for lab in newlabs]))
        for lab,y0 in zip(np.array(newlabs)[ranks],ystarts):
            self.drawgetSpectralticks(self._tax,lab,pax=self._pax,yext=[y0,y0+1])
       
        self._threshax.figure.canvas.draw()  
        self._tax.figure.canvas.draw()  
        self._pax.figure.canvas.draw() 
         
    def undraw_oldpicks(self):
        doomedlabels = [lab for lab in self._allpicklabels if not lab in list(self.threshdict.keys())]
        print('Undrawing %s'%(doomedlabels))
        for obj in self._threshax.get_children()+self._tax.get_children()+self._pax.get_children():
            
            if obj.get_label() in doomedlabels: obj.remove()
            elif hasattr(obj,'get_color'): 
                    if str(obj.get_color()) in self._thrcolors: obj.remove()    
        self._threshax.figure.canvas.draw()
        self._tax.figure.canvas.draw()  
        self._pax.figure.canvas.draw() 
    
        
        
    def plot_ampdetection(self,ax=None):
        if ax == None:
            f, ax = plt.subplots(1,1, figsize=(6.,6.),facecolor='w')
        _ = blipS.catch_undetectedBlips(self.spikes_spectral, self.data, sr=self.sr, \
                                    zthresh=self.amp_thresh, blip_margin=self.spike_margin, \
                                    dtime=self.dtime, pol=self.recObj.polarity,plotax=ax)
    
    def plot_examples(self):
        f = blipS.plot_blipsDetectedExamples(self.spikes_spectral+self.offset, self.spikes_amplitude+self.offset,self.data,\
                                 snipdur=self.snipdur,sr=self.sr,t_offset=self.offset,\
                                 artregions = self.arts_fused)
    
    
    def plot_spectrogramSnippet(self,tbound,retrieve = True,start0=False,**kwargs):
    
        import matplotlib.gridspec as gridspec
        from matplotlib.ticker import MultipleLocator

        getkwarg = lambda kwargdict,key,default: kwargdict[key] if key in kwargdict else default

        
        #figure-params      
        fwidth = getkwarg(kwargs,'fwidth',10)
        fheight = getkwarg(kwargs,'fheight',7)  
        dotsperinch = getkwarg(kwargs,'dotsperinch',10)#set to 0 if avg_lim should not be displayed at spectrogram
        rawcol = getkwarg(kwargs,'rawcol','k')
        spikecol = getkwarg(kwargs,'spikecol','r')
        powcol =  getkwarg(kwargs,'powcol','b')
        dotcol =  getkwarg(kwargs,'dotcol','deepskyblue')
        cmap = getkwarg(kwargs,'cmap','gray')
        

        
        sr = self.recObj.sr
        ww = self.window
        
        #getting power and raw snippets
        ptsraw = (np.array(tbound)*sr).astype('int')
        ptspow = ptsraw - np.int(self.offset*sr) 

        rawsnip = self.recObj.raw_data[ptsraw[0]:ptsraw[1]]
        powsnip = self.avg_power[ptspow[0]:ptspow[1]]
        
        #time
        Tpts = np.int(np.diff(tbound)*sr)
        tvec = np.linspace(0,np.diff(tbound),Tpts) if start0 else np.linspace(tbound[0],tbound[1],Tpts) 
        
        Ndots = fwidth*dotsperinch #to display the avg_lim
        dotpos = np.linspace(tvec[0],tvec[-1],Ndots)
        
        #getting spectrogram
        spraw = self.recObj.raw_data[ptsraw[0]-np.int(ww/2.):ptsraw[1]+np.int(ww/2.)-1]
        sp = blipS.get_spectrogram(spraw,ww)
        highnorm,lownorm = self.normfacs ###TO DO: calcualte apower if normfacs not there, make it a property!
        sp_norm = (sp-lownorm)/(highnorm-lownorm)
        sp_norm[sp_norm<0.], sp_norm[sp_norm>1.] = 0., 1.#with minmax this doesnt matter!
            
        
        #getting spikes and threshold
        sppath = self.recObj._get_filepath('spikedict0')
        if retrieve and os.path.isfile(sppath):
            print('retrieving')
            spdict = self.recObj._open_byExtension('spikedict0')
            zthresh = spdict['data']['zthresh']
            spikes = spdict['data']['spikes_spectral']
        else:
            zthresh = self.zthresh 
            spikes = self.spectral_spikes
        spiket = spikes[(spikes>=tbound[0]) & (spikes<=tbound[1])]
        spikep = (spiket*sr).astype('int') - ptsraw[0]
        if start0: spiket = spiket - tbound[0]
    
    
        gs = gridspec.GridSpec(3,1,height_ratios = [2.5,2.5,5.],hspace=0.05,wspace=0.)
        f = plt.figure(facecolor='w',figsize=(fwidth,fheight))
        f.subplots_adjust(left=0.05,right=0.99,top=0.98,bottom=0.07)
        #raw data plot
        rax = f.add_subplot(gs[0])
        rax.plot(tvec,rawsnip,rawcol)
        rax.vlines(spiket,3.5*np.std(rawsnip),4.5*np.std(rawsnip),spikecol,lw=2)
        rax.yaxis.set_minor_locator(MultipleLocator(1))
        rax.set_yticks(np.arange(-4,4,2))
        
        #avg_power
        pax = f.add_subplot(gs[1],sharex=rax)
        pax.plot(tvec,powsnip,powcol)
        pax.axhline(zthresh,color=spikecol,linestyle='--',linewidth=2)
        pax.plot(tvec[spikep],powsnip[spikep],'o',ms=5,mew=0,mfc=spikecol,alpha=0.7)
        pax.set_yticks([0,2,4])
        pax.set_ylim([-1.,5.5])
        pax.yaxis.set_minor_locator(MultipleLocator(1))
        
        #spectrogram
        sax = f.add_subplot(gs[2],sharex=rax)
        sax.imshow(sp_norm.T,cmap=cmap,origin='lower',aspect='auto',extent=[tvec[0],tvec[-1],0.,sr/2.],\
                   interpolation='nearest')#
        for limval,mark in zip(self.avg_lim,['^','v']):sax.plot(dotpos,np.ones(Ndots)*limval,mark,\
                                                               ms=6,mfc=dotcol,mew=0,zorder=20)
        sax.tick_params(tickdir='out', labelsize=12,pad=-5)
        sax.yaxis.set_minor_locator(MultipleLocator(10))
        sax.set_ylim([0.,sr/2.])
        sax.set_xlabel('Time [s]',fontsize=9.,fontweight='normal',labelpad = 2.)
        
        for ax,ylab in zip([rax,pax,sax],['mV','zS','Frequency [Hz]']):
            ax.set_ylabel(ylab,fontsize=9.,fontweight='normal',labelpad = 2.)
            ax.set_xlim([tvec[0],tvec[-1]])
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.tick_params(tickdir='out', labelsize=8,width=1.,pad=0.3,length=3)
        for ax in [rax,pax]: plt.setp(ax.get_xticklabels(), visible=False)
        

     
    def plot_detected(self,showlist = ['raw','power'],show_thresh=True,**kwargs):
        from scipy.stats import scoreatpercentile
        from matplotlib.ticker import MultipleLocator
        
        recObj = self.recObj
        if 'ax' in kwargs and len(showlist)==0:
            ax = kwargs['ax']
            for spikes,mycol in zip([self.spikes_spectral,self.spikes_amplitude],[self.spectral_color,self.amp_color]):
                ax.vlines(spikes+self.offset,1,2,color=mycol)
            ax.set_ylim([1,2])
            ax.set_yticks([])
        
        else:
            f,axarr = plt.subplots(len(showlist)+1,1,figsize=(16,5),facecolor='w',sharex=True, \
                                   gridspec_kw = {'height_ratios':[1]+len(showlist)*[3]})
            axcount=len(axarr)-1
            if 'raw' in showlist:
                ax = axarr[axcount]
                recObj.plot(['raw'],ax=ax)
                ax.yaxis.set_major_locator(MultipleLocator(1))
                ax.set_ylabel('mV',fontsize=11)
                axcount -= 1
            if 'power' in showlist:
                ax = axarr[axcount]
                tvec = np.linspace(0.,len(self.avg_power)/recObj.sr,len(self.avg_power))+self.offset
                ax.plot(tvec,self.avg_power,color='k',linewidth=2)
                if show_thresh:
                    ax.axhline(self.zthresh,color=self.spectral_color,linestyle='--',linewidth=2)
                ax.set_ylabel('z(power)',fontsize=11)
                axcount -= 1   
            for spikes,mycol in zip([self.spikes_spectral,self.spikes_amplitude],[self.spectral_color,self.amp_color]):
                ax = axarr[axcount]
                ax.vlines(spikes+self.offset,1,2,color=mycol)
                ax.set_ylim([1,2])
                ax.set_yticks([])
            for ax in axarr: ax.set_xlim([recObj.start,recObj.stop])
    
            f.subplots_adjust(top=0.85,left=0.04,right=0.99,hspace=0.)
            f.text(0.01,0.98,'spectral',color=self.spectral_color,ha='left',va='top',fontweight='bold')
            f.text(0.99,0.98,'amp',color=self.amp_color,ha='right',va='top',fontweight='bold')
            f.show()
            return f


    def writeget_results(self,**kwargs):
        
        
        if not hasattr(self,'recObj'):            
            if 'recObj' in kwargs:self.set_recObj(kwargs['recObj'])
            else: logger.error('Need to specify recObj as kwarg')
        
        self.sr = np.float(self.recObj.sr)    

        if self.save_figs:
            fdir = os.path.join(self.recObj.figpath,self.recObj.id+'__spikeDetection')
            hf.createPath(fdir)
            
            for ftag,plotfn in zip(['__FofThresh','__ampDetection','__detExamples'],\
                                   [self.plot_fofthresh,self.plot_ampdetection,self.plot_examples]):
                figname = os.path.join(fdir,self.recObj.id+ftag+self.figformat)
                plotfn()
                f = plt.gcf()
                self.figsave(f,figname,bbox_inches='tight')
                logger.info('Saved plot: %s'%(figname))
         
         
        if self.save_data:
           logger.info('Saving detected spikes to dictionary.')
           ddict = {}
           ddict['spikes_spectral'] = self.spikes_spectral + self.offset
           ddict['spikes_amp'] = self.spikes_amplitude + self.offset
           ddict['spikes'] = np.sort(np.r_[ddict['spikes_spectral'],ddict['spikes_amp']])
           ddict['fOfThresh'] = self.f_of_thresh
           ddict['zthresh'] = self.zthresh
           ddict['polarity'] = self.recObj.polarity
           
           ddict['mask_startStop_sec'] = self.arts_fused if self.consider_artifacts else None#np.array([[],[]])
           ddict['t_offset'] = self.offset 
           ddict['t_total'] = self.offset + len(self.data)/np.float(self.sr)
           temp = ddict['t_total'] - self.offset
           artdur = np.sum(np.diff(np.clip(self.arts_fused,self.offset,self.arts_fused.max()))) \
                    if np.size(self.arts_fused)>0 else 0
           ddict['t_analyzed'] =  temp - artdur

           #save path and link to raw data
           #todo continue here!
           rmode = 'w' if not os.path.isfile(self.recObj.resultsfileH) else 'r+'
           attr_keys = ['t_analyzed','t_offset','t_total','zthresh']

           with h5py.File(self.recObj.resultsfileH,rmode) as dst:
                if self.recObj.cfg_ana['Preprocessing']['groupkey'] in dst:
                   del dst[self.recObj.cfg_ana['Preprocessing']['groupkey']]
                if self.groupkey in dst:
                    del dst[groupkey]
                rawgroup = dst.create_group(self.recObj.cfg_ana['Preprocessing']['groupkey'])
                rawgroup.attrs['path'] = self.recObj.rawfileH
                grp = dst.create_group(self.groupkey)
                for key,vals in ddict.items():
                    if key in attr_keys:
                        grp.attrs[key] = vals
                    else:
                        mytype = 'i' if key=='fOfThresh' else 'f'
                        grp.create_dataset(key,data=vals,dtype=mytype)

                igr = fdest.create_group('info')
                for key, val in self.infodict.items():
                   igr.attrs[key] = val

                method_dskeys = ['avg_lim','manthresh','norm','thr_range']
                mgr = fdest.create_group('methods')
                for key,vals in self.methodsdict.items():
                    mgr.attrs[key] = vals


                mgr.create_dataset('avg_lim',np.array(self.avg_lim),dtype='f')
                if not type(self.manthresh) == type(None):
                    mgr.create_dataset('manthresh',np.array(self.manthresh),dtype='f')
                mgr.create_dataset('norm',np.array(self.norm),dtype='f')
                mgr.create_dataset('thr_range',np.array(self.thr_range),dtype='f')




           #hf.saveto_hdf5({self.recObj.cfg_ana['Preprocessing']['groupkey']: {'path':self.recObj.rawfileH}}, self.recObj.resultsfileH,\
           #               mergewithexisting=True,
           #            overwrite_groups=True,overwrite_file=False)
           if os.path.isfile(self.recObj.rawfileH): hf.makelink_hdf5(self.recObj.rawfileH, self.recObj.resultsfileH, self.recObj.cfg_ana['Preprocessing']['groupkey']+'/link')

           self.datadict = {key:val for key,val in list(ddict.items())}



           self._save_resultsdict(self.recObj)
           if self.save_apower:
               if not self.retrieved_apower:
                   logger.info('copying temporary files into results file')
                   for dsname in ['normfacs', 'apower']:
                       dspath = self.recObj.resultsfileH.replace('.h5', '_Temp_%s.h5' % (dsname))
                       if os.path.isfile(dspath):
                           internal_path = '/' + self.groupkey + '/data/'+dsname
                           hf.merge_hdf5_special(dspath,self.recObj.resultsfileH,internal_path)

           

class SpikeSorting(Analysis):
    
    def __init__(self,mode='explore'):
        for key,val in list(self.defaults.items()):
            setattr(self,key,val)                    
        for key,val in list(self.plot_dict.items()):
            setattr(self,key,val)
        if mode == 'standard':
            self.setparam('ncomp_list',[3])
            self.setparam('nclust_list',[5])
            

        self.inspectcol = 'c'
        self.pickedFPcol = 'm'
       
        self.method = 'PCA-based sorting of sparse discharges and removal of noise clusters.'
        
        self.figures = []
        self.doublets = []
        

    @property
    def plot_dict(self):
        return {'figformat': '.png',\
                'mydpi': 100,\
                'n_examples': 3}
        
    def writeget_results(self,recObj,**kwargs):
        '''kwargs will be passed to run_clustering and can be nclust, ncomp and noiseclust'''
        if not hasattr(self,'spikeobjs'):
            self.run_clustering(recObj,**kwargs)
        #figure-generation

        if self.save_figs:
            if not hasattr(self,'fdir'): 
                setattr(self,'fdir',os.path.join(recObj.figpath,recObj.id+'__spikesorting'))
                hf.createPath(self.fdir)
            
            if not hasattr(self,'clustfig'): 
                self.plot_pick_clusters()
                plt.close('all')
            
            #clean figure from interactive mess

            self._preparesave_clustfig()

            
            figname = os.path.join(self.fdir,recObj.id+'__clusterremoval_ncomp%s'%(str(self.ncomp))+self.figformat)
            self.figsave(self.clustfig,figname)
            logger.info('Saved Noise-Cluster-Removal figure at %s'%(figname))
            
            
    
            logger.info('FP-removal: plotting and saving %d random examples at %s'%(self.n_examples,self.fdir))
            for ii in range(self.n_examples):
                f = blipsort.plot_fpExamples(self.truetimes,self.noisetimes,np.array([]),\
                            self.recObj.raw_data,sr=self.recObj.sr\
                            ,cutwin = self.cutwin,t_offset=self.recObj.offset,\
                            boxblips=np.hstack(self.stimes),\
                            artregions = self.recObj.artifactTimes.T)
                f.suptitle('PCA Based FP Removal    ncomp: %s nclust:%s noise%s    pol:%s    Fig%d '\
                   %(self.ncomp,self.nclust,self.noiseclust,self.polarity,ii+1))
                figname = os.path.join(self.fdir,recObj.id+'__fpRemoval_ex%d'%(ii+1)+self.figformat)
                self.figsave(f,figname)
                
        plt.close('all') 
        
        #save-data
        if self.save_data:
            #copying data from previous analyses
            group_prev = recObj.cfg_ana['EdDetection']['groupkey']
            self.datadict = {}
            for key in ['t_offset','t_total','t_analyzed','mask_startStop_sec']:
                self.datadict[key] = hf.open_hdf5(recObj.resultsfileH,'/'+group_prev+'/data/'+key)

            for attr in ['polarity','individual_picks','noisetimes','doublettimes']: self.datadict[attr] = getattr(self,attr)
            self.datadict['spikes'] = getattr(self,'truetimes')
            self._save_resultsdict(recObj)

    def run_clustering(self, recObj,**kwargs):
        
        
        self.recObj = recObj
          
        paramlist = []   
        for attrname in ['ncomp','nclust','noiseclust']:
            if attrname in kwargs:
                attrval = kwargs[attrname]
                setattr(self,attrname,attrval)
            else: attrval = getattr(self,attrname)
            paramlist.append(attrval)
            
        ncomp,nclust,noiseclust = paramlist
        self.polarity = recObj.polarity
        
        logger.info('Running clustering, ncomp: %d, nclust: %d'%(ncomp,nclust))
        logger.info('Using polarity: %s'%(self.polarity))
        #just to play with different polarities, remove later
        
        #self.polarity = str(recObj.polarity) --> and replace by this
        
        self.pca_dict = blipsort.get_pca_dict(self.recObj.raw_data,self.recObj.spikes0,[self.ncomp],[self.nclust],sr=self.recObj.sr,\
                        sgParams=self.savgol_params,minsearchwin=self.minsearchwin,cutwin=self.cutwin,\
                        pol=self.polarity,t_offset=0.)
        
        if self.polarity=='mix':
            neg_times,neg_ids = self.pca_dict[self.ncomp][self.nclust][0][0],self.pca_dict[self.ncomp][self.nclust][1][0]
            pos_times,pos_ids = self.pca_dict[self.ncomp][self.nclust][0][1],self.pca_dict[self.ncomp][self.nclust][1][1]
            pols = np.array(['neg']*len(neg_times)+['pos']*len(pos_times))
            self.stimes = np.r_[neg_times,pos_times]
            clids = np.r_[neg_ids,pos_ids]+1
        else:
            self.stimes = self.pca_dict[self.ncomp][self.nclust][0]
            clids = self.pca_dict[self.ncomp][self.nclust][1]+1
            pols = np.array([self.polarity]*len(self.stimes))
        
        logger.info('Initializing %d sparse spikeobjects.'%(len(self.stimes)))   
        self.spikeobjs = [Spike(stime,refobj=self,parent=recObj) for stime in self.stimes]
        
        for sobj,cid,pol in zip(self.spikeobjs,clids,pols):
            sobj.clustid = cid
            sobj.pol = pol

        self.remove_doublets()

        #now set the ids
        for ii,sobj in enumerate(self.spikeobjs):setattr(sobj,'id',ii)

        self._set_noisebools()

    def remove_doublets(self):

        #getting the alignment for spikesorting
        allcenters = np.array([sobj.centertime for sobj in self.spikeobjs])
        sidx = np.argsort(allcenters) # sorting according to alignment time
        allc = allcenters[sidx]
        sobjs = np.array(self.spikeobjs)[sidx] # sorted spikeobjs

        #get indices of where spikeobj alignments occur in bursts
        isis = np.diff(allc)

        if np.sum(isis < self.doublethresh) > 0:

            logger.info('Removing doublets')
            burst_parts = np.where(isis < self.doublethresh)[0]
            burst_starts = np.append(burst_parts[0], burst_parts[np.where(np.diff(burst_parts) > 1)[0] + 1])
            burst_stops = np.append(burst_parts[np.where(np.diff(burst_parts) > 1)] + 1, burst_parts[-1] + 1)

            #removing doublets
            for start, stop in zip(burst_starts, burst_stops):
                burst_spikes = sobjs[start:stop + 1]
                # compare which one in the burst is closest to center --> can survive!
                centercent = np.mean([sobj.centertime for sobj in burst_spikes])
                survivor = burst_spikes[np.argmin(np.abs([sobj.time for sobj in burst_spikes] - centercent))]
                doomed = [sobj for sobj in burst_spikes if not sobj == survivor]
                self.doublets += doomed
                for doomedspike in doomed: self.spikeobjs.remove(doomedspike)
        else:
            logger.info('No doublets found')

        logger.info('N(doublets): %d, N(spikes): %d, N(all):%d'%(len(self.doublets),len(self.spikeobjs),len(allcenters)))



    def _set_noisebools(self):     
        #now set the ones in noiseclust as noise
        get_noisebools = lambda noiselist,spikeobj: 2*[spikeobj.clustid in noiselist]
        
        logger.info('Setting .isFP and .in_noiseclust for spikeobjs; noiseclusts: %s'%(str(self.noiseclust)))
        #checking whether noiseclust contains two lists, which would mean that you have separately assigned noiseclusters!
        if self.polarity=='mix' and str(self.noiseclust).count('[')==3:
            for sobj in self.spikeobjs:
                if sobj.pol=='neg':
                    sobj.isFP,sobj.in_noiseclust = get_noisebools(self.noiseclust[0],sobj)
                if sobj.pol=='pos':    
                    sobj.isFP,sobj.in_noiseclust = get_noisebools(self.noiseclust[1],sobj)
        else:
            for sobj in self.spikeobjs: 
                sobj.isFP,sobj.in_noiseclust = get_noisebools(self.noiseclust,sobj)




    
    def plot_clust_and_trace(self):
        self.plot_pick_clusters()
        self.plot_clustersOnData()
        self.visually_checked = True
        plt.show()
        
    @eam.lazy_property
    def colorlist(self):
        return blipsort.get_cluster_colors(nclust=self.nclust)
    
    def _get_mousecolor(self,mbutton):
        if mbutton == 1: col = self.inspectcol 
        elif mbutton == 3: col = self.pickedFPcol
        else : col = 'grey'
        return col
    
    def plot_clustersOnData(self):
        
        logger.info('Plotting clusters on trace.')
        #starting points to display ticks of spikes
        ylowmin = self.recObj.raw_data.min()-0.4
        yupmax = self.recObj.raw_data.max()+0.4
        groupmin = self.recObj.raw_data.min()-0.2
        groupmax = self.recObj.raw_data.max()+ 0.2
        
        self.groupspikes = [stime for stime in self.recObj.spikes0 if not stime in self.stimes]
        
                
        f = plt.figure(figsize=(16,4),facecolor='w')
        f.subplots_adjust(left=0.05,right=0.94,bottom=0.15)
        f.text(0.01,0.98,'Mark to inspect: L-Click tick',color=self.inspectcol,ha='left',va='top',fontweight='bold')
        f.text(0.99,0.98,'assign individual FP: R-Click tick',color=self.pickedFPcol,ha='right',va='top',fontweight='bold')
        f.text(0.5,0.98,'reset: M-Click tick',color='k',ha='center',va='top',fontweight='bold')
        
        
        #labels
        f.text(0.99,0.9,'artifacts',ha='right',va='top',color='khaki',fontweight='bold',fontsize=12)
        f.text(0.99,0.9-0.05,'group',ha='right',va='top',color='grey',fontweight='bold',fontsize=12)
        for cc,col in enumerate(self.colorlist):
            f.text(0.99,0.9-(cc+2)*0.05,'cl: %s'%(cc+1),ha='right',va='top',color=col,fontweight='bold',fontsize=12)
        
        ax = f.add_subplot(111)
        ax.plot(self.recObj.tdata,self.recObj.raw_data,'k',lw=0.5) #LFP
        
        #artifacts
        for artstart,artstop in self.recObj.artifactTimes: ax.axvspan(artstart,artstop,color='khaki',alpha=0.5)
        
        #ticks
        ax.vlines(self.groupspikes,groupmin,groupmax,color='grey',linewidth=2,alpha=0.5,zorder=-1)
        for sobj in self.spikeobjs:
            if sobj.pol=='pos': ax.plot([sobj.time,sobj.time],[0,yupmax],color=self.colorlist[sobj.clustid-1],label=sobj.id,picker=4,zorder=-1)
            elif sobj.pol=='neg': ax.plot([sobj.time,sobj.time],[ylowmin,0.],color=self.colorlist[sobj.clustid-1],label=sobj.id,picker=4,zorder=-1)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('mV')
        
        self._traceax = ax
        self.tracefig = f
        self.figures.append(self.tracefig)
        self.tracefig.canvas.mpl_connect('pick_event', self._onpick_trace)
        
        self.visually_checked = True
        
        
        
    def plot_pick_clusters(self,done_button=True):
        from matplotlib import gridspec   
             
        logger.info('Creating clusterplot, polarity: %s'%(self.polarity))
        #now interactive clusterremoval, make this a function: plot_interactive_clustremoval
        ncols = self.nclust#np.max(np.unique(np.hstack(self.cids)))+1#max works also when list contains single element
        self._datastd = self.recObj.raw_data.std()
        
         
        if self.polarity == 'mix':
            nrows = 2
            figD,fbottom,ftop = (ncols*2.5+2,5),0.12,0.92#that is mixed polarity
            
        else: 
            nrows=1
            figD,fbottom,ftop = (ncols*2.5+2,3),0.2,0.85
        
        
        self.checklist = []
        f = plt.figure(facecolor='w',figsize=figD)
        f.subplots_adjust(left=0.08,bottom=fbottom,right=0.98,top=ftop)
        f.suptitle('Polarity: %s'%self.polarity)
        
        #label it to delete later
        lab1 = f.text(0.01,0.98,'mark to inspect: L-Click waveform',color=self.inspectcol,ha='left',va='top',fontweight='bold')
        lab2 = f.text(0.99,0.98,'assign individual FP: R-Click waveform',color=self.pickedFPcol,ha='right',va='top',fontweight='bold')
        self._clustpicktexts = [lab1,lab2]
        
        self.clustfig = f
        

        gsMain = gridspec.GridSpec(nrows, ncols)
        for row in range(nrows):
            if self.polarity == 'pos': pol_factor =-1
            elif self.polarity == 'neg': pol_factor = 1
            #print self.polarity
            if self.polarity == 'mix':
                if row == 0:
                    pol_factor = - 1
                    sobjs = [sobj for sobj in self.spikeobjs if sobj.pol=='pos']
                elif row == 1:
                    pol_factor = 1
                    sobjs = [sobj for sobj in self.spikeobjs if sobj.pol=='neg']
            else:                
                sobjs = self.spikeobjs
                
            axlist = [self.clustfig.add_subplot(gsMain[row,cc]) for cc in range(self.nclust)]
            self.make_panel_row(sobjs,axlist,pol_factor=pol_factor)
            
            if self.polarity == 'mix':
                if row==1: f.text(0.01,0.2,'pol: neg',rotation=90,ha='left',va='bottom',fontweight='bold')
                elif row==0:f.text(0.01,0.7,'pol: pos',rotation=90,ha='left',va='bottom',fontweight='bold')
        
        if self.polarity == 'mix': self.checklist = self.checklist[::-1]
        self.clustfig.canvas.mpl_connect('pick_event', self._onpick_clust)
        self.figures.append(self.clustfig)   
    
        if done_button: self.make_done_button(f,xanch=0.95,yanch=0.025,width=0.045,height=0.05)
        self.clustfig.canvas.draw()
    
    
    def _preparesave_clustfig(self):
        '''removes unnecessary features from interactive cluster-figure'''
        
        spike_from_line = lambda thisline: self.spikeobjs[np.int(thisline.get_label())]
        #remove interactive instructions
        for mytext in self._clustpicktexts: mytext.remove()
        
        #recolor that were handpicked but in the noisecluster to grey
        handpicked_lines = [myline for ax in self.clustfig.axes for myline in ax.lines if myline.get_color()==self.pickedFPcol]
        handpicked_lines_noiseclust = np.array([myline for myline in handpicked_lines if spike_from_line(myline).in_noiseclust])
        logger.info('N handpicked in noiseclust: %d'%(len(handpicked_lines_noiseclust)))
        for clustline in handpicked_lines_noiseclust: self._respond_clustpick(2,clustline)
        
        #remove blue lines from inspection
        handpicked_lines = [myline for ax in self.clustfig.axes for myline in ax.lines if myline.get_color()==self.inspectcol]
        for clustline in handpicked_lines: self._respond_clustpick(2,clustline)
        
        #new legend
        self.clustfig.text(0.01,0.98,'FP clusters are: %s'%(str(self.noiseclust)),ha='left',va='top')
        self.clustfig.text(0.99,0.98,'individually assigned FPs',color=self.pickedFPcol,ha='right',va='top',fontweight='bold')
        
        #remove done button
        if hasattr(self,'sandexax'):  self.sandexax.set_visible(False)
        
        plt.draw()

    def _onpick_trace(self,event):

        tracetick = event.artist
        #self.tracetick = tracetick
        
        #spikeid = np.int(tracetick.get_label())
        #logger.info('Picked spike: %d'%(spikeid))

        self._respond_tracepick(event.mouseevent.button,tracetick)
        
        #first check whether other plot even exists
        if hasattr(self,'checklist'): 
            clustline = [myline for ax in self.clustfig.axes for myline in ax.lines if myline.get_label()==tracetick.get_label()][0]
            self._respond_clustpick(event.mouseevent.button,clustline)
        else:
            logger.warning('Cluster figure does not exist. Use self.plot_pick_clusters in addition!')
    
    def _onpick_clust(self,event):
        
        clustline = event.artist
        
        #draw in cluster accordingly
        self._respond_clustpick(event.mouseevent.button,clustline)

        tracetick = [ch for ch in self._traceax.get_children() if ch.get_label()==clustline.get_label() and ch.get_marker() not in ['^','v']][0]
        self._respond_tracepick(event.mouseevent.button,tracetick)

    
    def _respond_tracepick(self,mbutton,tracetick):

        xpos = tracetick.get_xdata()[0]
        
        spikeid = np.int(tracetick.get_label())

        #delete previous marker
        mymarker = [ch for ch in self._traceax.get_children() if ch.get_label()==tracetick.get_label() and ch.get_marker() in ['^','v']]
        if len(mymarker)>0: 
            mymarker[0].remove()
            del mymarker[0]
            #print 'removed marker ',tracetick.get_label()
                
        if self.spikeobjs[spikeid].pol == 'neg': mark,ypos = '^',tracetick.get_ydata()[0]-0.2 
        elif self.spikeobjs[spikeid].pol == 'pos': mark,ypos = 'v',tracetick.get_ydata()[1]+0.2     
        if mbutton in [1,3]:
            col = self._get_mousecolor(mbutton)
            self._traceax.plot(xpos,ypos,marker=mark,ms=15,mec=col,mfc=col,label=tracetick.get_label())
            if col == self.pickedFPcol: self.spikeobjs[spikeid].isFP = True
        
        if not mbutton == 2:    
            spiketime = self.spikeobjs[spikeid].time
            if  spiketime >= self._traceax.viewLim.xmax or  spiketime <= self._traceax.viewLim.xmin:
                winw = self._traceax.viewLim.width
                self._traceax.set_xlim([spiketime-winw/2.,spiketime+winw/2.])
            
        self.tracefig.canvas.draw()    

        
    def _respond_clustpick(self,mbutton,clustline):
        
        if mbutton in [1,3]:
            maxZ = np.max([ll.zorder for ll in clustline.axes.lines])
            clustline.set_zorder(maxZ+1)
        elif mbutton == 2: 
            clustline.set_color('grey')
            clustline.set_zorder(0)
        col = self._get_mousecolor(mbutton)
        clustline.set_color(col)
        
        spikeid = np.int(clustline.get_label())
        if col == self.pickedFPcol: self.spikeobjs[spikeid].isFP = True        
        
        
        try:self.clustfig.canvas.draw()
        except: pass
        
    

    def make_panel_row(self,sobjs,axlist,pol_factor=1):
        from matplotlib.widgets import CheckButtons
        from matplotlib.ticker import MaxNLocator
        
        y_lim =[-12.*self._datastd,6*self._datastd]
        if pol_factor ==-1: y_lim = [-1*y_lim[1],-1*y_lim[0]]
        
        tvec = np.arange(-self.cutwin[0],self.cutwin[1]-1./self.recObj.sr,1./self.recObj.sr)
        
        nclust = self.nclust
        
        spiketimes = np.array([sobj.time for sobj in sobjs])      
        snips = blipsort.get_waveformSnippets(spiketimes,self.recObj.raw_data*pol_factor,\
                                                  sr=self.recObj.sr,minwin=self.minsearchwin,\
                                                  blipint = self.cutwin)
        for ss,sobj in enumerate(sobjs): sobj.snip = snips[ss]
        
        checkdict={}
        #change this for mix! nclust should be indexed from list then
        for cc in range(nclust):
            ax = axlist[cc]
            clustspikes = [sobj for sobj in sobjs if sobj.clustid == cc+1]
            avg = np.mean([sobj.snip for sobj in clustspikes],axis=0)

            minpt = np.argmin(avg)
            ptp = np.max(avg[minpt:])-avg[minpt]

            
            for sobj in clustspikes: ax.plot(tvec,sobj.snip*pol_factor,color='grey',label=sobj.id,picker=5)#multiplying by pol-factor re-inverst
            ax.plot(tvec,avg*pol_factor,self.colorlist[cc],linewidth=2)
            ax.text(0.98,0.05,'PTP %1.2f'%(ptp),fontsize=12,color= self.colorlist[cc],transform=ax.transAxes,ha='right',va='bottom',fontweight='bold')
            ax.text(0.98,0.97,'N: %d'%(len(clustspikes)),fontsize=10,color='k',transform=ax.transAxes,ha='right',va='top')
            ### here: cluster number
            ###here: checkbox in color --> if picked set isFP to True

            #polarity checkboxes
            ccbool = True if cc+1 in self.noiseclust else False
            pos = ax.get_position()
            checkax = self.clustfig.add_axes([pos.xmin+0.01*pos.width, pos.ymin-0.12*pos.height, 0.3*pos.width, 0.4*pos.height])        
            checkax.set_axis_off()
            check = CheckButtons(checkax,  ['cl %d is FP'%(cc+1)], [ccbool])
            check.labels[0].set_color(self.colorlist[cc])
            check.on_clicked(self._checkfn)
            checkdict[cc] = {'ax':checkax,'check': check,'bool':ccbool}
            
            ax.set_xticks([-0.2,-0.1,-0.,0.1,0.2,0.3,0.4,0.5])
            ax.set_yticks(np.arange(-8,8,1))
     
            ax.set_xlabel('Time [s]')            
            ax.set_ylim(y_lim)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            if cc==0: ax.set_ylabel('mV')
            else: ax.set_yticklabels([''])
            ax.set_xlim([-self.cutwin[0],self.cutwin[1]])
        self.checklist.append(checkdict)

    def _checkfn(self,label):
        for checkdict in self.checklist:
            for cc in sorted(checkdict.keys()):
                checkdict[cc]['bool'] = checkdict[cc]['check'].lines[0][0].get_visible()

        if len(self.checklist)==1:
            self.noiseclust = [cc+1 for cc in sorted(checkdict.keys()) if checkdict[cc]['bool']]
        else: self.noiseclust = [[cc+1  for cc in sorted(checkdict.keys()) if checkdict[cc]['bool']] for checkdict in self.checklist]
        logger.info('Current noise clusters: %s'%(self.noiseclust))
        
        self._set_noisebools()
        


    @property
    def individual_picks(self):
        if hasattr(self,'clustfig'):
            handpicked = [self.spikeobjs[np.int(myline.get_label())] for ax in self.clustfig.axes for myline in ax.lines if myline.get_color()==self.pickedFPcol]
            return np.array([sobj.time for sobj in handpicked if not sobj.in_noiseclust])
        else: return np.array([])
    
    @property
    def noisetimes(self):
        return np.unique([sobj.time for sobj in self.spikeobjs if sobj.isFP==True])

    @property
    def doublettimes(self):
        return np.array([sobj.time for sobj in self.doublets])

    @property
    def truetimes(self):
        return np.array([stime for stime in self.recObj.spikes0 if not stime in np.r_[self.noisetimes,self.doublettimes]])
        
    

class Spike(object):
    def __init__(self,stime,refobj=None,parent=None):
        
        if refobj: self.refobj = refobj
        if parent: self.parent = parent
        self.id = id
        self.time = stime

    @property
    def id(self):
        if not hasattr(self,'_id'): self._id = 'na'
        return self._id

    @id.setter
    def id(self,idval):
        self._id = idval

    @property
    def manualFP(self):
        if not hasattr('_manualFP'): self._manualFP = False
        return self._manualFP 
    
    @manualFP.setter
    def manualFP(self,boolval):
        self._manualFP = boolval
        
    
    @property
    def clustid(self):
        if not hasattr(self,'_clustid'): self._clustid = 'na'
        return self._clustid 
    
    @clustid.setter 
    def clustid(self,clustval):
        self._clustid = clustval
        
    @property
    def mintime(self):
        if not hasattr(self,'_mintime'): self._mintime = 'na'
        return self._mintime

    @property
    def centertime(self):


        if not self.pol in ['pos','neg']:
            logger.warning('Need to set valid (pos/neg) polarity of spike before evaluating centertime!')
            return np.nan

        if not hasattr(self,'_centertime'):

            pfac = -1 if self.pol == 'pos' else 1
            sr = self.parent.sr
            startM,stopM = self.refobj.minsearchwin
            startC, stopC = self.refobj.cutwin
            if np.logical_and(self.time < (self.parent.stop - stopC), self.time > startC):
                pstart, pstop = np.int(self.time*sr - startM*sr),np.int(self.time*sr + stopM*sr)
                self._centertime = self.time - startM + np.argmin(pfac*self.parent.raw_data[pstart:pstop])/sr
            else: return np.nan
        return self._centertime


    @property
    def isFP(self):
        if not hasattr(self,'_isFP'): self._isFP = 'na'
        return self._isFP 
    
    @isFP.setter 
    def isFP(self,boolval):
        self._isFP = boolval    
        
    @property
    def in_noiseclust(self):
        if not hasattr(self,'_in_noiseclust'): self._in_noiseclust = 'na'
        return self._in_noiseclust 
    
    @in_noiseclust.setter 
    def in_noiseclust(self,boolval):
        self._in_noiseclust = boolval    
    
    @property
    def cutout_int(self):
        if hasattr(self,'refobj'):
            return [self.time-self.refobj.cutwin[0],self.time+self.refobj.cutwin[1]]
        else: return None
    
    @property
    def cutout(self):
        if hasattr(self,'refobj') and hasattr(self,'parent'):
            pstart,pstop = np.array(self.cutout_int)*self.parent.sr 
            return self.parent.raw_data[np.int(pstart):np.int(pstop)]
        else: return None
        
    @property
    def snip(self):
        if not hasattr(self,'_snip'): self._snip = None
        return self._snip 
    
    @snip.setter
    def snip(self,sniptrace):
        self._snip = sniptrace
    
    @property
    def pol(self):
        if not hasattr(self,'_pol'): self._pol = None
        return self._pol
    
    @pol.setter        
    def pol(self,polaritystr):
        self._pol = polaritystr

            
def set_quality(recObj,**kwargs):
    '''How sure can YOU (not the algorithm) distinguish discharges/EA from non-discharges/non-EA?'''
    
    if 'quality' in kwargs: 
        logger.debug('reading quality from kwargs')
        quality = kwargs['quality'].strip()
    else:
        logger.debug('asking for quality interactively')
        quality = str(input('Valid qualities:%s \nEnter quality: '%(recObj._quality_range)))

        
    assert quality in recObj._quality_range,'Quality %s invalid, please choose from %s'%(quality,str(recObj._quality_range))
            
    recObj.quality = quality
    assessedby = getpass.getuser()
    logger.info('Writing data-quality: %s'%(quality))
    tf = open(recObj._get_filepath('quality'),'w')
    tf.write(quality)
    tf.write('\nAssessed by: %s'%assessedby)
    tf.close()
        
        
def write_comment(recObj,**kwargs):
    stopword = 'd'
    assessedby = getpass.getuser()
    tf = open(recObj._get_filepath('comments'),'w')
    if 'comment' in kwargs: 
        logger.debug('reading comment from kwargs')
        tf.write(kwargs['comment'])
        
    else:
        logger.debug('asking for comment interactively')
        print('Please comment on analysis (anything odd/noteworthy), press %s to close:'%(stopword))
        while True:
            comment = str(input())
            if comment.strip() == stopword:break    
            tf.write(comment+'\n')
    tf.write('\n\nAssessed by: %s'%assessedby)
    tf.close()

