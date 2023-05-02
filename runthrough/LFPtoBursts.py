if not 'settings_run' in locals():from settings_run import *

import core.ea_analysis as eana
subrun = cfg['analysis']


checkrun = lambda cobj: type(cobj).__name__ in subrun['run']#to check whether the respective analysis should be run
#setparams = lambda cobj: [cobj.setparam(item[0],item[1]) for item in list(subrun[type(cobj).__name__]['setparams'].items())] and None if type(subrun[type(cobj).__name__]['setparams'])==dict else None
setparams = lambda cobj,cdict: [cobj.setparam(item[0],item[1]) for item in list(cdict[type(cobj).__name__]['setparams'].items())] and None if type(cdict[type(cobj).__name__]['setparams'])==dict else None



aRec.create_paths()

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
# ANALYSIS
if mp_on:
    MP = cmp.Multiprocessing()
    MP.update_settings(cfg)
###############################################
## DISCHARGE DETECTION
check_mp = lambda subclass: getattr(MP,subclass)['flag'] if 'MP' in locals() else False

if check_mp('EdDetection'):
    EDD = cmp.EdDetection()
    if checkrun(EDD):
        print ('Multiprocessing ED detection')

        setparams(EDD, subrun)

        EDD.init_from_raw(aRec.rawfileH)
        EDD.set_mp_params(MP)  # needs to be after init_from_raw to have the sr!
        #artifactfile = "/home/weltgeischt/EAnalysis_EXAMPLES/Blondiaux/DATA/Bs3_examples_12/Bs3_examples_12__artifacts.txt"
        EDD.set_artifactfile(aRec.artifactfile)
        EDD.writeget_results(aRec)


else:
    EDD = edd.EdDetection()
    if checkrun(EDD):
        if 'take_params_from' in subrun[type(EDD).__name__]:
            if len(subrun[type(EDD).__name__]['take_params_from'])>0:
                import subprocess
                print ('Calling victimize_spikedetection.py to apply params from %s to %s'%(subrun[type(EDD).__name__]['take_params_from'],aRec.id))
                codefile = os.path.join(cfg['settings']['code_path'],'core_extensions','victimize_spikedetection.py')
                subprocess.call([sys.executable,codefile,aRec.init_ymlpath])
        else:
            setparams(EDD,subrun)
            if subrun['EdDetection']['interactive']: EDD.plotpick_thresholds(aRec)
            else: EDD.set_recObj(aRec)
            EDD.writeget_results()

###############################################
## SPIKE SORTING
if check_mp('SpikeSorting'):
    SPS = cmp.SpikeSorting()
    if checkrun(SPS):
        setparams(SPS,subrun)
        SPS.set_mp_params(MP)
        SPS.writeget_results(aRec,plot_clusters=False,mp_mode='map')
else:
    SPS = edd.SpikeSorting()
    if checkrun(SPS):
        setparams(SPS,subrun)
        if subrun['SpikeSorting']['interactive']:
            SPS.run_clustering(aRec)
            SPS.plot_clust_and_trace()
        SPS.writeget_results(aRec)


###############################################
## Burst Classification
if check_mp('BurstDetection'):#take care its the flag for BurstDetection here and not BurstClassification
    BC = cmp.BurstClassification()
    if checkrun(BC):
        setparams(BC,subrun)

        #first get the parameters from burst detection
        BD = eana.BurstDetection()
        try:setparams(BD, subrun)  # works only if burstdetection is specified in the ymlfile
        except:print ('Taking default parameters for burst detection (params not specified in %s)' % initpath)
        BC.set_maxdist_mergelim(BD.maxdist,BD.mergelim)
        BC.writeget_results(aRec)

else:
    BC = eana.BurstClassification()
    if checkrun(BC):
        setparams(BC,subrun)
        BC.run(aRec)


###############################################
## State Analysis
SA = eana.StateAnalysis()
if checkrun(SA):
    setparams(SA, subrun)
    SA.run(aRec)


###############################################
## Diagnostics
# TODO mayor entries and savings

D = eana.Diagnostics()
if checkrun(D):
    setparams(D,subrun)
    D.write(aRec)
