if not 'settings_run' in locals():from settings_run import *

#create paths
subrun = cfg['preprocessing']['loading']
eam.checkmakepaths(aRec.rawfileH)
setparams = lambda cobj,cdict: [cobj.setparam(item[0],item[1]) for item in list(cdict[type(cobj).__name__]['setparams'].items())] \
                               and None if type(cdict[type(cobj).__name__]['setparams'])==dict else None


if multiproc_on_preproc:
    chanid = 'NA' if subrun['channel'] == 'interactive' else subrun['channel']
    PP = cmp.Preprocessing(subrun['source'], chanid=chanid)
    #setparams(PP, cfg)
    PP.writeget_result(aRec, to_file=aRec.rawfileH)

else:

    if subrun['channel'] == 'interactive':
        ddict = edd.open_source(subrun['source'])#works for only one channel in this mode
        chan = list(ddict.keys())[0]
    else:
        chan = subrun['channel']
        ddict = edd.open_source(subrun['source'],chanlist=[chan])#works for only one channel in this mode

    print('###### Resampling... ######')

    PP = edd.Preprocessing(ddict[chan]['trace'],ddict[chan]['sr'],chanid=chan,\
                        moreinfo=ddict[chan]['moreinfo'],sourcefile=subrun['source'])

    for item in list(cfg['preprocessing'].items()):
        if not item[0] == 'loading':
            PP.setparam(item[0], item[1])



    PP.write_result(aRec,to_file=aRec.rawfileH)

