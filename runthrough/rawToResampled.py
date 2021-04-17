if not 'settings_run' in locals():from settings_run import *

#create paths
subrun = cfg['preprocessing']['loading']
eam.checkmakepaths(aRec.rawfileH)


if multiproc_on_preproc:
    chanid = 'NA' if subrun['channel'] == 'interactive' else subrun['channel']
    PP = cmp.Preprocessing(subrun['source'], chanid=chanid)
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



    PP.write_result(aRec,to_file=aRec.rawfileH)

