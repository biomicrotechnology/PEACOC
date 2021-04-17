#1) cd <directory where you saved PEACOC>
#2) cd PEACOC
#3) ipython --pylab #to start python
import core.ea_management as eam
import core.helpers as hf

############################
#ALTERNATIVE 1) ACCESS DATA TROUGH THE RECORDING OBJECT (Rec)

ymlfile = 'PATH_TO_YMLFILE' #this is the file you used in commandline_runthrough.txt and is specific for a recording, e.g.: '/home/weltgeischt/EXAMPLES/Enya/yml_setup/EP11_1Hz2_HCi1__runparams.yml'
ymlfile = '/media/weltgeischt/MALWINE/SCOOSIE/DATA/yml_setups/NP12_ipsi1_01_000__runparams.yml'
aRec = eam.Rec(init_ymlpath=ymlfile)
#or directly for the data (set the sompath directly, if you want colorfully plotted bursts!)
aRec = eam.Rec(init_datapath='/media/weltgeischt/MALWINE/SCOOSIE/ANIMAL_DATA/NP12/NP12_ipsi1/NP12_ipsi1_01_000/NP12_ipsi1_01_000__blipSpy.h5',sompath='/home/weltgeischt/workspace/PEACOC/config/som.h5')
aRec.plot(['raw','spikes','bursts'])


aRec.spikerate#number EDs / time analyzed


myburst=aRec.bursts[5]
#have fun with the output...
myburst.dur
myburst.si #seizure index
myburst.roi
myburst.cname


aRec.plot(['raw','artifacts','spikes','singlets','bursts'])


#get cutout from recording
myreccut = eam.EAPeriod(10*60.,15.*60.,parentobj=aRec)

myreccut.plot(['raw','artifacts','spikes','singlets','bursts'])

myreccut.spikerate




###################
#ALTERNATIVE 2):  YOU CAN ACCESS THE DATA THAT GOT SAVED DURING RUNTHROUGH DIRECTLY
#opening saved data
# import load_data

aRec._file_ext #shows you possible filepath options
burstfilepath = aRec._get_filepath('burstclasses')
burstdict = hf.open_obj(burstfilepath)
burstdict.keys()
burstdict['data'][42]
burstdict['data']['params']

