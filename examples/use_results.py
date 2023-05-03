#1) cd <directory where you saved PEACOC>
#2) cd PEACOC
#3) ipython --pylab #to start python
import core.ea_management as eam
import core.helpers as hf

############################
#ALTERNATIVE 1) ACCESS DATA TROUGH THE RECORDING OBJECT (Rec)

ymlfile = 'PATH_TO_YMLFILE' #this is the file you used in commandline_runthrough.txt and is specific for a recording, e.g.: '/home/weltgeischt/EXAMPLES/Enya/yml_setup/EP11_1Hz2_HCi1__runparams.yml'
ymlfile = '/home/weltgeischt/EpilepsyProject/DATA_EXAMPLES/KA114_HC3_01__runparamsTest.yml'
aRec = eam.Rec(init_ymlpath=ymlfile)
#or directly for the data (set the sompath directly, if you want colorfully plotted bursts!)
ini_path = '/media/weltgeischt/MALWINE/SCOOSIE/ANIMAL_DATA/NP12/NP12_ipsi1/NP12_ipsi1_01_000/NP12_ipsi1_01_000__blipSpy.h5'#'/media/weltgeischt/MALWINE/SCOOSIE/ANIMAL_DATA/TEST/KA114/KA114_HC3/KA114_HC3_01/KA114_HC3_01__blipSpy.h5'
aRec = eam.Rec(init_datapath=ini_path,sompath='/home/weltgeischt/workspace/PEACOC/config/som.h5')
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
#opening saved data with h5py
import h5py
fname = '/media/weltgeischt/MALWINE/SCOOSIE/ANIMAL_DATA/TEST/KA114/KA114_HC3/KA114_HC3_01/KA114_HC3_01__blipSpy.h5'
with h5py.File(fname,'r') as hand:
    bmat = hand['burstclasses/data/values'][()]
    bparams = [el.decode() for el in hand['burstclasses/data/params'][()]]
print(bparams)
print(bmat.shape)