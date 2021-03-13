if not 'settings_run' in locals():from settings_run import *
from core.artisfaction import saveArtdict_txt

eam.checkmakepaths(aRec.artifactfile)

AD = edd.ArtifactDetection()
AD.plotcheck_artifacts(aRec) #--> here you click interatively
AD.harvest_artifacts()
saveArtdict_txt(AD.artdict,aRec.artifactfile,decpts = 2)

