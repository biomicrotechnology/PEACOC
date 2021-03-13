
if not 'settings_run' in locals():from settings_run import *


eam.checkmakepaths(aRec._polaritypath)


POL = edd.Polarity()
POL.plot_and_pick(aRec,checkbox_on=True,save_and_exit_button=True)


POL.set_polarity(aRec,POL.checked_polarity,to_file=aRec._polaritypath)
