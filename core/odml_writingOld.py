from __future__ import print_function
import os
import numpy as np
import logging
import odml
import ea_management as eam

#create an analysis section just for one address
def fetch_analysis(resultsfile,internal_path,include=['methods','info']):
    return subsection

def get_all_analysis(resultsfile):
    supersection
    for key in ...
        fetch_analysis
        append to supersection
    return supersection

def get_diagnostics

def write_odml





logger = logging.getLogger(__name__)

mockval = odml.Value(data='tba')


def makewrite_odml(recObj, overwrite=True, **kwargs):
    import time
    import pwd

    print ('To do- pass and warning when already exists or remove existing')

    logger.info('creating odml-analysis object for %s' % (recObj.id))
    Mother = EAanalysisMother(recObj)
    Mother.gather_children()

    odmlpath = kwargs['odmlpath'] if kwargs.has_key('odmlpath') else recObj.odmlpath

    if odmlpath is None:
        logger.info('No odmlpath available for %s --> creating new odml in place.' % (recObj.id))
        author = pwd.getpwuid(os.geteuid())[0]
        mydate = time.strftime('%Y-%m-%d')
        odmldoc = odml.Document(author=author, date=mydate)
        odmlpath = os.path.join(recObj.fullpath, recObj.id + '__metadata.odml')
        odmldoc.append(odml.Section(name=recObj.id, type='dataset'))
        anchorsect = odmldoc.sections[recObj.id]

    else:
        logger.info('Finding anchor for odml-section in %s' % (odmlpath))
        odmldoc = odml.tools.xmlparser.load(recObj.odmlpath)
        anchorsect = \
        list(odmldoc.itersections(filter_func=lambda x: x.get_path().count(recObj.id) and x.type == 'dataset'))[0]
        # list(odmldoc.itersections(filter_func=lambda x:x.properties['Id'].value.data==recObj.id))[0]

    if overwrite:
        siblingnames = [sect.name for sect in anchorsect.sections]
        if Mother.section.name in siblingnames:
            logger.info('Analysis section already exists and is overwritten.')
            doomedsect = anchorsect.sections[Mother.section.name]
            anchorsect.sections.remove(doomedsect)

    anchorsect.append(Mother.section)
    logger.debug('Writing odml.')
    odml.tools.xmlparser.XMLWriter(odmldoc).write_file(odmlpath)


class EAanalysisMother(object):
    def __init__(self, recObj, includeResults=True):
        self.recObj = recObj
        self.includeResults = includeResults

    @eam.lazy_property
    def section(self):
        sect = Analysis_EAmother().section
        sect.properties['FigurePathURL'].value.data = urljoin([self.recObj.figpath])
        sect.properties['DataPathURL'].value.data = urljoin([self.recObj.fullpath])
        if hasattr(self.recObj, 'dur0'): sect.properties['AnalysisDuration'].value.data = self.recObj.dur0
        sect.properties['Duration'].value.data = self.recObj.dur
        return sect

    @property
    def subsectkeys(self):
        if not hasattr(self, '_subsectkeys'):
            self._subsectkeys = ['raw', 'artifact_methods', 'spikedict0', 'spikedict', 'burstclasses', 'states']
        return self._subsectkeys

    @subsectkeys.setter
    def subsectkeys(self, mykeys):
        self._subsectkeys = mykeys

    def create_append_child(self, extensionkey):

        if os.path.isfile(self.recObj._get_filepath(extensionkey)):
            AO = EAanalysis(self.recObj, extensionkey)
            AO.fillsect_info()
            AO.fillsect_params()
            if self.includeResults:
                AO.fillsect_results()
            else:
                logger.info('Removing results section.')
                rsect = AO.section.sections['Results']
                AO.section.sections.remove(rsect)
            self.section.append(AO.section)
        else:
            logger.warning('File does not exist for %s' % (extensionkey))

    def gather_children(self):
        for ext in self.subsectkeys:
            logger.info('Appending %s to mother' % (ext))
            self.create_append_child(ext)
        self.section.append(QualityManagement(self.recObj).section)


class QualityManagement(object):
    def __init__(self, recObj):
        self.recObj = recObj
        self.stype = 'analysis/EA'

    @property
    def sectname(self):
        return 'QualityAndComment'

    @eam.lazy_property
    def section(self):
        logger.info('Creating odml-analysis section for %s, name: %s' % (self.recObj.id, self.sectname))

        sect = odml.Section(name=self.sectname, type=self.stype)
        # entering quality
        if os.path.isfile(self.recObj._get_filepath('quality')):
            tf = open(self.recObj._get_filepath('quality'), 'r')
            quality = tf.readline().strip()
            assessedBy = tf.readline().replace('Assessed by: ', '').strip()
            tf.close()
        else:
            quality, assessedBy = 'NA', 'NA'
            logger.info('No quality-assessment available, no file at %s.' % (self.recObj._get_filepath('quality')))

        # getting comments
        commentpath = self.recObj._get_filepath('comments')
        if os.path.isfile(commentpath):
            commentpathWrite = os.path.relpath(commentpath, os.path.commonprefix([self.recObj.fullpath, commentpath]))
            tf2 = open(commentpath, 'r')
            commentedBy = tf2.read().split('Assessed by: ')[-1].strip()
            tf2.close()

        else:
            commentpathWrite = 'NA'
            commentedBy = 'NA'
            logger.info('No comments available, no file at %s' % (commentpath))
        propdict = {'QualityRange': [' '.join(self.recObj._quality_range), '', 'string'], \
                    'Quality': [quality, '', 'string'], \
                    'QualityAssessedBy': [assessedBy, '', 'person'], \
                    'CommentPath': [commentpathWrite, '', 'string'], \
                    'CommentedBy': [commentedBy, '', 'person']}

        fillsect(propdict, sect)
        return sect


class EAanalysis(object):
    def __init__(self, recObj, extensionkey):
        self.recObj = recObj
        self.ext = extensionkey

    @eam.lazy_property
    def resultsfile(self):
        return self.recObj._open_byExtension(self.ext)

    @eam.lazy_property
    def infodict(self):
        return flatten_dict(self.resultsfile['info'])

    @eam.lazy_property
    def paramdict(self):
        return flatten_dict(self.resultsfile['methods'])

    @property
    def sectname(self):
        return self.infodict['Class']

    @eam.lazy_property
    def section(self):
        logger.info('Creating odml-analysis section for %s, name: %s' % (self.recObj.id, self.sectname))
        return Analysis_EA(name=self.sectname).section

    def fillsect_info(self):
        for key, val in sorted(self.infodict.items()):

            if key in self.section.properties:
                Prop = self.section.properties[key]
            else:
                Prop = odml.Property(name=key, value=mockval)

            if Prop.value.dtype == 'url':
                prefn = lambda myval: urljoin([myval])
            elif Prop.name == 'ResultFigure':
                prefn = lambda figname: self.remove_genpath(figname, mode='fig')
            else:
                prefn = lambda myval: myval

            if type(val) == list:
                self.listattach(val, Prop, fn=prefn)
            else:
                Prop.value.data = prefn(val)

            if not key in self.section.properties:
                logger.info('Introducing non-template property %s into section %s' % (key, self.section.name))
                self.section.append(Prop)

            self.section.properties['ResultFile'].value.data = self.remove_genpath(self.recObj._get_filepath(self.ext), \
                                                                                   mode='data')

    def fillsect_params(self):
        # from copy import deepcopy

        listables = [list, np.ndarray]

        logger.info('TO DO: Implement yaml-readout for units and definitions')

        sect = self.section.sections['Parameters']
        for key, val in sorted(self.paramdict.items()):
            if val is None: val = 'None'
            vinput = [odml.Value(data=thisval) for thisval in val] if type(val) in listables \
                else odml.Value(data=val)
            Prop = odml.Property(name=key, value=vinput)
            sect.append(Prop)

    def fillsect_results(self):
        logger.info('TO DO: Implement yaml-readout for units and definitions')
        sect = self.section.sections['Results']
        for key, val in sorted(self.resultsdict.items()):
            Val = odml.Value(data=val)
            Prop = odml.Property(name=key, value=Val)
            sect.append(Prop)

    @eam.lazy_property
    def resultsdict(self):
        if self.ext == 'raw': rdict = {}
        if self.ext == 'artifact_methods': rdict = {'nArtifacts': len(self.recObj.artifacts), 'tArtifacts': \
            np.sum(self.recObj.artifactTimes)}
        if self.ext == 'spikedict0': rdict = {'nSpikes': len(self.recObj.spikes0)}
        if self.ext == 'spikedict': rdict = {'nSpikes': len(self.recObj.spiketimes),
                                             'tFree': np.sum(self.recObj.freetimes)}
        if self.ext == 'burstclasses':
            bclasses = np.unique([burst.cname for burst in self.recObj.bursts])
            rdict = {'n' + bclass: len(eam.filter_objs(self.recObj.bursts, [lambda x: x.cname == bclass])) \
                     for bclass in bclasses}

            nsevere = len(eam.filter_objs(self.recObj.bursts, [lambda x: x.cname in eam.severe]))
            nmild = len(eam.filter_objs(self.recObj.bursts, [lambda x: x.cname in eam.mild]))
            nES = len(self.recObj.singlets)
            rdict.update(
                {'nES': nES, 'rSevere': nsevere / self.recObj.durAnalyzed, 'rMild': nmild / self.recObj.durAnalyzed})
        if self.ext == 'states':
            uniquestates = np.unique([S.state for S in self.recObj.states])
            rdict = {'t_' + statename: self.recObj.get_tStates(statename) for statename in uniquestates}
            tIIP = self.recObj.get_tStates('IIP')
            rMildIIP = 9999.  # deprecated! len(eam.filter_objs(self.recObj.bursts,[lambda x: x.cname in eam.mild]))/tIIP if tIIP>0. else ot.dfloat
            rdict.update({'rMild_inIIP': rMildIIP})

        return rdict

    def remove_genpath(self, path, mode='fig'):
        if mode == 'fig':
            commonprefix = os.path.commonprefix([self.recObj.figpath, path])
        elif mode == 'data':
            commonprefix = os.path.commonprefix([self.recObj.fullpath, path])
        return os.path.relpath(path, commonprefix)

    def listattach(self, vallist, prop, **kwargs):
        if kwargs.has_key('fn'): vallist = [kwargs['fn'](val) for val in vallist]
        vals = [odml.Value(data=val, unit=prop.value.unit, dtype=prop.value.dtype) for val in vallist]
        prop.value.data = vals


def flatten_dict(mydict):
    '''detects dictionaries in dictionaries and adds their entries as mykey__mysubkey: val to the top-level dictionary'''
    subdicts = {key: val for key, val in mydict.items() if type(val) == dict}
    topdict = {key: val for key, val in mydict.items() if not type(val) == dict}
    entries2 = {key + '__' + subkey: subval for key in subdicts.keys() for subkey, subval in subdicts[key].items()}
    topdict.update(entries2)
    return topdict


##FROM OLD odml_templates
#default values
dfloat = 9999.
ddate = '1900-01-01'
dstring = 'tba'
dtime = '00:00:00'
datedict = {'Date':[ddate,'','date','yyyy-mm-dd']}



def urljoin(ospath,scheme='file://'):
    return scheme+os.path.join(*ospath)

def fillsect(propdict,sect):
    for pname,vals in propdict.items():
        Val = odml.Value(data=vals[0],unit=vals[1],dtype=vals[2])
        Prop = odml.Property(name=pname,value=Val,definition=vals[3]) if len(vals)==4 else odml.Property(name=pname,value=Val)
        sect.append(Prop)
    return sect



class Analysis_EAmother():
    def __init__(self, name='EA_analysis'):
        self.stype = 'analysis'
        self.name = name

    @property
    def section(self):
        sect = odml.Section(name=self.name, type=self.stype)
        mainprops = {'Author': ['weltgesicht', '', 'person'], \
                     'FigurePathURL': [dstring, '', 'url'], \
                     'DataPathURL': [dstring, '', 'url'], \
                     'AnalysisDuration': [dfloat, 's', 'float', 'Duration-Offset due to wakeup'], \
                     'Duration': [dfloat, 's', 'float']}

        fillsect(mainprops, sect)
        return sect


class Analysis_EA():
    def __init__(self, name='AnalysisEA_X'):
        self.stype = 'analysis/EA'
        self.name = name

    @property
    def section(self):
        psub = odml.Section(name='Parameters', type='analysis/parameters')
        rsub = odml.Section(name='Results', type='analysis/results')
        mainsect = odml.Section(name=self.name, type=self.stype)

        mainprops = {'LogFile': [dstring, '', 'url'], \
                     'Method': [dstring, '', 'str', 'eg. power spectrum'], \
                     'Class': [dstring, '', 'str', 'eg. BurstClassification'], \
                     'Function': [dstring, '', 'str', 'eg. run'], \
                     'CodeFile': [dstring, '', 'url'], \
                     'CodeRevision': [dstring, '', 'string', 'git-hash'], \
                     'ResultFile': [dstring, '', 'url'], \
                     'ResultFigure': [dstring, '', 'str'], \
                     'DependsOn': [dstring, '', 'str', 'Section-Name of required sibling analysis section.'], \
                     'Time': [dtime, '', 'time', 'hh:mm:ss'], \
                     'User': [dstring, '', 'person'], \
                     'Host': [dstring, '', 'str', 'Name of machine.']}
        mainprops.update(datedict)

        fillsect(mainprops, mainsect)

        mainsect.append(psub)
        mainsect.append(rsub)
        return mainsect
