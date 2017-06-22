#! /usr/bin/env python

'''
Given APT output files, read in data relevant to the data simulator,
organize, and create input files for the simulator.


program number/id (proposal number) - xml
observation number - pointing file
visit number - pointing file
visit group - ?
obs_id -  ?  not present in keyword dictionary
visit_id -  ?  11characters = programID+observation+visit
sequence_id - ? something to do with parallels?
activity_id - just use a counter when reading in pointing file
             Or keep a running sum of 'Dith' column in pointing file
exposure_number - Exp column in pointing file? (e.g. all 12 dithers of grism
              observation get same exposure number)
observation_label - in xml and pointing file



'''

import argparse
from lxml import etree
from astropy.table import Table, Column
from astropy.io import ascii
import numpy as np
import sys,os
import collections
import rotations

#try to find set_telescope_pointing.py
try:
    import imp
    conda_path = os.environ['CONDA_PREFIX']
    set_telescope_pointing = imp.load_source('set_telescope_pointing', os.path.join(conda_path, 'bin/set_telescope_pointing.py'))
    stp_flag = True
except:
    print('WARNING: cannot find set_telescope_pointing.py. This is needed')
    print('to translate telescope roll angle to local roll angle.')
    sys.exit()
    
class AptInput:
    def __init__(self):
        self.input_xml = '' #'GOODSS_ditheredDatasetTest.xml'
        self.output_csv = None #'GOODSS_ditheredDatasetTest.csv'
        self.pointing_file = '' #'GOODSS_ditheredDatasetTest.pointing'
        self.siaf = ''
        

    def read_wfss_xml(self,infile):
        #read APT xml file for WFSS mode observations
        #first, set up variables 
        MyList = collections.OrderedDict()
        MyList['Module'] = []
        MyList['Subarray'] = [] 
        MyList['Grism'] = []
        MyList['PrimaryDitherType'] = [] 
        MyList['PrimaryDithers'] = []
        MyList['SubpixelPositions'] = []
        MyList['TargID'] = []
        MyFilterList = collections.OrderedDict()
        MyFilterList['Module'] = []
        MyFilterList['Subarray'] = [] 
        MyFilterList['Grism'] = []
        MyFilterList['PrimaryDitherType'] = [] 
        MyFilterList['PrimaryDithers'] = []
        MyFilterList['SubpixelPositions'] = []
        MyFilterList['Mode'] = []
        MyFilterList['ShortFilter'] = []
        MyFilterList['LongFilter'] = []
        MyFilterList['ReadoutPattern'] = []
        MyFilterList['Groups'] = []
        MyFilterList['Integrations'] = []
        MyTargList = []

        #read in the full file
        f = open(self.input_xml)
        fullfile = f.readlines()
        f.close()
        
        #now find the lines corresponding to the beginning of each
        #exposure list.
        wfss_start = np.array([]).astype(np.int)
        wfss_end = np.array([]).astype(np.int)
        explist_start = np.array([]).astype(np.int)
        directlist_start = np.array([]).astype(np.int)
        grismlist_start = np.array([]).astype(np.int)
        targlines = np.array([]).astype(np.int)

        #default values in case of missing data in APT file
        propid = '42424' 
        title = 'I need to find my towel'
        piname = 'D.N. Adams'
        pistart = 0
        piend = -1
        prop_category = 'GO'
        science_category = 'extrasolar towels'
        for linenum in range(len(fullfile)):
            if "<ncwfss:NircamWfss>" in fullfile[linenum]:
                wfss_start = np.append(wfss_start,linenum)
            if "</ncwfss:NircamWfss>" in fullfile[linenum]:
                wfss_end = np.append(wfss_end,linenum)
            if "<ncwfss:ExposureList>" in fullfile[linenum]:
                explist_start = np.append(explist_start,linenum)
            if "<ncwfss:DiExposure>" in fullfile[linenum]:
                directlist_start = np.append(directlist_start,linenum)
            if "<ncwfss:GrismExposure>" in fullfile[linenum]:
                grismlist_start = np.append(grismlist_start,linenum)
            #get the proposal ID number
            if "<ProposalID>" in fullfile[linenum]:
                propid = self.extract_value(fullfile[linenum])
            if "<Title>" in fullfile[linenum]:
                title = self.extract_value(fullfile[linenum])
            if "<PrincipalInvestigator>" in fullfile[linenum]:
                pistart = linenum
            if "</PrincipalInvestigator>" in fullfile[linenum]:
                piend = linenum
            if "<ProposalCategory>" in fullfile[linenum]:
                prop_category = self.extract_value(fullfile[linenum+1])[:-1]
            if "<ScientificCategory>" in fullfile[linenum]:
                science_category = self.extract_value(fullfile[linenum])  
            #if "<Target ID>" in fullfile[linenum]:
            #    targlines.append(linenum)

        if pistart > 0:
            for lnum in range(pistart,piend):
                if "<FirstName>" in fullfile[lnum]:
                    first = self.extract_value(fullfile[lnum])
                if "<LastName>" in fullfile[lnum]:
                    last = self.extract_value(fullfile[lnum])
            piname = first + ' ' + last
                
        #now, work on each wfss_start entry individually.
        #each one of these will have grism exposures, optional
        #direct exposure (singular), and out of field exposures (2).
        #We need to keep these all grouped together so that we end
        #with an exposure list that is in chronological order
        for wfssstart,wfssend in zip(wfss_start,wfss_end):
            for listele,addline in zip(MyList.keys(),range(1,7)):
                gt = fullfile[wfssstart+addline].find('>')
                lt = fullfile[wfssstart+addline].find('<',gt)
                MyList[listele] = fullfile[wfssstart+addline][gt+1:lt]
            #associate a target ID with each
            #prevtarg = np.where(targlines < wfssstart)
            #targ = self.extract_value(fullfile[prevtarg][-1]).split()
            #fulltarg = ''
            #for i in range(1,len(targ)):
            #    fulltarg = fulltarg + targ[i]
            ##MyTargList.append(fulltarg.strip())

            #now get info on the grism and optional direct images
            #that are only within the current wfss_start entry
    
            #first get all the grism exposure info
            gline = ((grismlist_start > wfssstart) &
                     (grismlist_start < wfssend))

            #make sure there is a WFSS entry in this exposure list
            if np.sum(gline) > 0:
                gentries = grismlist_start[gline]
        
                #loop over grism entry start lines
                for gindex in range(len(gentries)):
                    grismstart = gentries[gindex]
                    if gindex != 0:
                        prev_grism = gentries[gindex-1]
                    else:
                        prev_grism = wfssstart
                
                    MyFilterList['Mode'].append('WFSS')
                    for listele,addline in zip(MyFilterList.keys()[7:12],range(7,12)):
                        gt = fullfile[grismstart+addline-6].find('>')
                        lt = fullfile[grismstart+addline-6].find('<',gt)
                        MyFilterList[listele].append(fullfile[grismstart+addline-6][gt+1:lt])
                    for key in MyList:
                        MyFilterList[key].append(MyList[key])
                    MyTargList.append(fulltarg.strip())
                    #now get any direct exposure info that is tied to this 
                    #grism exposure
                    dline = ((directlist_start > prev_grism) &
                             (directlist_start < grismstart))
            
                    #make sure there is a direct image entry in this exposure list
                    if np.sum(dline) > 0:
                        for directstart in directlist_start[dline]:
                            MyFilterList['Mode'].append('Imaging')
                            MyTargList.append(fulltarg.strip())
                            for listele,addline in zip(MyFilterList.keys()[7:12],range(7,12)):
                                gt = fullfile[directstart+addline-6].find('>')
                                lt = fullfile[directstart+addline-6].find('<',gt)
                                MyFilterList[listele].append(fullfile[directstart+addline-6][gt+1:lt])
                            for key in MyList:
                                if key not in ['PrimaryDithers','SubpixelPositions']:
                                    MyFilterList[key].append(MyList[key])
                                else:
                                    MyFilterList[key].append('1')
            #now we need to add the two OUT OF FIELD expoures,
            #which are not in the xml file. They use the same
            #readout pattern/groups/ints as the direct image.
            #This is seen within APT itself, but is not in the
            #xml file. Since everything is the same as the direct
            #image taken immediately prior, we can just duplicate
            #the dictionary entries for the direct image.
            for key in MyFilterList:
                MyFilterList[key].extend((MyFilterList[key][-1],MyFilterList[key][-1]))

        #add proposal info
        MyFilterList['ProposalID'] = []
        if len(MyFilterList['Module']) > 0:
            MyFilterList['ProposalID'] = [np.int(propid)]*len(MyFilterList['Module'])
            MyFilterList['Title'] = [title]*len(MyFilterList['Module'])
            MyFilterList['PI_Name'] = [piname]*len(MyFilterList['Module'])
            MyFilterList['Proposal_category'] = [prop_category]*len(MyFilterList['Module'])
            MyFilterList['Science_category'] = [science_category]*len(MyFilterList['Module'])
            
        #now we need to deal with the pupil values.
        swpupillist = ['CLEAR'] * len(MyFilterList['Mode'])
        lwpupillist = ['CLEAR'] * len(MyFilterList['Mode'])
        for i in range(len(MyFilterList['Mode'])):
            if MyFilterList['Mode'][i] == 'WFSS':
                lwpupillist[i] = MyFilterList['Grism'][i]
            else:
                if '+' in MyFilterList['LongFilter'][i]:
                    p = MyFilterList['LongFilter'][i].find('+')
                    pup = MyFitlerList['LongFilter'][i][0:p]
                    f1 = MyFitlerList['LongFilter'][i][p+1:]
                    lwpupillist[i] = pup
                    MyFilterList['LongFilter'][i] = f1
            if '+' in MyFilterList['ShortFilter'][i]:
                p = MyFilterList['ShortFilter'][i].find('+')
                pup = MyFilterList['ShortFilter'][i][0:p]
                f1 = MyFilterList['ShortFilter'][i][p+1:]
                swpupillist[i] = pup
                MyFilterList['ShortFilter'][i] = f1
        MyFilterList['ShortPupil'] = swpupillist
        MyFilterList['LongPupil'] = lwpupillist

        #add in target names
        #print('list lengths: ',len(MyTargList),len(MyFilterList['ShortPupil'])))
        #stop
        #MyFilterList['TargID'] = MyTargList

        #add subpixeldithertype, to be consistent with imaging output
        MyFilterList['SubpixelDitherType'] = MyFilterList['SubpixelPositions']
        return MyFilterList


    def extract_value(self,line):
        #extract text from xml line
        gt = line.find('>')
        lt = line.find('<',gt)
        return line[gt+1:lt]
                                                  
        
    def read_imaging_xml(self,infile):
        #read APT xml file for imaging mode obs

        #first, a cheat. get proposal id by reading in file as ascii,
        #because I can't figure out the xml way to do it
                #read in the full file
        f = open(self.input_xml)
        fullfile = f.readlines()
        f.close()
        
        #get proposal information
        #default values in case of missing data in APT file
        propid = '42424' 
        title = 'Looking for my towel'
        piname = 'D.N. Adams'
        pistart = 0
        piend = -1
        prop_category = 'GO'
        science_category = 'extrasolar towels'

        for linenum in range(len(fullfile)):
            if "<ProposalID>" in fullfile[linenum]:
                propid = self.extract_value(fullfile[linenum])
            if "<Title>" in fullfile[linenum]:
                title = self.extract_value(fullfile[linenum])
            if "<PrincipalInvestigator>" in fullfile[linenum]:
                pistart = linenum
            if "</PrincipalInvestigator>" in fullfile[linenum]:
                piend = linenum
            if "<ProposalCategory>" in fullfile[linenum]:
                lt = fullfile[linenum+1].find('<')
                gt = fullfile[linenum+1].find('>')
                prop_category = fullfile[linenum+1][lt+1:gt][:-1]
            if "<ScientificCategory>" in fullfile[linenum]:
                science_category = self.extract_value(fullfile[linenum])             

        if pistart > 0:
            for lnum in range(pistart,piend):
                if "<FirstName>" in fullfile[lnum]:
                    first = self.extract_value(fullfile[lnum])
                if "<LastName>" in fullfile[lnum]:
                    last = self.extract_value(fullfile[lnum])
            piname = first + ' ' + last

                
        path = "//apt:Observation[apt:Instrument[contains(string(), '{}')]]/apt:Template/nci:NircamImaging".format('NIRCAM')
        
        targpath = "//apt:Observation"

        # READ XML file
        with open(infile) as f:
            tree = etree.parse(f)

        # APT makes extensive use of XML namespaces
        # (e.g. 'xmlns:nsmsasd="http://www.stsci.edu/JWST/APT/Template/NirspecMSAShortDetect"')
        # so we have to as well
        namespaces = tree.getroot().nsmap.copy()
        # There is no 'default' namespace for XPath (used below), but the lxml parser
        # does respect a default namespace, so we have to update its name from
        # 'None' to 'apt'
        namespaces['apt'] = namespaces[None]
        del namespaces[None]

        # Find your specific Observation
        results = tree.xpath(path, namespaces=namespaces)
        targresults = tree.xpath(targpath, namespaces=namespaces)

        #get target names. only one per Observation
        #for ExposureList in targresults:
        #    newresults = ExposureList.xpath("apt:Instrument[contains(string(), '{}')]]/apt:Template/nci:NircamImaging".format('NIRCAM'), namespaces=namespaces)

            #for item in mylist:
            #    entrylist = ExposureList.xpath("apt:TargetID",namespaces=namespaces)
            #    for entry in entrylist:
            #        mylist['TargetID'].append(entry.text)
            


        #set up variables for output
        MyList = {'Module': [], 'Subarray': [], 'PrimaryDitherType': [],
                  'PrimaryDithers': [], 'SubpixelDitherType': [],
                  'SubpixelPositions': []}
        MyFilterList = {'ShortFilter': [], 'LongFilter': [],
                        'ReadoutPattern': [], 'Groups': [], 'Integrations': []}
        finalList = {'Mode':[], 'Module': [], 'Subarray': [],
                     'PrimaryDitherType': [], 'PrimaryDithers': [],
                     'SubpixelDitherType': [], 'SubpixelPositions': [],
                     'ShortFilter': [], 'LongFilter': [], 'ReadoutPattern': [],
                     'Groups': [], 'Integrations': []}
        
        for ExposureList in results:
            #reset the lists in the dictionaries to be empty at
            #the beginning of each Template
            MyList = {'Module': [], 'Subarray': [], 'PrimaryDitherType': [],
                      'PrimaryDithers': [], 'SubpixelDitherType': [],
                      'SubpixelPositions': []}
            MyFilterList = {'ShortFilter': [], 'LongFilter': [],
                            'ReadoutPattern': [], 'Groups': [], 'Integrations': []}

            for item in MyList:
                entryList = ExposureList.xpath('nci:%s' % item,namespaces=namespaces)
                for entry in entryList:
                    MyList[item].append(entry.text)
            for item in MyFilterList:
                entryList = ExposureList.xpath('nci:Filters/nci:FilterConfig/nci:%s' % item,namespaces=namespaces)
                for entry in entryList:
                    MyFilterList[item].append(entry.text)
            
            #Add in a mode keyword so that we can easily separate
            #imaging from wfss entries. This will be useful once
            #these outputs are passed to the tool for making simulator
            #input files
            MyList['Mode'] = ['Imaging']*len(MyList['Module'])
                       
            #duplicate entries in the MyList dictionary so that the length
            #matches the myFilterList dictionary
            n_module = len(MyList['Module'])
            n_filter = len(MyFilterList['ShortFilter'])
            reps = n_filter - n_module
            for key in MyFilterList:
                finalList[key] = finalList[key] + MyFilterList[key]
            for key in MyList:
                finalList[key] = finalList[key] + MyList[key]*(reps+1)

        #check the filters. In the case where a pupil wheel-mounted filter
        #is used, the filter name will be "filter1+filter2". Separate into
        #filter and pupil entries
        shortplist = ['CLEAR']*len(finalList['ShortFilter'])
        longplist = ['CLEAR']*len(finalList['LongFilter'])
        for key in ['ShortFilter','LongFilter']:
            for i in range(len(finalList['ShortFilter'])):
                filt = finalList[key][i]
                if '+' in filt:
                    p = filt.find('+')
                    pupil = filt[0:p]
                    f1 = filt[p+1:]
                    if key == 'ShortFilter':
                        finalList[key][i] = f1                
                        shortplist[i] = pupil
                    if key == 'LongFilter':
                        finalList[key][i] = f1
                        longplist[i] = pupil
        finalList['ShortPupil'] = shortplist
        finalList['LongPupil'] = longplist

        #for consistency with the output from the WFSS reader
        finalList['Grism'] = ['N/A'] * len(finalList['Mode'])

        #add proposal info lines
        finalList['ProposalID'] = [np.int(propid)]*len(finalList['Module'])
        finalList['Title'] = [title]*len(finalList['Module'])
        finalList['PI_Name'] =  [piname]*len(finalList['Module'])
        finalList['Proposal_category'] = [prop_category]*len(finalList['Module'])
        finalList['Science_category'] = [science_category]*len(finalList['Module'])

        return finalList


    def expand_for_dithers(self,dict):
        #Expand a given dictionary to create one entry
        #for each dither
        #define the dictionary to hold the expanded entries

        #in here we should also reset the primary and subpixel dither
        #numbers to 1, to avoid confusion.
        
        expanded = {}
        for key in dict:
            expanded[key] = []
    
        #loop over entries in dict and duplicate by the
        #number of dither positions  
        keys = np.array(dict.keys())
        for i in range(len(dict['PrimaryDithers'])):
            entry = np.array([item[i] for item in dict.values()])
            subpix = entry[keys == 'SubpixelPositions']
            #in WFSS, SubpixelPositions will be either '4-Point' or '9-Point'
            primary = entry[keys == 'PrimaryDithers']
            reps = np.int(subpix[0][0]) * np.int(primary[0])
            for key in keys:
                for j in range(reps):
                    expanded[key].append(dict[key][i])
        return expanded

    
    def base36encode(self,integer):
        chars, encoded = '0123456789abcdefghijklmnopqrstuvwxyz', ''

        while integer > 0:
            integer, remainder = divmod(integer, 36)
            encoded = chars[remainder] + encoded

        return encoded.zfill(2)


    def get_pointing_info(self,file,propid):
        #read in information from APT's pointing file
        tar = []
        tile = []
        exp = []
        dith = []
        aperture = []
        targ1 = []
        targ2 = []
        ra = []
        dec = []
        basex = []
        basey = []
        dithx = []
        dithy = []
        v2 = []
        v3 = []
        idlx = []
        idly = []
        level = []
        type_str = []
        expar = []
        dkpar = []
        ddist = []
        observation_number = []
        visit_number = []
        visit_id = []
        visit_grp = []
        activity_id = []
        observation_label = []
        observation_id = []
        seq_id = []

        act_counter = 1
        with open(file) as f:
            for line in f:
                if len(line) > 1:
                    elements = line.split()
                    
                    #look for lines that give visit/observation numbers
                    if line[0:2] == '* ':
                        paren = line.rfind('(')
                        if paren == -1:
                            obslabel = line[2:]
                        else:
                            obslabel = line[2:paren-1]
                    if line[0:2] == '**':
                        v = elements[2]
                        obsnum,visitnum = v.split(':')
                        obsnum = str(obsnum).zfill(3)
                        visitnum = str(visitnum).zfill(3)
                        
                    try:
                        #skip the line at the beginning of each
                        #visit that gives info on the target,
                        #but is not actually an observation
                        #These lines have 'Exp' values of 0,
                        #while observations have a value of 1
                        #(that I've seen so far)
                        if np.int(elements[1]) > 0:
                            act = self.base36encode(act_counter)
                            activity_id.append(act)
                            observation_label.append(obslabel)
                            observation_number.append(obsnum)
                            visit_number.append(visitnum)
                            vid = str(propid)+visitnum+obsnum
                            visit_id.append(vid)
                            vgrp = '01'
                            visit_grp.append(vgrp)
                            seq = '1'
                            seq_id.append(seq)
                            tar.append(np.int(elements[0]))
                            tile.append(np.int(elements[1]))
                            exnum = str(elements[2]).zfill(5)
                            exp.append(exnum)
                            dith.append(np.int(elements[3]))
                            aperture.append(elements[4])
                            targ1.append(np.int(elements[5]))
                            targ2.append(elements[6])
                            ra.append(elements[7])
                            dec.append(elements[8])
                            basex.append(elements[9])
                            basey.append(elements[10])
                            dithx.append(np.float(elements[11]))
                            dithy.append(np.float(elements[12]))
                            v2.append(np.float(elements[13]))
                            v3.append(np.float(elements[14]))
                            idlx.append(np.float(elements[15]))
                            idly.append(np.float(elements[16]))
                            level.append(elements[17])
                            type_str.append(elements[18])
                            expar.append(np.int(elements[19]))
                            dkpar.append(np.int(elements[20]))
                            ddist.append(np.float(elements[21]))
                            observation_id.append('V'+vid+'P00000000'+vgrp+seq+act)
                            act_counter += 1
                    except:
                        pass

        pointing = {'exposure':exp, 'dither':dith, 'aperture':aperture,
                    'targ1':targ1, 'targ2':targ2, 'ra':ra, 'dec':dec,
                    'basex':basex, 'basey':basey, 'dithx':dithx,
                    'dithy':dithy, 'v2':v2, 'v3':v3, 'idlx':idlx,
                    'idly':idly, 'obs_label':observation_label,
                    'obs_num':observation_number,'visit_num':visit_number,
                    'act_id':activity_id,'visit_id':visit_id,'visit_group':visit_grp,
                    'sequence_id':seq_id,'observation_id':observation_id}
        return pointing


    def combine_dicts(self,dict1,dict2):
        #Now combine the dictionaries from the xml file and the pointing file
        combined = dict1.copy()
        combined.update(dict2)
        return combined


    def expand_for_detectors(self,dict):
        #Expand dictionary to have one line per detector, rather than the
        #one line per module that is in the input
        finaltab = {}
        for key in dict:
            finaltab[key] = []
        finaltab['detector'] = []
        for i in range(len(dict['PrimaryDithers'])):
            module = dict['Module'][i]
            if module == 'ALL':
                detectors = ['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5']
            elif module == 'A':
                detectors = ['A1','A2','A3','A4','A5']
            elif module == 'B':
                detectors = ['B1','B2','B3','B4','B5']

            for key in dict:
                finaltab[key].extend(([dict[key][i]]*len(detectors)))
            finaltab['detector'].extend(detectors)
        return finaltab


    def ra_dec_update(self):
        #given the v2,v3 values in each entry, calculate RA,Dec
        #For now, assume a roll angle of zero. We would only
        #know roll angle once the program is scheduled
        print('Performing RA,Dec updates, assuming a roll angle of zero!!!')

        #read in siaf
        distortionTable = ascii.read(self.siaf,header_start=1)
            
        aperture_ra = []
        aperture_dec = []
        for i in range(len(self.exposure_tab['Module'])):

            #first find detector
            #need ra,dec and v2,v3 pairs from entry
            #to calculate ra,dec at each detector's reference location
            detector = 'NRC' + self.exposure_tab['detector'][i]
            sub = self.exposure_tab['Subarray'][i]
            aperture = detector + '_' + sub
            pointing_ra = np.float(self.exposure_tab['ra'][i])
            pointing_dec = np.float(self.exposure_tab['dec'][i])
            pointing_v2 = np.float(self.exposure_tab['v2'][i])
            pointing_v3 = np.float(self.exposure_tab['v3'][i])

            #assume for now a telescope roll angle of 0.
            local_roll = set_telescope_pointing.compute_local_roll(0.,pointing_ra,
                                                                   pointing_dec,
                                                                   pointing_v2,
                                                                   pointing_v3)
            #create attitude_matrix    
            attitude_matrix = rotations.attitude(pointing_v2,pointing_v3,
                                                 pointing_ra,pointing_dec,local_roll)
            
            #find v2,v3 of the reference location for the detector
            aperture_v2,aperture_v3 = self.ref_location(distortionTable,aperture)
            
            #calculate RA, Dec of reference location for the detector
            ra,dec = rotations.pointing(attitude_matrix,aperture_v2,aperture_v3)
            aperture_ra.append(ra)
            aperture_dec.append(dec)
        self.exposure_tab['ra_ref'] = aperture_ra
        self.exposure_tab['dec_ref'] = aperture_dec

        
    def ref_location(self,siaf,det):
        #find v2,v3 of detector reference location
        match = siaf['AperName'] == det
        if np.any(match) == False:
            print("Aperture name {} not found in input CSV file.".
                  format(aperture))
            sys.exit()
        v2 = siaf[match]['V2Ref']
        v3 = siaf[match]['V3Ref']
        return v2,v3

        
    def create_input_table(self):
        #read in xml file. Try reading as imaging mode first.
        tab_im = self.read_imaging_xml(self.input_xml)

        #try reading as WFSS
        #if len(tab[tab.keys()[0]]) == 0:
        tab_wfss = self.read_wfss_xml(self.input_xml)

        if len(tab_im['Mode']) == 0:
            if len(tab_wfss['Mode']) > 0:
                tab = tab_wfss
            else:
                print('WARNING: neither imaging nor WFSS observations found! Quitting.')
                sys.exit()
        else:
            if len(tab_wfss['Mode']) > 0:
                tab = {}
                for key in tab_im:
                    tab[key] = tab_im[key] + tab_wfss[key]
            else:
                tab = tab_im

        #expand the dictionary for multiple dithers. Expand such that there
        #is one entry in each list for each exposure, rather than one entry
        #for each set of dithers
        xmltab = self.expand_for_dithers(tab)

        #read in the pointing file and produce dictionary
        pointing_tab = self.get_pointing_info(self.pointing_file,xmltab['ProposalID'][0])
        
        #combine the dictionaries
        obstab = self.combine_dicts(xmltab,pointing_tab)

        #expand for detectors. Create one entry in each list for each
        #detector, rather than a single entry for 'ALL' or 'BSALL'
        self.exposure_tab = self.expand_for_detectors(obstab)

        #calculate the correct V2,V3 and RA,Dec for each exposure/detector
        self.ra_dec_update()

        #output to a csv file. 
        if self.output_csv is None:
            indir,infile = os.path.split(self.input_xml)
            self.output_csv = 'Observation_table_for_'+infile+'.csv'
        ascii.write(Table(self.exposure_tab), self.output_csv, format='csv', overwrite=True)
        print('Final csv exposure list written to {}'.format(self.output_csv))
        

    def dict_lengths(self,dict):
        minlength = 99999999
        maxlength = 0
        for key in dict:
            ll = len(dict[key])
            if ll > maxlength:
                maxlength = ll
            if ll < minlength:
                minlength = ll
        return minlength,maxlength
    

    def add_options(self,parser=None,usage=None):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,description='Simulate JWST ramp')
        parser.add_argument("input_xml",help='XML file from APT describing the observations.')
        parser.add_argument("pointing_file",help='Pointing file from APT describing observations.')
        parser.add_argument("siaf",help='Name of CSV version of SIAF')
        parser.add_argument("--output_csv",help="Name of output CSV file containing list of observations.",default=None)
        return parser
    

if __name__ == '__main__':

    usagestring = 'USAGE: apt_inputs.py NIRCam_obs.xml NIRCam_obs.pointing SIAF_March2017.csv'

    input = AptInput()
    parser = input.add_options(usage = usagestring)
    args = parser.parse_args(namespace=input)
    input.create_input_table()
