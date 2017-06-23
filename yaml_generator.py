#! /usr/bin/env python

'''
Function to produce yaml files that can be used as input for 
the ramp simulator

inputs: 
read in from a file? 
passed as lists (i.e. read them from APT file, then pass into this code)?


inputs (each can be a single string/float or a list for multiple exposures?):

instrument - APT
mode - APT
readpattern - APT
ngroup - APT
nint - APT
array_name - APT+help
filter - APT
pupil - APT
pointsource - USER
galaxyListFile - USER
extended - usER
movingTargetList - USER
movingTargetSersic? - USER
movingTargetExtended? - USER
movingTargetToTrack? - USER
bkgdrate - USER
ra  - APT
dec - APT
rotation - APT?
file (output) - USER
grism_source_image - USER
grism_input_only - USER

program_number - APT
observation_number - APT
visit_number - APT
visit_group - APT
obs_id - APT
visit_id - APT
sequence_id - APT
activity_id - APT
exposure_number - APT

seq_id changes based on coordinatedparallel entry 1 for prime 2-5 for parallel

template - observation template  "NIRCam Imaging" for all? WFSS is not listed as an entry in the keyword dictionary
https://mast.stsci.edu/portal/Mashup/Clients/jwkeywords/index.html

'''



import sys,os
import argparse
import numpy as np
from glob import glob
from astropy.time import Time,TimeDelta
from astropy.table import Table
from astropy.io import ascii
sys.path.append(os.getcwd())
import apt_inputs
    

class SimInput:
    def __init__(self):
        self.info = {}
        self.input_xml = None
        self.pointing_file = None
        self.siaf = None
        self.reffile_setup()
        self.output_dir = './'

        
        
    def create_inputs(self):
        #this is the main function, called when inputs are from command line
        
        #get input from APT (optional)
        #parse APT info: NRC_SBALL broken out into individual detectors
    
        #get user-specified input - from argparse or set as self.whatever = myval
        #combine user-specified info and APT info into single dictionary

        #make the inputs a list of dictionaries, and then pass them one at a time
        #to the yaml-writing function
        #Or should it be a dictionary where each element is a list, and then
        #one entry at a time is pulled out from each key?

        if ((self.input_xml is not None) & (self.pointing_file is not None) & (self.siaf is not None)):
            print('Using {}, {}, and {} to generate observation table.'.
                  format(self.input_xml,self.pointing_file,self.siaf))
            indir,infile = os.path.split(self.input_xml)
            final_file = 'Observation_table_for_'+infile+'_with_yaml_parameters.csv'
            apt = apt_inputs.AptInput()
            apt.input_xml = self.input_xml
            apt.pointing_file = self.pointing_file
            apt.siaf = self.siaf
            apt.create_input_table()
            self.info = apt.exposure_tab

            #Add start time info to each element
            self.make_start_times()
                
            #Add a list of output yaml names to the dictionary
            self.make_output_names()

            #Add source catalogs
            self.add_catalogs()
            
        elif self.table_file is not None:
            print('Reading table file: {}'.format(self.table_file))
            info = ascii.read(self.table_file)
            self.info = self.table_to_dict(info)
            final_file = self.table_file+'_with_yaml_parameters.csv'

        else:
            print("WARNING. You must include either an ascii table file of observations.")
            print("or xml and pointing files from APT plus an ascii siaf table.")
            sys.exit()
     
        #For each element in the lists, use the detector name to
        #find the appropriate reference files. Create lists, and add
        #these lists to the dictionary
        darks = []
        superbias = []
        linearity = []
        saturation = []
        gain = []
        astrometric = []
        ipc = []
        pam = []
        for det in self.info['detector']:
            darks.append(self.get_dark(det))
            superbias.append(self.get_reffile(self.superbias_list,det))
            linearity.append(self.get_reffile(self.linearity_list,det))
            saturation.append(self.get_reffile(self.saturation_list,det))
            gain.append(self.get_reffile(self.gain_list,det))
            astrometric.append(self.get_reffile(self.astrometric_list,det))
            ipc.append(self.get_reffile(self.ipc_list,det))
            pam.append(self.get_reffile(self.pam_list,det))
        self.info['dark'] = darks
        self.info['superbias'] = superbias
        self.info['linearity'] = linearity
        self.info['saturation'] = saturation
        self.info['gain'] = gain
        self.info['astrometric'] = astrometric
        self.info['ipc'] = ipc
        self.info['pixelAreaMap'] = pam

        #add background rate to the table
        self.info['bkgdrate'] = np.array([self.bkgdrate]*len(self.info['Mode']))

        #grism entries
        grism_source_image = ['False'] * len(self.info['Mode'])
        grism_input_only = ['False'] * len(self.info['Mode'])
        for i in range(len(self.info['Mode'])):
            if self.info['Mode'][i] == 'WFSS':
                grism_source_image[i] = 'True'
                grism_input_only[i] = 'True'
        self.info['grism_source_image'] = grism_source_image
        self.info['grism_input_only'] = grism_input_only
        
        #assume a rotation angle of 0 for now
        self.info['rotation'] = [0.] * len(self.info['Mode']) 
        
        #level-3 associated keywords that are not present in APT file.
        #not quite sure how to populate these
        self.info['visit_group'] = ['01'] * len(self.info['Mode'])
        #self.info['sequence_id'] = ['1'] * len(self.info['Mode'])
        seq = []
        for par in self.info['CoordinatedParallel']:
            if par.lower() == 'true':
                seq.append('2')
            if par.lower() == 'false':
                seq.append('1')
        self.info['sequence_id'] = seq
        self.info['obs_template'] = ['NIRCam Imaging'] * len(self.info['Mode'])
        
        #write out the updated table, including yaml filenames
        #start times, and reference files
        print('Updated observation table file saved to {}'.format(final_file))
        ascii.write(Table(self.info),final_file,format='csv',overwrite=True)
        
        #Now go through the lists one element at a time
        #and create a yaml file for each.
        for i in range(len(self.info['detector'])):
            file_dict = {}
            for key in self.info:
                file_dict[key] = self.info[key][i]
            
            self.write_yaml(file_dict)


    def add_catalogs(self):
        #Add list(s) of source catalogs to table
        self.info['point_source'] = [None] * len(self.info['Module'])
        self.info['galaxyListFile'] = [None] * len(self.info['Module'])
        self.info['extended'] = [None] * len(self.info['Module'])
        self.info['convolveExtended'] = [False] * len(self.info['Module'])
        self.info['movingTarg'] = [None] * len(self.info['Module'])
        self.info['movingTargSersic'] = [None] * len(self.info['Module'])
        self.info['movingTargExtended'] = [None] * len(self.info['Module'])
        self.info['movingTargToTrack'] = [None] * len(self.info['Module'])
        
        for i in range(len(self.info['ShortFilter'])):
            if np.int(self.info['detector'][i][-1]) < 5:
                filtkey = 'ShortFilter'
                pupilkey = 'ShortPupil'
            else:
                filtkey = 'LongFilter'
                pupilkey = 'LongPupil'
            filt = self.info[filtkey][i]
            pup = self.info[pupilkey][i]

            if self.point_source[0] is not None:
                #In here, we assume the user provided a catalog to go with each filter
                #so now we need to find the filter for each entry and generate a list that makes sense
                self.info['point_source'][i] = self.catalog_match(filt,pup,self.point_source,'point source')
            if self.galaxyListFile[0] is not None:
                self.info['galaxyListFile'][i] = self.catalog_match(filt,pup,self.galaxyListFile,'galaxy')
            if self.extended[0] is not None:
                self.info['extended'][i] = self.catalog_match(filt,pup,self.extended,'extended')
            if self.movingTarg[0] is not None:
                self.info['movingTarg'][i] = self.catalog_match(filt,pup,self.movingTarg,'moving point source target')
            if self.movingTargSersic[0] is not None:
                self.info['movingTargSersic'][i] = self.catalog_match(filt,pup,self.movingTargSersic,'moving sersic target')
            if self.movingTargExtended[0] is not None:
                self.info['movingTargExtended'][i] = self.catalog_match(filt,pup,self.movingTargExtended,'moving extended target')
            if self.movingTargToTrack[0] is not None:
                self.info['movingTargToTrack'][i] = self.catalog_match(filt,pup,self.movingTargToTrack,'non-sidereal moving target')

        if self.convolveExtended == True:      
            self.info['convolveExtended'] = [True] * len(self.info['Module'])
            

    def catalog_match(self,filter,pupil,catalog_list,cattype):
        #given a filter and pupil value, along with a list of input
        #catalogs, find the catalogs that match the pupil or filter
        if pupil[0].upper() == 'F':
            match = [s for s in catalog_list if pupil.lower() in s.lower()]
            if len(match) == 0:
                self.no_catalog_match(pupil,cattype)
                return None
            elif len(match) > 1:
                self.multiple_catalog_match(pupil,cattype,match)
            return match[0]
        else:
            match = [s for s in catalog_list if filter.lower() in s.lower()]
            if len(match) == 0:
                self.no_catalog_match(filter,cattype)
                return None
            elif len(match) > 1:
                self.multiple_catalog_match(filter,cattype,match)
            return match[0]

    
    def no_catalog_match(self,filter,cattype):
        #tell user if no catalog match was found
        print("WARNING: unable to find filter ({}) name".format(filter))
        print("in any of the given {} inputs".format(cattype))
        print("Using the first input for now. Make sure input catalog names have")
        print("the appropriate filter name in the filename to get matching to work.")


    def multiple_catalog_match(self,filter,cattype,matchlist):
        #tell the user if more than one catalog matches the filter/pupil
        print("WARNING: multiple {} catalogs matched! Using the first.".format(cattype))
        print("Observation filter: {}".format(filter))
        print("Matched point source catalogs: {}".format(matchlist))

            
    def table_to_dict(self,tab):
        #convert the ascii table of observations to the
        #needed dictionary
        dict = {}
        for colname in tab.colnames:
            dict[colname] = tab[colname].data
        return dict
    

    def make_start_times(self):
        #create exposure start times for each entry
        #the time and date to start with are optional inputs
        date_obs = []
        time_obs = []
        expstart = []
        nframe = []
        nskip = []
        namp = []

        b = self.obsdate+'T'+self.obstime
        base = Time(b)

        #pick some arbirary overhead values
        act_overhead = 40 #seconds. (filter change)
        visit_overhead = 600 #seconds. (slew)

        #get visit,activity_id info for first exposure
        actid = self.info['act_id'][0]
        visit = self.info['visit_num'][0]
    
        #read in file containing subarray definitions
        subarray_def = self.get_subarray_defs()

        #now read in readpattern definitions
        readpatt_def = self.get_readpattern_defs()
        
        for i in range(len(self.info['Module'])):
            #Get dither/visit 
            #Files with the same activity_id should have the same start time
            #Overhead after a visit break should be large, smaller between
            #exposures within a visit
            next_actid = self.info['act_id'][i]
            next_visit = self.info['visit_num'][i]

            #get the values of nframes and nskip 
            readpatt = self.info['ReadoutPattern'][i]
            
            #Find the readpattern of the file
            readpatt = self.info['ReadoutPattern'][i]
            groups = np.int(self.info['Groups'][i])
            integrations = np.int(self.info['Integrations'][i])

            match2 = readpatt == readpatt_def['name']
            if np.sum(match2) == 0:
                print("WARNING!! Readout pattern {} not found in definition file.".format(readpatt))
                sys.exit()
                
            #Now get nframe and nskip so we know how many frames in a group
            fpg = np.int(readpatt_def['nframe'][match2][0])
            spg = np.int(readpatt_def['nskip'][match2][0])
            nframe.append(fpg)
            nskip.append(spg)

            #need to find number of amps used
            sub = self.info['Subarray'][i]
            det = 'NRC' + self.info['detector'][i]
            sub = det + '_' + sub

            match = sub == subarray_def['AperName']
            if np.sum(match) == 0:
                print("WARNING!! Subarray {} not found in definition file.".format(sub))
                sys.exit()
            amp = subarray_def['num_amps'][match][0]
            namp.append(amp)    

            if next_actid == actid:
                #in this case, the start time should remain the same
                date_obs.append(self.obsdate)
                time_obs.append(self.obstime)
                expstart.append(base.mjd)
                continue
            elif next_visit > visit:
                #visit break. Larger overhead
                overhead = visit_overhead
            elif ((next_actid > actid) & (next_visit == visit)):
                #same visit. Smaller overhead
                overhead = act_overhead
            else:
                #should never get in here
                print("Error. Fix me")
                
            #For cases where the base time needs to change
            #continue down here
            xs = subarray_def['xstart'][match][0]
            xe = subarray_def['xend'][match][0]
            ys = subarray_def['ystart'][match][0]
            ye = subarray_def['yend'][match][0]
            xd = xe - xs + 1
            yd = ye - ys + 1
            frametime = self.calcFrameTime(xd,yd,amp)
            
            #Estimate total exposure time
            exptime = ((fpg+spg) * groups + fpg) * integrations * frametime

            #Delta should include the exposure time, plus overhead
            delta = TimeDelta(exptime+overhead,format='sec')
            base += delta
            self.obsdate,self.obstime = base.iso.split()
            
            #Add updated dates and times to the list
            date_obs.append(self.obsdate)
            time_obs.append(self.obstime)
            expstart.append(base.mjd)
            
        self.info['date_obs'] = date_obs
        self.info['time_obs'] = time_obs
        #self.info['expstart'] = expstart
        self.info['nframe'] = nframe
        self.info['nskip'] = nskip
        self.info['namp'] = namp


    def get_readpattern_defs(self):
        #read in the readpattern definition file
        tab = ascii.read(self.readpatt_def_file)
        return tab
    
        
    def get_subarray_defs(self):
        #read in subarray definition file and return table
        sub = ascii.read(self.subarray_def_file)
        return sub
        

    def calcFrameTime(self,xd,yd,namp):
        #calculate the exposure time of a single frame of the proposed output ramp
        #based on the size of the croped dark current integration
        return (xd/namp + 12.) * (yd+1) * 10.00 * 1.e-6

        
    def make_output_names(self):
        #create output yaml file names to go with all of the
        #entries in the dictionary
        onames = []
        fnames = []
        for i in range(len(self.info['Module'])):
            act = str(self.info['act_id'][i]).zfill(2)
            det = self.info['detector'][i]
            mode = self.info['Mode'][i]
            dither = str(self.info['dither'][i]).zfill(2)
            onames.append(os.path.abspath(os.path.join(self.output_dir,'Act{}_{}_{}_Dither{}.yaml'.format(act,det,mode,dither))))
            fnames.append(os.path.abspath(os.path.join(self.output_dir,'Act{}_{}_{}_Dither{}_uncal.fits'.format(act,det,mode,dither))))
        self.info['yamlfile'] = onames
        self.info['outputfits'] = fnames
        
        
    def get_dark(self,detector):
        #return the name of a dark current file to use as input
        #based on the detector being used
        files = self.dark_list[detector]
        rand_index = np.random.randint(0,len(files)-1)
        return files[rand_index]


    def get_reffile(self,refs,detector):
        #return the appropriate reference file for detector
        #and given reference file dictionary. Assume that
        #refs is a dictionary in the form of:
        #{'A1':'filenamea1.fits','A2':'filenamea2.fits'...}
        for key in refs:
            if detector in key:
                return refs[key]
        print("WARNING: no file found for detector {} in {}"
              .format(detector,refs))

    
    def write_yaml(self,input):
        #create yaml file for a single exposure/detector
        #input is a dictionary containing all needed info

        #select the right filter
        if np.int(input['detector'][-1]) < 5:
            filtkey = 'ShortFilter'
            pupilkey = 'ShortPupil'
        else:
            filtkey = 'LongFilter'
            pupilkey = 'LongPupil'

        with open(input['yamlfile'],'w') as f:
            f.write('Inst:\n')
            f.write('  instrument: {}          #Instrument name\n'.format('NIRCam'))
            f.write('  mode: {}                #Observation mode (e.g. imaging, WFSS, moving_target)\n'.format(input['Mode']))
            f.write('  nresetlines: 512                        #eventially use dictionary w/in code to look this up\n')
            f.write('\n')
            f.write('Readout:\n')
            f.write('  readpatt: {}        #Readout pattern (RAPID, BRIGHT2, etc) overrides nframe,nskip unless it is not recognized\n'.format(input['ReadoutPattern']))
            f.write('  nframe: {}        #Number of frames per group\n'.format(input['nframe']))
            f.write('  nskip: 0         #Number of skipped frames between groups\n'.format(input['nskip']))
            f.write('  ngroup: {}              #Number of groups in integration\n'.format(input['Groups']))
            f.write('  nint: {}          #Number of integrations per exposure\n'.format(input['Integrations']))
            f.write('  namp: {}         #Number of amplifiers used in readout (4 for full frame, 1 for subarray)\n'.format(input['namp']))
            apunder = input['aperture'].find('_')
            full_ap = 'NRC' + input['detector'] + '_' + input['aperture'][apunder+1:]
            f.write('  array_name: {}    #Name of array (FULL, SUB160, SUB64P, etc) overrides subarray_bounds below\n'.format(full_ap))
            f.write('  subarray_bounds: 0, 0, 159, 159          #Coords of subarray corners. (xstart, ystart, xend, yend) Over-ridden by array_name above. Currently not used. Could be used if output saved in raw format\n')
            f.write('  filter: {}       #Filter of simulated data (F090W, F322W2, etc)\n'.format(input[filtkey]))
            f.write('  pupil: {}        #Pupil element for simulated data (CLEAR, GRISMC, etc)\n'.format(input[pupilkey]))
            f.write('\n')
            f.write('Reffiles:                                 #Set to None or leave blank if you wish to skip that step\n')
            f.write('  dark: {}   #Dark current integration used as the base\n'.format(input['dark']))
            f.write('  hotpixmask: None                        #Hot pixel mask to go with the dark integration. If none, the script will find hot pixels. Fits file. Ones are hot. Zeros not.\n')
            f.write('  superbias: {}     #Superbias file. Set to None or leave blank if not using\n'.format(input['superbias']))
            f.write('  subarray_defs: NIRCam_subarray_definitions.list                #File that contains a list of all possible subarray names and coordinates\n')
            f.write('  readpattdefs: nircam_read_pattern_definitions.list           #File that contains a list of all possible readout pattern names and associated NFRAME/NSKIP values\n')
            f.write('  linearity: {}    #linearity correction coefficients\n'.format(input['linearity']))
            f.write('  saturation: {}    #well depth reference files\n'.format(input['saturation']))
            f.write('  gain: {} #Gain map\n'.format(input['gain']))
            f.write('  phot: nircam_mag15_countrates.list  #File with list of all filters and associated quantum yield values and countrates for mag 15 star\n')
            f.write('  pixelflat: None \n')
            f.write('  illumflat: None                               #Illumination flat field file\n')
            f.write('  astrometric: {}  #Astrometric distortion file (asdf)\n'.format(input['astrometric']))
            f.write('  distortion_coeffs: NIRCam_SIAF_2016-09-29.csv         #CSV file containing distortion coefficients\n')
            f.write('  ipc: {} #File containing IPC kernel to apply\n'.format(input['ipc']))
            f.write('  invertIPC: True       #Invert the IPC kernel before the convolution. True or False. Use True if the kernel is designed for the removal of IPC effects, like the JWST reference files are.\n')
            f.write('  crosstalk: xtalk20150303g0.errorcut.txt              #File containing crosstalk coefficients\n')
            f.write('  occult: None                                    #Occulting spots correction image\n')
            f.write('  filtpupilcombo: nircam_filter_pupil_pairings.list       #File that lists the filter wheel element / pupil wheel element combinations. Used only in writing output file\n')
            f.write('  pixelAreaMap: {}      #Pixel area map for the detector. Used to introduce distortion into the output ramp.\n'.format(input['pixelAreaMap']))
            f.write('  flux_cal: /ifs/jwst/wit/witserv/data4/nrc/hilbert/simulated_data/NIRCam_zeropoints.list  #File that lists flux conversion factor and pivot wavelength for each filter. Only used when making direct image outputs to be fed into the grism disperser code.')
            f.write('\n')
            f.write('nonlin:\n')
            f.write('  limit: 60000.0                           #Upper singal limit to which nonlinearity is applied (ADU)\n')
            f.write('  accuracy: 0.000001                        #Non-linearity accuracy threshold\n')
            f.write('  maxiter: 10                              #Maximum number of iterations to use when applying non-linearity\n')
            f.write('  robberto:  False                         #Use Massimo Robberto type non-linearity coefficients\n')
            f.write('\n')
            f.write('\n')
            f.write('cosmicRay:\n')
            f.write('  path: /ifs/jwst/wit/witserv/data4/nrc/hilbert/simulated_data/cosmic_ray_library/               #Path to CR library\n')
            f.write('  library: SUNMIN    #Type of cosmic rayenvironment (SUNMAX, SUNMIN, FLARE)\n')
            f.write('  scale: 1.5     #Cosmic ray scaling factor\n')
            f.write('  suffix: IPC_NIRCam_{}    #Suffix of library file names\n'.format(input['detector']))
            f.write('  seed: {}                           #Seed for random number generator\n'.format(np.random.randint(1,2**32-2)))
            f.write('\n')
            f.write('simSignals:\n')
            f.write('  pointsource: {}   #File containing a list of point sources to add (x,y locations and magnitudes)\n'.format(input['point_source']))
            f.write('  psfpath: /ifs/jwst/wit/witserv/data4/nrc/hilbert/simulated_data/psf_files/        #Path to PSF library\n')
            f.write('  psfbasename: nircam                        #Basename of the files in the psf library\n')
            f.write('  psfpixfrac: 0.1                           #Fraction of a pixel between entries in PSF library (e.g. 0.1 = files for PSF centered at 0.1 pixel intervals within pixel)\n')
            f.write('  psfwfe: 123                               #PSF WFE value (0,115,123,132,136,150,155)\n')
            f.write('  psfwfegroup: 0                             #WFE realization group (0 to 9)\n')
            f.write('  galaxyListFile: {}    #File containing a list of positions/ellipticities/magnitudes of galaxies to simulate\n'.format(input['galaxyListFile']))
            f.write('  extended: {}          #Extended emission count rate image file name\n'.format(input['extended']))
            f.write('  extendedscale: 1.0                          #Scaling factor for extended emission image\n')
            f.write('  extendedCenter: 1024,1024                   #x,y pixel location at which to place the extended image if it is smaller than the output array size\n')
            f.write('  PSFConvolveExtended: True #Convolve the extended image with the PSF before adding to the output image (True or False)\n')
            f.write('  movingTargetList: {}          #Name of file containing a list of point source moving targets (e.g. KBOs, asteroids) to add.\n'.format(input['movingTarg']))
            f.write('  movingTargetSersic: {}  #ascii file containing a list of 2D sersic profiles to have moving through the field\n'.format(input['movingTargSersic']))
            f.write('  movingTargetExtended: {}      #ascii file containing a list of stamp images to add as moving targets (planets, moons, etc)\n'.format(input['movingTargExtended']))
            f.write('  movingTargetConvolveExtended: True       #convolve the extended moving targets with PSF before adding.\n')
            f.write('  movingTargetToTrack: {} #File containing a single moving target which JWST will track during observation (e.g. a planet, moon, KBO, asteroid)	This file will only be used if mode is set to "moving_target" \n'.format(input['movingTargToTrack']))
            
            f.write('  zodiacal:  None                          #Zodiacal light count rate image file \n')
            f.write('  zodiscale:  1.0                            #Zodi scaling factor\n')
            f.write('  scattered:  None                          #Scattered light count rate image file\n')
            f.write('  scatteredscale: 1.0                        #Scattered light scaling factor\n')
            f.write('  bkgdrate: {}                         #Constant background count rate (electrons/sec/pixel)\n'.format(input['bkgdrate']))
            f.write('  poissonseed: {}                  #Random number generator seed for Poisson simulation)\n'.format(np.random.randint(1,2**32-2)))
            f.write('  photonyield: True                         #Apply photon yield in simulation\n')
            f.write('  pymethod: True                            #Use double Poisson simulation for photon yield\n')
            f.write('\n')
            f.write('Telescope:\n')
            f.write('  ra: {}                      #RA of simulated pointing\n'.format(input['ra_ref']))
            f.write('  dec: {}                    #Dec of simulated pointing\n'.format(input['dec_ref']))
            f.write('  rotation: {}                    #y axis rotation (degrees E of N)\n'.format(input['rotation']))
            f.write('\n')
            f.write('newRamp:\n')
            f.write('  combine_method: PROPER\n')
            f.write('  proper_combine: BOTH\n')
            f.write('  proper_signal_limit: 5000\n')
            f.write('  linearized_darkfile: None\n')
            f.write('  dq_configfile: dq_init.cfg\n')
            f.write('  sat_configfile: saturation.cfg\n')
            f.write('  superbias_configfile: superbias.cfg\n')
            f.write('  refpix_configfile: refpix.cfg\n')
            f.write('  linear_configfile: linearity.cfg\n')
            f.write('\n')
            f.write('Output:\n')
            f.write('  file: {}   #Output filename\n'.format(input['outputfits']))
            f.write('  format: DMS          #Output file format Options: DMS, SSR(not yet implemented)\n')
            f.write('  save_intermediates: False   #Save intermediate products separately (point source image, etc)\n')
            f.write('  grism_source_image: {}   #grism\n'.format(input['grism_source_image']))
            f.write('  grism_input_only: {}     #grism\n'.format(input['grism_input_only']))
            f.write('  unsigned: True   #Output unsigned integers? (0-65535 if true. -32768 to 32768 if false)\n')
            f.write('  dmsOrient: True    #Output in DMS orientation (vs. fitswriter orientation).\n')
            f.write('  program_number: {}    #Program Number\n'.format(input['ProposalID']))
            f.write('  title: {}   #Program title\n'.format(input['Title']))
            f.write('  PI_Name: {}  #Proposal PI Name\n'.format(input['PI_Name']))
            f.write('  Proposal_category: {}  #Proposal category\n'.format(input['Proposal_category']))
            f.write('  Science_category: {}  #Science category\n'.format(input['Science_category']))
            f.write("  observation_number: '{}'    #Observation Number\n".format(input['obs_num']))
            f.write('  observation_label: {}    #User-generated observation Label\n'.format(input['obs_label']))
            f.write("  visit_number: '{}'    #Visit Number\n".format(input['visit_num']))
            f.write("  visit_group: '{}'    #Visit Group\n".format(input['visit_group']))
            f.write("  visit_id: '{}'    #Visit ID\n".format(input['visit_id']))
            f.write("  sequence_id: '{}'    #Sequence ID\n".format(input['sequence_id']))
            f.write("  activity_id: '{}'    #Activity ID. Increment with each exposure.\n".format(input['act_id']))
            f.write("  exposure_number: '{}'    #Exposure Number\n".format(input['exposure']))
            f.write("  obs_id: '{}'   #Observation ID number\n".format(input['observation_id']))
            f.write("  date_obs: '{}'  #Date of observation\n".format(input['date_obs']))
            f.write("  time_obs: '{}'  #Time of observation\n".format(input['time_obs']))
            f.write("  obs_template: '{}'  #Observation template\n".format(input['obs_template']))
            #f.write('  expstart: {}  #MJD exposure start time\n'.format(input['expstart']))
        print("Output file written to {}".format(input['yamlfile']))

        
    def reffile_setup(self):
        #create lists of reference files
        self.det_list = ['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5'] 
        sb_dir = '/ifs/jwst/wit/witserv/data4/nrc/hilbert/superbias/cv3/deliver_to_CRDS/'
        self.superbias_list = {}
        for det in self.det_list:
            self.superbias_list[det] = sb_dir+'NRC'+det+'_superbias_from_list_of_biasfiles.list.fits'

        ref_dir = '/ifs/jwst/wit/witserv/data7/nrc/reference_files/SSB/CV3/cv3_reffile_conversion/'

        lin_dir = ref_dir + 'linearity/'
        self.linearity_list = {'A1':lin_dir+'NRCA1_17004_LinearityCoeff_ADU0_2016-05-14_ssblinearity_v2_DMSorient.fits',
                               'A2':lin_dir+'NRCA2_17006_LinearityCoeff_ADU0_2016-05-14_ssblinearity_v2_DMSorient.fits',
                               'A3':lin_dir+'NRCA3_17012_LinearityCoeff_ADU0_2016-05-14_ssblinearity_v2_DMSorient.fits',
                               'A4':lin_dir+'NRCA4_17048_LinearityCoeff_ADU0_2016-05-15_ssblinearity_v2_DMSorient.fits',
                               'A5':lin_dir+'NRCALONG_17158_LinearityCoeff_ADU0_2016-05-16_ssblinearity_v2_DMSorient.fits',
                               'B1':lin_dir+'NRCB1_16991_LinearityCoeff_ADU0_2016-05-17_ssblinearity_v2_DMSorient.fits',
                               'B2':lin_dir+'NRCB2_17005_LinearityCoeff_ADU0_2016-05-18_ssblinearity_v2_DMSorient.fits',
                               'B3':lin_dir+'NRCB3_17011_LinearityCoeff_ADU0_2016-05-20_ssblinearity_v2_DMSorient.fits',
                               'B4':lin_dir+'NRCB4_17047_LinearityCoeff_ADU0_2016-05-20_ssblinearity_v2_DMSorient.fits',
                               'B5':lin_dir+'NRCBLONG_17161_LinearityCoeff_ADU0_2016-05-22_ssblinearity_v2_DMSorient.fits'}
        
        gain_dir = ref_dir + 'gain/'
        self.gain_list = {'A1':gain_dir+'NRCA1_17004_Gain_ISIMCV3_2016-01-23_ssbgain_DMSorient.fits',
                          'A2':gain_dir+'NRCA2_17006_Gain_ISIMCV3_2016-01-23_ssbgain_DMSorient.fits',
                          'A3':gain_dir+'NRCA3_17012_Gain_ISIMCV3_2016-01-23_ssbgain_DMSorient.fits',
                          'A4':gain_dir+'NRCA4_17048_Gain_ISIMCV3_2016-01-23_ssbgain_DMSorient.fits',
                          'A5':gain_dir+'NRCA5_17158_Gain_ISIMCV3_2016-01-23_ssbgain_DMSorient.fits',
                          'B1':gain_dir+'NRCB1_16991_Gain_ISIMCV3_2016-01-23_ssbgain_DMSorient.fits',
                          'B2':gain_dir+'NRCB2_17005_Gain_ISIMCV3_2016-02-25_ssbgain_DMSorient.fits',
                          'B3':gain_dir+'NRCB3_17011_Gain_ISIMCV3_2016-01-23_ssbgain_DMSorient.fits',
                          'B4':gain_dir+'NRCB4_17047_Gain_ISIMCV3_2016-02-25_ssbgain_DMSorient.fits',
                          'B5':gain_dir+'NRCB5_17161_Gain_ISIMCV3_2016-02-25_ssbgain_DMSorient.fits'}

        sat_dir = ref_dir + 'welldepth/'
        self.saturation_list = {'A1':sat_dir+'NRCA1_17004_WellDepthADU_2016-03-10_ssbsaturation_wfact_DMSorient.fits',
                                'A2':sat_dir+'NRCA2_17006_WellDepthADU_2016-03-10_ssbsaturation_wfact_DMSorient.fits',
                                'A3':sat_dir+'NRCA3_17012_WellDepthADU_2016-03-10_ssbsaturation_wfact_DMSorient.fits',
                                'A4':sat_dir+'NRCA4_17048_WellDepthADU_2016-03-10_ssbsaturation_wfact_DMSorient.fits',
                                'A5':sat_dir+'NRCA5_17158_WellDepthADU_2016-03-10_ssbsaturation_wfact_DMSorient.fits',
                                'B1':sat_dir+'NRCB1_16991_WellDepthADU_2016-03-10_ssbsaturation_wfact_DMSorient.fits',
                                'B2':sat_dir+'NRCB2_17005_WellDepthADU_2016-03-10_ssbsaturation_wfact_DMSorient.fits',
                                'B3':sat_dir+'NRCB3_17011_WellDepthADU_2016-03-10_ssbsaturation_wfact_DMSorient.fits',
                                'B4':sat_dir+'NRCB4_17047_WellDepthADU_2016-03-10_ssbsaturation_wfact_DMSorient.fits',
                                'B5':sat_dir+'NRCB5_17161_WellDepthADU_2016-03-10_ssbsaturation_wfact_DMSorient.fits'}

        ipc_dir = ref_dir + 'ipc/'
        self.ipc_list = {'A1':ipc_dir+'NRCA1_17004_IPCDeconvolutionKernel_2016-03-18_ssbipc_DMSorient.fits',
                         'A2':ipc_dir+'NRCA2_17006_IPCDeconvolutionKernel_2016-03-18_ssbipc_DMSorient.fits',
                         'A3':ipc_dir+'NRCA3_17012_IPCDeconvolutionKernel_2016-03-18_ssbipc_DMSorient.fits',
                         'A4':ipc_dir+'NRCA4_17048_IPCDeconvolutionKernel_2016-03-18_ssbipc_DMSorient.fits',
                         'A5':ipc_dir+'NRCA5_17158_IPCDeconvolutionKernel_2016-03-18_ssbipc_DMSorient.fits',
                         'B1':ipc_dir+'NRCB1_16991_IPCDeconvolutionKernel_2016-03-18_ssbipc_DMSorient.fits',
                         'B2':ipc_dir+'NRCB2_17005_IPCDeconvolutionKernel_2016-03-18_ssbipc_DMSorient.fits',
                         'B3':ipc_dir+'NRCB3_17011_IPCDeconvolutionKernel_2016-03-18_ssbipc_DMSorient.fits',
                         'B4':ipc_dir+'NRCB4_17047_IPCDeconvolutionKernel_2016-03-18_ssbipc_DMSorient.fits',
                         'B5':ipc_dir+'NRCB5_17161_IPCDeconvolutionKernel_2016-03-18_ssbipc_DMSorient.fits'}

        dist_dir = '/ifs/jwst/wit/witserv/data4/nrc/hilbert/distortion_reference_file/jwreftools/nircam/'
        self.astrometric_list = {}
        for det in self.det_list:
            self.astrometric_list[det] = dist_dir+'NRC'+det+'_FULL_distortion.asdf'

        pam_dir = '/ifs/jwst/wit/witserv/data4/nrc/hilbert/simulated_data/'
        self.pam_list = {}
        for det in self.det_list:
            self.pam_list[det] = pam_dir+'jwst_nircam_area_0001.fits'

        dark_dir = '/ifs/jwst/wit/nircam/isim_cv3_files_for_calibrations/darks/'
        self.dark_list = {}
        for det in self.det_list:
            if 'A' in det:
                mod = 'A'
            else:
                mod = 'B'
            if '5' in det:
                mdet = mod + 'LONG'
            else:
                mdet = det
            ddir = dark_dir + mdet + '/'
            dfiles = glob(ddir+'*uncal.fits')
            self.dark_list[det] = dfiles
            
                    
    def add_options(self,parser=None,usage=None):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,description='Simulate JWST ramp')
        parser.add_argument("--input_xml",help='XML file from APT describing the observations.')
        parser.add_argument("--pointing_file",help='Pointing file from APT describing observations.')
        parser.add_argument("--siaf",help='CSV version of SIAF. Needed only in conjunction with input_xml+pointing.')
        parser.add_argument("--output_dir",help='Directory into which the yaml files are output',default='./')
        parser.add_argument("--table_file",help='Ascii table containing observation info. Use this or xml+pointing+siaf files.')
        parser.add_argument("--subarray_def_file",help="Ascii file containing subarray definitions",default=None)
        parser.add_argument("--readpatt_def_file",help='Ascii file containing readout pattern definitions',default=None)
        parser.add_argument("--point_source",help='point source catalog file',nargs='*',default=[None])
        parser.add_argument("--galaxyListFile",help='galaxy (sersic) source catalog file',nargs='*',default=[None])
        parser.add_argument("--extended",help='extended source catalog file',nargs='*',default=[None])
        parser.add_argument("--convolveExtended",help='Convolve extended sources with NIRCam PSF?',action='store_true')
        parser.add_argument("--movingTarg",help='Moving (point source) target catalog (sources moving through fov)',nargs='*',default=[None])
        parser.add_argument("--movingTargSersic",help='Moving galaxy (sersic) target catalog (sources moving through fov)',nargs='*',default=[None])
        parser.add_argument("--movingTargExtended",help='Moving extended source target catalog (sources moving through fov)',nargs='*',default=[None])
        parser.add_argument("--movingTargToTrack",help='Catalog of non-sidereal targets for non-sidereal tracking obs.',nargs='*',default=[None])
        parser.add_argument("--bkgdrate",help='Uniform background rate (e-/s) to add to observation.',default=0.)
        parser.add_argument("--obsdate",help='Date string of observation, YYYY-MM-DD',default='2020-10-14')
        parser.add_argument("--obstime",help='Time string of observation, HH:MM:SS',default='16:30:44.42')
        return parser
    

if __name__ == '__main__':

    usagestring = 'USAGE: yaml_generator.py NIRCam_obs.xml NIRCam_obs.pointing'

    input = SimInput()
    parser = input.add_options(usage = usagestring)
    args = parser.parse_args(namespace=input)
    input.reffile_setup()
    input.create_inputs()





    

    
