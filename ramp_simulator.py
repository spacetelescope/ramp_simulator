#! /usr/bin/env python

'''
Rewritten version of Kevin Volk's rampsim.py. Updated to be more general
and flexible such that other instruments can use the code

Lots of documentation to add here....

V0.0 - re-wrote original script. 
V0.1 - add ability to properly combine the dark plus simulated signals,
       by linearizing the dark ramp, making the sum, and then re-
       introducing the non-linearity.
       Re-order steps a bit so that paramter check is the first thing done

1. read in parameter file (yaml)
2. make adjustments to inputs (like checkParameterfile) 


TSO work.....

NEED UPDATED PSF LIBRARY!! NEED SUB-SAMPLED PSFS FOR EVERYTHING. ALSO NEED
PSFS FOR WLP8+FILTER COMBINATIONS, AS WELL AS FW+PW COMBOS...

Allowed optics:
SW:
CLEAR+ 070,090,115,150,200,140M,182M,210M,197N,212N
WLP8+ 150,200,140M,182M,210M,197N,212N
F164N+ 150W2
F162M+ 150W2

LW:
when SW requested is [CLEAR,F162M,F164N]: then appropriate LW filters are:
250M,300M,335M,360M,410M,430M,460M,480M,277W,322W2,356W,444W
when SW requested is WLP8, allowed filters are: 322W2+323N, 444W+405N, 444W+466N, 444W+470N

Allowed readout patterns:
rapid, bright, shallow.  NO medium or deep allowed.

subarrays allowed:
64x64
160x160
400x400
full frame

adjust to create nint > 1??
or do nint=1 and output source catalogs at end of integration. Use that
as input for the next integration.

same as normal imaging, star list, galaxy list,
but create a larger image than the grism_direct_image???

1. make an extended-range signal rate frame
2. inputs: direction and speed of telescope slew
3. calculate the position of each source at the time of each frame
  3a. transform input positions to unditsorted coords
  3b. shift everything by N arsec in the appropriate direction
  3c. put back into distorted coords.
But how does this work with extended objects???
do steps 3a to 3c for all pixels that have signal > something?


4. add the right amount of signal to each pixel between the position of a given source in frame 0 and frame N, given the signal rate and the slew rate.
OR
4. repeatedly shift and add frame 0 in the appropriate direction to produce the object streaks...but this
won't take into account distortion differences across the detector...




'''
#adjust error array along the way in parallel
#check work using pixelscale. kevins original script has lots of pixelscale/3600. which 
#doesnt make sense. shouldnt you multiply by 3600 to get degrees? dividing gets you...what?


import argparse,sys, glob, os
import yaml
import scipy.ndimage.interpolation as interpolation
import scipy.signal as s1
#from jwst_lib.models import RampModel #build 6
from jwst.datamodels import RampModel #build 7
import numpy as np
import math,random
from astropy.io import fits,ascii
from astropy.table import Table
from itertools import izip
import datetime
from astropy.modeling.models import Sersic2D
import rotations  #this is Colin's set of rotation functions
from asdf import AsdfFile
import polynomial #more of Colin's functions
from astropy.convolution import convolve
import read_siaf_table
import moving_targets

inst_list = ['nircam','niriss','nirspec','miri','fgs']
modes = {'nircam':['imaging','wfss','tso'],'niriss':['imaging','wfss','soss']}
inst_abbrev = {'nircam':'NRC','nirspec':'NRS','miri':'MIRI','niriss':'NRS'}
pointSourceModes = ['imaging']
pixelScale = {'nircam':{'sw':0.031,'lw':0.063}}
full_array_size = {'nircam':2048}
allowedOutputFormats = ['DMS']
#need a list of filters that go with each mode....

class RampSim():

    def __init__(self):
        #if a grism signal rate image is requested, expand
        #the width and height of the signal rate image by this 
        #factor, so that the grism simulation software can 
        #track sources that are outside the requested subarray
        #in order to calculate contamination.
        self.grism_direct_factor = np.sqrt(2.)

        #self.coord_adjust contains the factor by which the nominal output array size
        #needs to be increased (used for TSO and WFSS modes), as well as the coordinate
        #offset between the nominal output array coordinates, and those of the expanded 
        #array. These are needed mostly for TSO observations, where the nominal output  
        #array will not sit centered in the expanded output image.
        self.coord_adjust = {'x':1.,'xoffset':0.,'y':1.,'yoffset':0.}

    def run(self):
        #read in the parameter file
        self.readParameterFile()

        #check the consistency of the input readpattern and nframe/nskip
        self.readpatternCheck()
        
        #from parameters, generate other useful variables
        self.checkParams()

        #read in the subarray definition file
        self.readSubarrayDefinitionFile()

        #define the subarray bounds from param file
        self.getSubarrayBounds()

        #this is now done in getSubarrayBounds
        #set the number of amps used in the readout 
        #self.setNumAmps()

        #read in the input dark current frame, and crop using
        #the subarray bounds provided in the parameter file
        self.getBaseDark()
        
        #make sure there is enough data (frames/groups) in the input integration to produce
        #the proposed output integration
        self.dataVolumeCheck()

        #If the proper method of combining the dark and simulated signal
        #is to be used on any pixels, then get the linearized version of
        #the dark ramp here. If it is not specified by the user, then 
        #create it on the fly using the SSB pipeline. Better to do this 
        #on the full dark before cropping, so that reference pixels can be
        #used in the processing.
        self.linDark = None
        if self.params['newRamp']['combine_method'].upper() in ['HIGHSIG','PROPER']:
            if self.runStep['linearized_darkfile'] == False:
                #before superbias subtraction, refpix subtraction and linearizing
                #Reorganize the dark according to the requested output readout pattern
                self.dark,darkzeroframe = self.reorderDark(self.dark)
                print('DARK has been reordered to match the input readpattern of {}'.format(self.dark.meta.exposure.readpatt))
                #Linearize the dark ramp via the SSB pipeline. Also save a diff image of 
                #the original dark minus the superbias and refpix subtracted dark, to use later.
                self.linDark, self.sbAndRefpixEffects = self.linearizeDark(self.dark)
                
                #Now we need to linearize the zeroframe. Place it into a RampModel instance
                #before running the pipeline steps
                if darkzeroframe != None:
                    zeroModel = RampModel()
                    zyd,zxd = darkzeroframe.shape
                    filler = np.zeros((1,1,zyd,zxd))
                    zeroModel.data = np.expand_dims(np.expand_dims(darkzeroframe,axis=0),axis=0)
                    zeroModel.err = filler
                    zeroModel.groupdq = filler
                    zeroModel.meta = self.dark.meta
                    zeroModel.meta.exposure.ngroups = 1
                    zeroModel.meta.exposure.nframes = 1
                    zeroModel.meta.exposure.groupgap = 0
                    zeroModel.meta.exposure.readpatt = 'RAPID'
                    zeroModel, zero_sbAndRefEffects = self.linearizeDark(zeroModel)
                    zero_sbAndRefEffects = zero_sbAndRefEffects[0,0,:,:]
                    #print('linearized the dark current zero frame. {}'.format(zeroModel.data.shape))

                    #now crop the zero frame to match the specified output size
                    zeroModel = self.cropDark(zeroModel)
                    zero_sbAndRefEffects = zero_sbAndRefEffects[self.subarray_bounds[1]:self.subarray_bounds[3]+1,self.subarray_bounds[0]:self.subarray_bounds[2]+1] 
                    #print('cropped the linearized dark current zero frame. {}'.format(zeroModel.data.shape))

            else:
                self.readLinearDark()
            #Crop the linearized dark to the requested
            #subarray size
            self.linDark = self.cropDark(self.linDark)

            #save the linearized dark for testing
            if self.params['Output']['save_intermediates']:
                h0=fits.PrimaryHDU()
                h1 = fits.ImageHDU(self.linDark.data)
                hl=fits.HDUList([h0,h1])
                hl.writeto(self.params['Output']['file'][0:-5] + '_linearizedDark.fits',clobber=True)

        #Now crop the dark current ramp 
        #down to the requested subarray size
        self.dark = self.cropDark(self.dark)

        #save the cropped dark for testing
        if self.params['Output']['save_intermediates']:
            h0=fits.PrimaryHDU()
            h1 = fits.ImageHDU(self.dark.data)
            hl=fits.HDUList([h0,h1])
            hl.writeto(self.params['Output']['file'][0:-5] + '_croppedDark.fits',clobber=True)
        

        #Find the difference between the cropped original dark and the cropped linearized dark
        #This will be used later to re-add the superbias and refpix-associated signal to the final
        #output ramp.
        if self.params['newRamp']['combine_method'].upper() in ['HIGHSIG','PROPER']:
            self.sbAndRefpixEffects = self.sbAndRefpixEffects[self.subarray_bounds[1]:self.subarray_bounds[3]+1,self.subarray_bounds[0]:self.subarray_bounds[2]+1] 
            
            #save this image
            #h0 = fits.PrimaryHDU(self.sbAndRefpixEffects)
            #hl = fits.HDUList([h0])
            #hl.writeto('sbAndRefpix.fits',clobber=True)
            #sys.exit()
            
        #calculate the exposure time of a single frame, based on the size of the subarray
        self.calcFrameTime()

        #calculate the rate of cosmic ray hits expected per frame
        self.getCRrate()
        
        #using the input instrument name, load appropriate 
        #instrument-specific dictionaries
        self.instrument_specific_dicts(self.params['Inst']['instrument'].lower())

        #If the output is a TSO ramp, use the slew rate and angle (and whether a grism
        #direct image is requested) to determine how much larger than nominal the
        #signal rate image should be
        if self.params['Inst']['mode'] == 'tso' or self.params['Output']['grism_source_image']:
            self.calcCoordAdjust()

        #image dimensions
        self.nominal_dims = np.array([self.subarray_bounds[3]-self.subarray_bounds[1]+1,self.subarray_bounds[2]-self.subarray_bounds[0]+1])
        self.grism_dims = (self.nominal_dims * np.array([self.coord_adjust['y'],self.coord_adjust['x']])).astype(np.int)
        print("Nominal output image dimensions: {}".format(self.nominal_dims))
        print("Grism output image dimensions: {}".format(self.grism_dims))

        #generate the signal image from the input reference files
        #The output contains signal in ADU that accumulate
        #in one frametime.
        print('Creating signal rate image of synthetic inputs.')
        frameimage = self.addedSignals()

        #if moving targets are requested (KBOs, asteroids, etc, NOT TSO mode
        #where the telescope slews), then create a RAPID integration which 
        #includes those targets
        mov_targs = None
        if self.runStep['movingTargets']:

            mov_targs = self.movingTargetInputs(self.params['simSignals']['movingTargetList'],'movingTarget')
            mov_targs_zero = mov_targs[0,:,:]

            #now rearrange the RAPID mov_targs integration to the requested readout pattern
            if self.params['Readout']['readpatt'].upper() != 'RAPID':
                mov_targs = self.changeReadPattern(mov_targs,self.params['Readout']['nframe'],self.params['Readout']['nskip'],self.params['Readout']['ngroup'])


        #moving target using an extended object
        mov_targs_ext = None
        if self.runStep['movingTargetsExtended']:
            print("Extended moving targets!!!")
            mov_targs_ext = self.movingTargetInputs(self.params['simSignals']['movingTargetExtended'],'movingTargetExtended')
            if self.params['simSignals']['movingTargetConvolveExtended']:
                 psffile = self.params['simSignals']['psfpath'] + self.params['simSignals']['psfbasename'] + '_' + self.params['Readout']['filter'].lower() + '_' + str(self.params['simSignals']['psfwfe']) + '_' + str(self.params['simSignals']['psfwfegroup']) + '_0p0_0p0.fits'
                 psf = fits.getdata(psffile)
                 for i in xrange(mov_targs_ext.shape[0]):
                     iframe = mov_targs_ext[i,:,:]
                     iframe = s1.fftconvolve(iframe,psf,mode="same")
                     mov_targs_ext[i,:,:] = iframe

            #keep the 0th frame
            mov_targs_ext_zero = mov_targs_ext[0,:,:]

            #now rearrange the RAPID mov_targs_ext integration to the requested readout pattern
            if self.params['Readout']['readpatt'].upper() != 'RAPID':
                mov_targs_ext = self.changeReadPattern(mov_targs_ext,self.params['Readout']['nframe'],self.params['Readout']['nskip'],self.params['Readout']['ngroup'])

        mov_targs_integration = None
        mov_targs_zeroframe = None
        if mov_targs is not None:
            if mov_targs_ext is not None:
                print("Moving targets integration is sum of point sources and extended targs")
                mov_targs_integration = mov_targs + mov_targs_ext
                mov_targs_zeroframe = mov_targs_zero + mov_targs_ext_zero
            else:
                print("Moving targets integration done with point sources")
                mov_targs_integration = mov_targs
                mov_targs_zeroframe = mov_targs_zero
        elif mov_targs_ext is not None:
            print("Moving targets integration done with extended targets")
            mov_targs_integration = mov_targs_ext
            mov_targs_zeroframe = mov_targs_ext_zero


        #read in cosmic ray library files if CRs are to be added to the data later
        if self.runStep['cosmicray']:
            self.readCRFiles()

        #read in saturation map
        if self.runStep['saturation_lin_limit']:
            try:
                self.satmap,self.satheader = self.readCalFile(self.params['Reffiles']['saturation'])
                bad = ~np.isfinite(self.satmap)
                self.satmap[bad] = 1.e6
            except:
                print('WARNING: unable to open saturation file {}.'.format(self.params['Reffiles']['saturation']))
                print("Please provide a valid file, or place 'none' in the saturation entry in the parameter file,")
                print("in which case the nonlin limit value in the parameter file ({}) will be used for all pixels.".format(self.params['nonlin']['limit']))
                sys.exit()
        else:
            print('CAUTION: no saturation map provided. Using {} for all pixels.'.format(self.params['nonlin']['limit']))
            dy,dx = self.dark.data.shape[2:]
            self.satmap = np.zeros((dy,dx)) + self.params['nonlin']['limit']
            
        #If some or all of the pixels will be combined using the proper method,
        #do that here
        if self.params['newRamp']['combine_method'].upper() in ['HIGHSIG','PROPER']:
            proper_ramp,proper_zero = self.doProperCombine(frameimage,mov_targs_integration,mov_targs_zeroframe,self.sbAndRefpixEffects,zeroModel.data[0,0,:,:],zero_sbAndRefEffects)
            
        #If the standard method will be used for at least some pixels, do that here.
        if self.params['newRamp']['combine_method'].upper() in ['HIGHSIG','STANDARD']:
            standard_ramp,standard_zero = self.doStandardCombine(frameimage,mov_targs_integration,mov_targs_zeroframe)

        #If the proper technique is being used on hot pixels and the standard technique
        #on darker pixels, then combine the results here.
        if self.params['newRamp']['combine_method'].upper() == 'HIGHSIG':
            final_ramp,final_zero = self.combineFinalRamps(standard_ramp,proper_ramp,standard_zero,proper_zero)
        if self.params['newRamp']['combine_method'].upper() == 'PROPER':
            final_ramp = proper_ramp
            final_zero = proper_zero
        if self.params['newRamp']['combine_method'].upper() == 'STANDARD':
            final_ramp = standard_ramp
            final_zero = standard_zero

        #set any pixels that are above their saturation level to just above their saturation
        #level. We want to prevent crazy values from popping up when the ramp is saved. When being
        #saved in DMS format, the ramp is converted to a 16-bit integer, so we don't want to feed in
        #really large values to that conversion.
        #print(final_ramp.shape)
        #bad = final_zero > self.satmap
        #minval = np.minimum(self.satmap+200.,65535)
        #final_zero[bad] = minval[bad]
        
        #Set any pixels with signals above 65535 to 65535.  We want to prevent crazy values from 
        #popping up when the ramp is saved. When being saved in DMS format, the ramp is converted 
        #to a 16-bit integer, so we don't want to feed in really large values to that conversion.
        #Similarly, set any large negative values to zero. These values most likely
        #come from doNonLin, where a bad initial guess or bad linearity coefficients
        #caused the solution to fail to converge
        if final_zero is not None:
            bad = final_zero > 65535
            final_zero[bad] = 65535
            bad2 = final_zero < 0
            final_zero[bad2] = 0

        for grp in xrange(final_ramp.shape[0]):
            frame = final_ramp[grp,:,:]
            bad = frame > 65535
            frame[bad] = 65535
            bad2 = frame < 0
            frame[bad2] = 0
            final_ramp[grp,:,:] = frame

        #save the integration
        if self.params['Output']['format'].upper() == 'DMS':
            print("Saving integration in DMS format.")
            self.saveDMS(final_ramp,final_zero,self.params['Output']['file'])
        else:
            print("Saving the output data in a format other than DMS not yet implemented!!!")
            sys.exit()


    def changeReadPattern(self,array,outnframe,outnskip,outngroup):
        #Assume an input integration with a RAPID readout pattern
        #Average/skip frames to convert the integration to another pattern
        ingroup,iny,inx = array.shape
        newpatt = np.zeros((outngroup,iny,inx))

        #total number of frames per group
        deltaframe = outnskip + outnframe

        #indexes of frames to average
        frames = np.arange(outnskip,deltaframe)
        #accumimage = np.zeros((iny,inx),dtype=np.int32)

        #Loop over groups
        for i in range(outngroup):
            #average together the appropriate frames, skip the appropriate frames

            #If averaging needs to be done
            if outnframe > 1:
                newpatt[i,:,:] = np.mean(array[frames,:,:],axis=0)

            #If no averaging needs to be done
            else:
                newpatt[i,:,:] = array[frames[0],:,:]

        return newpatt




    def movingTargetInputs(self,file,mode):
        #read in listfile of moving targets and perform needed calculations to get inputs
        #for moving_targets.py

        #outputs
        #target_dict = {}
        #dictionary? or just a list of tuples?

        #mode can be 'TSO' or 'movingTarget'

        #TRY TO MAKE THIS FUNCTION FLEXIBLE, SO THAT WE CAN USE IT FOR THE POINT SOURCE
        #MOVING TARGETS, EXTENDED MOVING TARGETS, AND ALSO THE TSO MODE, WHERE THE INPUTS
        #WILL BE THE POINT SOURCE LIST, GALAXY LIST, AND EXTENDED IMAGE.

        #get countrate for mag 15 source, for scaling later
        try:
            mag15rate = self.countvalues[self.params['Readout']['filter']]
        except:
            print("Unable to find mag 15 countrate for {} filter in {}.".format(self.params['Readout']['filter'],self.params['Reffiles']['phot']))
            print("Fix me!")
            sys.exit()

        #read input file
        if self.runStep['movingTargets']:
            mtlist,pixelFlag,pixvelflag = self.readMTFile(file)
        elif self.runStep['TSO']:
            print('fixme! implement TSO')
            sys.exit()

        #get necessary information for coordinate transformations
        coord_transform = None
        if self.runStep['astrometric']:

            #Read in the CRDS-format distortion reference file
            with AsdfFile.open(self.params['Reffiles']['astrometric']) as dist_file:
                coord_transform = dist_file.tree['model']

        #Using the requested RA,Dec of the reference pixel, along with the 
        #V2,V3 of the reference pixel, and the requested roll angle of the telescope
        #create a matrix that can be used to translate between V2,V3 and RA,Dec
        #for any pixel.
        #v2,v3 need to be in arcsec, and RA, Dec, and roll all need to be in degrees
        attitude_matrix = rotations.attitude(self.refpix_pos['v2'],self.refpix_pos['v3'],self.ra,self.dec,self.params['Telescope']["rotation"])

        if mode == 'TSO':
            #calculate the velocity of the telescope slew in arcsec/sec
            print('still need to implement. Fix me!!!')
            sys.exit()
            ra_vel = 0.0
            dec_vel = 0.0

        #exposure times for all frames
        frameexptimes = self.frametime * np.arange(-1,self.params['Readout']['ngroup'] * (self.params['Readout']['nframe'] + self.params['Readout']['nskip']))
            
        #output image dimensions
        dims = np.array(self.dark.data[0,0,:,:].shape)
        newdimsx = np.int(dims[1] * self.coord_adjust['x'])
        newdimsy = np.int(dims[0] * self.coord_adjust['y'])
        mt_integration = np.zeros((len(frameexptimes)-1,newdimsy,newdimsx))

        for entry in mtlist:

            #for each object, calculate x,y or ra,dec of initial position
            pixelx,pixely,ra,dec,ra_str,dec_str = self.getPositions(entry[0],entry[1],attitude_matrix,coord_transform,pixelFlag)

            print("got initial x,y for moving target")

            #now generate a list of x,y position in each frame
            if pixvelflag == False:
                #calculate the RA,Dec in each frame
                ra_frames = ra + (entry[3]/3600.) * frameexptimes
                dec_frames = dec + (entry[4]/3600.) * frameexptimes

                x_frames = []
                y_frames = []
                for in_ra,in_dec in izip(ra_frames,dec_frames):
                    #calculate the x,y position at each frame
                    px,py,pra,pdec,pra_str,pdec_str = self.getPositions(in_ra,in_dec,attitude_matrix,coord_transform,False)
                    x_frames.append(px)
                    y_frames.append(py)
                x_frames = np.array(x_frames)
                y_frames = np.array(y_frames)
                    
            else:
                #if input velocities are pixels/hour, then generate the list of
                #x,y in each frame directly
                x_frames = pixelx + (entry[3]/3600.) * frameexptimes
                y_frames = pixely + (entry[4]/3600.) * frameexptimes
                        
                
            #So now we have xinit,yinit, a list of x,y positions for each frame, and the frametime.
            #subsample factor can be hardwired for now. outx and outy are also known. So all we need is the stamp
            #image, then we can call moving_targets.py and feed it these things, which contain all the info needed

            if mode == 'movingTarget':
                psffile = self.params['simSignals']['psfpath'] + self.params['simSignals']['psfbasename'] + '_' + self.params['Readout']['filter'].lower() + '_' + str(self.params['simSignals']['psfwfe']) + '_' + str(self.params['simSignals']['psfwfegroup']) + '_0p0_0p0.fits'

                stamp = fits.getdata(psffile)
            elif mode == 'movingTargetExtended':
                #stamp = self.getData(entry[0])
                stamp = self.getImage(entry[0],arrayshape,rotateflag,angle,place=[0,0]) 

            #Scale the stamp image to the requested magnitude
            scale = 10.**(0.4*(15.0-entry['magnitude']))
            rate = scale * mag15rate
            stamp *= rate

            #each entry will have stamp image as array, ra_init,dec_init,ra_velocity,dec_velocity,frametime,numframes,subsample_factor,outputarrayxsize,outputarrayysize (maybe without the values that will be the same to each entry.

            #entryList = (stamp,ra,dec,entry[3]/3600.,entry[4]/3600.,self.frametime,numframes,subsample_factor,outx,outy)
            #entryList = (stamp,x_frames,y_frames,self.frametime,subsample_factor,outx,outy)
            mt = moving_targets.MovingTarget()
            mt.subsampx = 3
            mt.subsampy = 3
            mt_integration += mt.create(stamp,x_frames,y_frames,self.frametime,newdimsx,newdimsy)

        #save the moving target input integration
        if self.params['Output']['save_intermediates']:
            h0 = fits.PrimaryHDU(mt_integration)
            hl = fits.HDUList([h0])
            mtoutname = self.params['Output']['file'][0:-5] + '_movingTargetIntegration.fits'
            hl.writeto(mtoutname,clobber=True)
            print("Integration showing only moving targets saved to {}".format(mtoutname))

        return mt_integration


    def getPositions(self,inx,iny,matrix,transform,pixelflag):
        #input a row containing x,y or ra,dec values, and figure out
        #x,y, RA, Dec, and RA string and Dec string 
        try:
            entry0 = float(inx)
            entry1 = float(iny)
            if not pixelflag:
                ra_str,dec_str = self.makePos(entry0,entry1)
                ra = entry0
                dec = entry1
        except:
            #if inputs can't be converted to floats, then 
            #assume we have RA/Dec strings. Convert to floats.
            ra_str = inx
            dec_str = iny
            ra,dec = self.parseRADec(ra_str,dec_str)

        #Case where point source list entries are given with RA and Dec
        if not pixelflag:

            #If distortion is to be included - either with or without the full set of coordinate
            #translation coefficients
            if self.runStep['astrometric']:
                pixelx,pixely = self.RADecToXY_astrometric(ra,dec,matrix,transform)
            else:
                #No distortion at all - "manual mode"
                pixelx,pixely = self.RADecToXY_manual(ra,dec)

        else:
            #Case where the point source list entry locations are given in units of pixels
            #In this case we have the source position, and RA/Dec are calculated only so 
            #they can be written out into the output source list file.

            pixelx = entry0
            pixely = entry1

            ra,dec,ra_str,dec_str = self.XYToRADec(pixelx,pixely,matrix,transform)
        return pixelx,pixely,ra,dec,ra_str,dec_str



    def calcCoordAdjust(self):
        #calculate the factors by which to expand the output array size, as well as the coordinate
        #offsets between the nominal output array and the input lists if the observation being
        #modeled is TSO or TSO+grism output modes

        dtor = math.radians(1.)

        if self.params['Inst']['mode'] == 'tso':
            grouptime = self.frametime * (self.params['Readout']['nframe'] + self.params['Readout']['nskip'])
            inttime = grouptime * self.params['Readout']['ngroup']
            slewDist = self.params['Telescope']['slewRate'] * inttime #arcsecs
            slewDistx = slewDist * np.cos((270.+self.params['Telescope']['slewAngle'])*dtor)
            slewDisty = slewDist * np.sin((270.+self.params['Telescope']['slewAngle'])*dtor)
            
            #How many pixels in x and y is the total slew?
            slewPixx = slewDistx / self.pixscale[0]
            slewPixy = slewDisty / self.pixscale[1]

            #What factor is this of the total output image size? 
            self.coord_adjust['x'] = 1. + slewPixx / (self.subarray_bounds[2]-self.subarray_bounds[0]+1)
            self.coord_adjust['y'] = 1. + slewPixy / (self.subarray_bounds[3]-self.subarray_bounds[1]+1)
            self.coord_adjust['xoffset'] = 0.
            self.coord_adjust['yoffset'] = 0.
           
            print('total integration time {}. total slew in arcsec {}'.format(inttime,slewDist))
            print('slew in pixels {} {}, slew in arcsec {} {}:'.format(slewPixx,slewPixy,slewDistx,slewDisty))
            print('factors {} {}'.format(self.coord_adjust['x'],self.coord_adjust['y']))

            #requested TSO AND grism direct image
            if self.params['Output']['grism_source_image']:
                self.coord_adjust['x'] = (self.coord_adjust['x'] + self.grism_direct_factor - 1.)
                self.coord_adjust['y'] = (self.coord_adjust['y'] + self.grism_direct_factor - 1.)
                self.coord_adjust['xoffset'] = np.int((self.grism_direct_factor - 1.) * (self.subarray_bounds[2]-self.subarray_bounds[0]+1) / 2.)
                self.coord_adjust['yoffset'] = np.int((self.grism_direct_factor - 1.) * (self.subarray_bounds[3]-self.subarray_bounds[1]+1) / 2.)

                print('factors {} {}'.format(self.coord_adjust['x'],self.coord_adjust['y']))


        #Normal imaging with grism image requested
        if self.params['Output']['grism_source_image']:
            self.coord_adjust['x'] = self.grism_direct_factor
            self.coord_adjust['y'] = self.grism_direct_factor
            self.coord_adjust['xoffset'] = np.int((self.grism_direct_factor - 1.) * (self.subarray_bounds[2]-self.subarray_bounds[0]+1) / 2.)
            self.coord_adjust['yoffset'] = np.int((self.grism_direct_factor - 1.) * (self.subarray_bounds[3]-self.subarray_bounds[1]+1) / 2.)
                


    def combineFinalRamps(self,standard_ramp,proper_ramp,standard_zero,proper_zero):
        #If the proper technique is being used on hot pixels and the standard technique
        #on darker pixels, then combine the results here.
        
        #Get the map of cosmic rays added to the data
        crmap = np.zeros_like(proper_ramp[-1,:,:]).astype('bool')
        if self.runStep['cosmicray']:
            crs = ascii.read(self.params['Output']['file'][0:-5] + '_cosmicrays.list',data_start=2)
            for x,y,z in izip(crs['Image_x'],crs['Image_y'],crs['Max_CR_Signal']):
                if z >= self.params['newRamp']['proper_signal_limit']:
                    crmap[x,y] = True
                    
        #Does the user want hot pixels, CRs, or both combined with the proper method?
        if self.params['newRamp']['proper_combine'].upper() in ['HOTPIX','BOTH']:
            #If hot pix are to be used, read in the hot pixel mask, if supplied.
            if self.runStep['hotpixfile']:
                hotpix,hotpixhead = self.readCalFile(self.params['Refffiles']['hotpixfile'])
                hotpix = hotpix.astype('bool')
            else:
                print("Hot pixel mask not provided. Will use the proper technique on all pixels with")
                print("signal levels above {} in the final group.".format(self.params['newRamp']['proper_signal_limit']))
                finalprop = proper_ramp[-1,:,:]
                hotpix = np.logical_and(finalprop >= self.params['newRamp']['proper_signal_limit'],crmap == False)           
        #Put together a final mask of pixels to combine with the proper method
        if self.params['newRamp']['proper_combine'].upper() == 'COSMICRAYS':
            finalmask = crmap
        if self.params['newRamp']['proper_combine'].upper() == 'HOTPIX':
            finalmask = hotpix
        if self.params['newRamp']['proper_combine'].upper() == 'BOTH':
            finalmask = np.logical_or(hotpix,crmap)
                
        #Now combine the data
        g,y,x = standard_ramp.shape
        final_ramp = np.zeros((g,y,x))
        for group in xrange(standard_ramp.shape[0]):
            stdframe = standard_ramp[group,:,:]
            proframe = proper_ramp[group,:,:]
            grp = np.zeros((y,x))
            grp[finalmask] = proframe[finalmask]
            grp[~finalmask] = stdframe[~finalmask]
            final_ramp[group,:,:] = grp

        final_zero = np.zeros((y,x))    
        final_zero[finalmask] = proper_zero[finalmask]
        final_zero[~finalmask] = standard_zero[~finalmask]

        return final_ramp,final_zero

    

    def doStandardCombine(self,signalimg,moving_targs,moving_targs_zero):
            
        #signal adjustments - (gain, poisson noise, CR hits)
        #Return a full ramp where each frame is constructed based on the
        #single signalimage frame.
        print('Creating a synthetic signal ramp from the synthetic signal rate image using the standard technique.')
        synthetic_ramp,syn_zeroframe = self.standardFrameToRamp(signalimg)

        #Ensure that no signals are present in the reference pixels if present
        #by using a mask
        maskimage = np.zeros((self.ffsize,self.ffsize),dtype=np.int)
        maskimage[4:self.ffsize-4,4:self.ffsize-4] = 1.
        
        #crop the mask to match the requested output array
        if self.params['Readout']['array_name'] != "FULL":
            maskimage = maskimage[self.subarray_bounds[1]:self.subarray_bounds[3]+1,self.subarray_bounds[0]:self.subarray_bounds[2]+1]

        synthetic_ramp *= maskimage
        syn_zeroframe *= maskimage

        
        #if intermediate products are being saved, save the synthetic signal ramp
        if self.params['Output']['save_intermediates']:
            synthName = self.params['Output']['file'][0:-5] + '_syntheticSignalRamp.fits'
            self.saveSingleFits(synthetic_ramp,synthName)
            print("Ramp of synthetic signals saved as {}".format(synthName))
            
        #add the synthetic ramp to the dark current ramp,
        #making sure to adjust the dark current ramp if necessary
        #to account for nframe/nskip values
        print('Adding the synthetic signal ramp to the base dark current ramp.')
        standard_ramp,standard_zeroframe = self.addSyntheticToDark(synthetic_ramp,self.dark,syn_zeroframe=syn_zeroframe)

        #Add moving targets integration if present
        if moving_targs is not None:
            standard_ramp += moving_targs
            if standard_zeroframe is not None:
                standard_zeroframe += moving_targs_zero

        return standard_ramp,standard_zeroframe



    def doProperCombine(self,signalimage,moving_targs_int,moving_targs_zeroframe,refpix_effects,zeroframe,zeroRefEffects):
        #make linear synthetic ramp, including any necessary frame averaging
        print('Creating a synthetic signal ramp from the synthetic signal rate image using the proper technique.')

        if self.params['Inst']['mode'] != 'tso':
            syn_linear_ramp,syn_linear_zero = self.frameToRamp(signalimage)
        else:
            #When creating a TSO ramp, start with a signal rate image rather than an image
            #of the signal per frame
            print('If distortion is added when signalimage is created, you need to remove it here')
            print('since the scene will be different for each group here. distortion will have to be')
            print('added once each group is created. Maybe this argues that that should be the standard')
            print('practice for all modes.')
            sys.exit()
            syn_linear_ramp,syn_linear_zero = self.TSOframeToRamp(signalimage/self.frametime)


        #Ensure that no signals are present in the reference pixels if present
        #by using a mask
        maskimage = np.zeros((self.ffsize,self.ffsize),dtype=np.int)
        maskimage[4:self.ffsize-4,4:self.ffsize-4] = 1.
        
        #crop the mask to match the requested output array
        if self.params['Readout']['array_name'] != "FULL":
            maskimage = maskimage[self.subarray_bounds[1]:self.subarray_bounds[3]+1,self.subarray_bounds[0]:self.subarray_bounds[2]+1]

        syn_linear_ramp *= maskimage
        syn_linear_zero *= maskimage

            
        #if intermediate products are being saved, save the synthetic signal ramp
        if self.params['Output']['save_intermediates']:
            synthName = self.params['Output']['file'][0:-5] + '_LinearSyntheticSignalRamp.fits'
            self.saveSingleFits(syn_linear_ramp,synthName)
            print("Ramp of linear synthetic signals saved as {}".format(synthName))

        #add synthetic ramp to the dark ramp
        print('Adding the synthetic signal ramp to the dark current ramp.')
        lin_outramp,lin_zeroframe = self.addSyntheticToDark(syn_linear_ramp,self.linDark,syn_zeroframe=None)

        #add the dark current zero frame to the synthetic signal zero frame
        if zeroframe is not None:
            lin_zeroframe = syn_linear_zero + zeroframe
        else:
            lin_zeroframe = None

        #Add the moving target integration if it is present
        if moving_targs_int is not None:
            lin_outramp += moving_targs_int
            if lin_zeroframe is not None:
                lin_zeroframe += moving_targs_zeroframe

        #if intermediate products are being saved, save the synthetic signal ramp
        if self.params['Output']['save_intermediates']:
            synthName2 = self.params['Output']['file'][0:-5] + '_LinearSyntheticPlusDarkSignalRamp.fits'
            self.saveSingleFits(lin_outramp,synthName2)
            print("Ramp of linear synthetic plus dark signals saved as {}".format(synthName2))
    
        #If requested, insert non-linearity into the summed ramp
        if self.runStep['nonlin']:
            nonlin,nonlinheader = self.readCalFile(self.params['Reffiles']['linearity'])
            #If the saturation map is given, don't settle for a single number for the nonlin limit. 
            #Read in saturation map, and translate into a linearity-corrected saturation map, and
            #use that as the nonlin limit. If the saturation map isn't provided, then fall back
            #to a single value.
            if self.runStep['saturation_lin_limit']:
            #    satmap,sathead = self.readCalFile(self.params['Reffiles']['saturation'])
        
                #Remove the non-linearity from the saturation map values, arriving at a
                #saturation map of corrected signals
                adj_satmap = self.nonLinFunc(self.satmap,nonlin,self.satmap)
            else:
                adj_satmap = np.zeros_like(newimage) + self.params['nonlin']['limit']


            #Now insert the non-linearity into the ramp as well as the zeroframe if it exists
            properramp = self.doNonLin(lin_outramp,nonlin,adj_satmap)
            if lin_zeroframe is not None:
                properzero = self.doNonLin(lin_zeroframe,nonlin,adj_satmap)
            else:
                properzero = None
        else:
            properramp = lin_outramp
            properzero = lin_zeroframe

        #Save for testing
        h0 = fits.PrimaryHDU()
        h1 = fits.ImageHDU(properramp)
        hl = fits.HDUList([h0,h1])
        ffff = self.params['Output']['file'][0:-5] + '_synPlusDark_addNonlin.fits'
        hl.writeto(ffff,clobber=True)


        #Look for pixels with crazy values. These are most likely due to the Newton's Method being
        #used to insert the non-linearity failing to converge, probably from bad linearity coefficients
        #or a bad initial guess. Set these pixels' signals to zero
        bad = ((properramp < -100) | (properramp > 1e5))
        badlast = ((properramp[-1,:,:] < -100) | (properramp[-1,:,:] > 1e5))
        #properramp[bad] = 0.
        numbad = np.sum(badlast)
        yb,xb = properramp[-1,:,:].shape
        print("Non-linearity added in to combined synthetic+dark signal ramp. {} pixels in the final group, ({}% of the detector)".format(numbad,numbad*1./(xb*yb)))
        print("have bad signal values. These will be set to 0 or 65535 when the integration is saved.")

        nonconvmap = np.zeros_like(properramp)
        nonconvmap[bad] = 1
        if self.params['Output']['save_intermediates']:
            h0 = fits.PrimaryHDU()
            h1 = fits.ImageHDU(nonconvmap)
            hl = fits.HDUList([h0,h1])
            crazymap = self.params['Output']['file'][0:-5] + '_synPlusDark_lin_nonconverging_pix.fits'
            hl.writeto(crazymap,clobber=True)
            print("A map of these pixels has been saved to {}".format(crazymap))

        #Now add the superbias and reference pixel effects back into the ramp to get it back
        #to a 'raw' state
        properramp += refpix_effects[0,:,:,:]
        properzero += zeroRefEffects

        #To simulate raw data, we need integers
        properramp = np.around(properramp)
        properramp = properramp.astype(np.int32)
        if properzero is not None:
            properzero = np.around(properzero)
            properzero = properzero.astype(np.int32)
            
        return properramp,properzero
        

    def standardFrameToRamp(self,frameimg):
        #Once we have a frame that contains the signal added from external sources, 
        #we need to adjust these signals to account for detector effects (poisson
        #noise, gain, and we need to add cosmic rays if requested. 
        #Using the base image, multiply by exposure time and generate the collection
        #of frames/groups that will form the final integration

        yd,xd = frameimg.shape

        #output variables
        outimage = np.zeros((self.params['Readout']['ngroup'],yd,xd),dtype=np.float)
        totalsignalimage = np.zeros_like(frameimg,dtype=np.float)
        oldsignalimage = np.zeros_like(frameimg,dtype=np.float)
        zframe = None

        #read in gain map to be used below
        if self.runStep['gain']:
            gainim,gainhead = self.readCalFile(self.params['Reffiles']['gain'])
            #set any NaN's to 1.0
            bad = ~np.isfinite(gainim)
            gainim[bad] = 1.0

            #Pixels that have a gain value of 0 will be reset to have values of 1.0
            zeros = gainim == 0
            gainim[zeros] = 1.0

        #read in non-linearity coefficients to be used later
        if self.runStep['nonlin']:
            nonlin,nonlinheader = self.readCalFile(self.params['Reffiles']['linearity'])
            #set any NaN's to 0
            bad = ~np.isfinite(nonlin)
            nonlin[bad] = 0.

        #set up functions to apply cosmic rays later
        #Need the total number of active pixels in the output array to multiply the CR rate by
        if self.runStep['cosmicray']:
            npix = int(frameimg.shape[0]*frameimg.shape[1]+0.02)
            crhits,crs_perframe = self.CRfuncs(npix)

            #open output file to contain the list of cosmic rays
            crlistout = self.params['Output']['file'][0:-5] + '_cosmicrays.list'
            self.openCRListFile(crlistout,crhits)

            #counter for use in cosmic ray addition while looping over frames
            framenum = 0

        #difference between the latest outimage frame and the latest newsignalimage frame
        #This is important when nframe>1
        delta = 0.

        #Loop over each group
        for i in range(self.params['Readout']['ngroup']):
            accumimage=np.zeros_like(frameimg,dtype=np.int32)

            #Loop over frames within each group if necessary
            #create each frame
            for j in range((self.params['Readout']['nframe']+self.params['Readout']['nskip'])):

                #add poisson noise 
                workimage = self.doPoisson(frameimg)
                #print('poisson noise turned off for testing')
                #workimage = np.copy(frameimg)

                #add cosmic rays
                #print('cosmic rays turned off for testing')
                if self.runStep['cosmicray']:
                    workimage = self.doCosmicRays(workimage,i,j,self.params['Readout']['nframe'],crs_perframe[framenum])
                    framenum = framenum + 1

                #apply the inverse gain to go from electrons to ADU
                if self.runStep['gain']:
                    invgainimage = 1./gainim
                    workimage = workimage * invgainimage

                #Adjust values that are above the saturation level. The non-linearity correction
                #is only valid from 0 to the saturation level. Signals higher than this will
                #a) produce garbage output from the non-lin function, and b) can be large enough
                #to cause an overflow in the multiplication (i.e. the signal values are larger
                #than the maximum 32-bit float value). Saturated pixels should have a value of 0
                #in workimage, since workimage is added to totalsignalimage and it is the latter
                #that is run through doNonLin.
                #satpix = workimage > lin_satmap
                #workimage[satpix] = 0.
                
                #introduce non-linearity into the ramps
                if self.runStep['nonlin']:
                    totalsignalimage = totalsignalimage + workimage
                    newsignalimage = self.doNonLin(totalsignalimage,nonlin,self.satmap)

                    #print('out of dononlin:',newsignalimage[799,803])

                    workimage = newsignalimage - oldsignalimage
                    oldsignalimage = np.copy(newsignalimage)

                    #print('signal added to previous read before making int: ',workimage[799,803])
                    
                    #print('max,min values in non-linearized image after nonlin',np.nanmax(newsignalimage),np.nanmin(newsignalimage))
                    #print('max,min values in workimage after nonlin',np.nanmax(workimage),np.nanmin(workimage))

                    
                #Here is where we check to see which frame we are working on.
                #If NSKIP != 0, then make sure we're working on a frame that is
                #kept before proceeding. It's important to update totalsignalimage,
                #newsignalimage, and old signalimage even for the frames that aren't kept.
                #take this out-if j < self.params['Readout']['nframe']:

                # now round off and truncate to integers, simulating the A/D conversion
                # workimage is the signal accumulated in the current frame
                #NOTE: any NaN values will be translated into -2147483648
                workimage = np.around(workimage)
                workimage = workimage.astype(np.int32)
                #print('signal added to previous read after making int: ',workimage[799,803])
                    
                #if the frame is one that is kept (one of the last nframe frames):
                if j >= self.params['Readout']['nskip']:
                    #if nframe is > 1, then we need to average the nframe frames and
                    #place that into the group.
                    if self.params['Readout']['nframe'] != 1:
                        print('averaging frame {}'.format(i*(self.params['Readout']['nframe']+self.params['Readout']['nskip'])+j))
                        accumimage = accumimage + ((self.params['Readout']['nframe']-(j-self.params['Readout']['nskip']))*1./self.params['Readout']['nframe'])*workimage
                    else:
                        #if nframe=1, then no averaging is necessary
                        #Is doing this faster than just using the averaging line above for nframe=1?
                        print('adding frame {}'.format(i*(self.params['Readout']['nframe']+self.params['Readout']['nskip'])+j))
                        accumimage = accumimage + workimage
                else:
                    #frames that are skipped, according to nskip. The signal from these frames 
                    #still must be included in the ramp, but not averaged into the group in the 
                    #case where nframe is more than 1.
                    print('skipping frame {}'.format(i*(self.params['Readout']['nframe']+self.params['Readout']['nskip'])+j))
                    accumimage = accumimage + workimage


                #print('accumimage: ',accumimage[799,803])
                #print('max,min values in accumimage ',np.nanmax(accumimage),np.nanmin(accumimage))
                    
                #if working on the zeroth frame, save for possible output
                if ((i == 0) and (j == 0) and (self.dark.meta.exposure.readpatt == 'RAPID')):
                    zframe = workimage
                
            #Now put the signal group into the output cube
            accumimage=accumimage.astype(np.float)
            if i > 0:
                outimage[i,:,:] = outimage[i-1,:,:] + accumimage + delta
            else:
                outimage[i,:,:] = accumimage
            #print('final output signal: ',outimage[i,799,803])
            print("Group {} has been generated.".format(i+1))
            #print('max,min values of finished group image ',np.max(outimage[i,:,:]),np.min(outimage[i,:,:]))

            #Calculate the delta between outimage and the last iteration of newsignal. This needs to be
            #added to the next iteration of outimage. (This is important for cases where nframe>1)
            delta = newsignalimage - outimage[i,:,:]
            #print('i = {}, delta = {}'.format(i+1,delta[799,803]))

        if self.runStep['cosmicray']:
            #close the cosmic ray list file
            self.cosmicraylist.close()



        #Look for pixels with crazy values. These are most likely due to the Newton's Method being
        #used to insert the non-linearity failing to converge, probably from bad linearity coefficients
        #or a bad initial guess. Set these pixels' signals to zero
        bad = ((outimage < -100) | (outimage > 1e10))
        badlast = ((outimage[-1,:,:] < -100) | (outimage[-1,:,:] > 1e10))
        #outimage[bad] = 0.
        numbad = np.sum(badlast)
        yb,xb = outimage[-1,:,:].shape
        print("Non-linearity added in to synthetic signal ramp. {} pixels in the final group, ({}% of the detector)".format(numbad,numbad*1./(xb*yb)))
        print("have bad signal values. These will be set to 0 or 65535 when the integration is saved.")

        nonconvmap = np.zeros_like(outimage)
        nonconvmap[bad] = 1
        if self.params['Output']['save_intermediates']:
            h0 = fits.PrimaryHDU()
            h1 = fits.ImageHDU(nonconvmap)
            hl = fits.HDUList([h0,h1])
            crazymap = self.params['Output']['file'][0:-5] + '_lin_nonconverging_pix.fits'
            hl.writeto(crazymap,clobber=True)
            print("A map of these pixels has been saved to {}".format(crazymap))

        return outimage,zframe

    def linearizeDark(self,dark):
        #Beginning with the input dark current ramp, run the dq_init, saturation, superbias
        #subtraction, refpix and nonlin pipeline steps in order to produce a linearized
        #version of the ramp. This will be used when combining the dark ramp with the 
        #simulated signal ramp.
        from jwst.dq_init import DQInitStep
        from jwst.saturation import SaturationStep
        from jwst.superbias import SuperBiasStep
        from jwst.refpix import RefPixStep
        from jwst.linearity import LinearityStep

        #Build 6
        #from jwst_pipeline.dq_init import DQInitStep
        #from jwst_pipeline.saturation import SaturationStep
        #from jwst_pipeline.superbias import SuperBiasStep
        #from jwst_pipeline.refpix import RefPixStep
        #from jwst_pipeline.linearity import LinearityStep

        print('Creating a linearized version of the dark current input ramp.')
        
        #Run the DQ_Init step
        #print(self.params['newRamp']['dq_configfile'],os.path.isfile(self.params['newRamp']['dq_configfile']))
        linDark = DQInitStep.call(dark,config_file=self.params['newRamp']['dq_configfile'])
        
        #If the saturation map is provided, use it. If not, default to whatever is in CRDS
        if self.runStep['saturation_lin_limit']:
            linDark = SaturationStep.call(linDark,config_file=self.params['newRamp']['sat_configfile'],override_saturation=self.params['Reffiles']['saturation'])
        else:
            linDark = SaturationStep.call(linDark,config_file=self.params['newRamp']['sat_configfile'])

        #If the superbias file is provided, use it. If not, default to whatever is in CRDS
        if self.runStep['superbias']:
            linDark = SuperBiasStep.call(linDark,config_file=self.params['newRamp']['superbias_configfile'],override_superbias=self.params['Reffiles']['superbias'])
        else:
            linDark = SuperBiasStep.call(linDark,config_file=self.params['newRamp']['superbias_configfile'])


        #save the refpix subtracted dark for testing
        if self.params['Output']['save_intermediates']:
            print('save the superbias-subtracted dark for testing')
            h0=fits.PrimaryHDU()
            h1 = fits.ImageHDU(linDark.data)
            hl=fits.HDUList([h0,h1])
            hl.writeto(self.params['Output']['file'][0:-5] + '_sbSubbedDark.fits',clobber=True)


        #Reference pixel correction
        linDark = RefPixStep.call(linDark,config_file=self.params['newRamp']['refpix_configfile'])


        #save the sb and refpix subtracted dark for testing
        if self.params['Output']['save_intermediates']:
            print('save the sb and refpix subtracted dark for testing')
            h0=fits.PrimaryHDU()
            h1 = fits.ImageHDU(linDark.data)
            hl=fits.HDUList([h0,h1])
            hl.writeto(self.params['Output']['file'][0:-5] + '_sbAndRefpixSubbedDark.fits',clobber=True)


        #save a copy of the superbias- and reference pixel-subtracted dark. This will be used later
        #to add these effects back in after the synthetic signals have been added and the non-linearity
        #effects are added back in when using the PROPER combine method.
        if self.params['newRamp']['combine_method'].upper() in ['HIGHSIG','PROPER']:
            sbAndRefpixEffects = dark.data - linDark.data

        #Linearity correction - save the output so that you won't need to re-run the pipeline when using the same
        #dark current file in the future.
        #Use the linearity coefficient file if provided
        linearoutfile = self.params['Output']['file'][0:-5] + '_linearized_dark_current_ramp.fits'
        if self.runStep['linearity']:
            linDark = LinearityStep.call(linDark,config_file=self.params['newRamp']['linear_configfile'],override_linearity=self.params['Reffiles']['linearity'],output_file=linearoutfile)
        else:
            linDark = LinearityStep.call(linDark,config_file=self.params['newRamp']['linear_configfile'],output_file=linearoutfile)

        print('Linearized dark saved as {}'.format(linearoutfile))
        return linDark,sbAndRefpixEffects

        
    def readLinearDark(self):
        #Read in the linearized version of the dark current ramp
        try:
            print('Reading in linearized dark current ramp from {}'.format(self.params['newRamp']['linearized_darkfile']))
            self.linDark = RampModel(self.params['newRamp']['linearized_darkfile'])
        except:
            print('WARNING: Unable to read in linearized dark ramp.')
            sys.exit()
            
    def saveDMS(self,ramp,zeroframe,filename):
        #save the new, simulated integration in DMS format (i.e. DMS orientation 
        #rather than raw fitswriter orientation, and using SSBs RampModel)
        
        #make sure the ramp to be saved has the right number of dimensions
        imshape = ramp.shape
        if len(imshape) == 3:
            ramp = np.expand_dims(ramp,axis=0)
            
        #insert data into model
        self.dark.data = ramp
        self.dark.err = np.zeros_like(ramp,dtype=np.int32)
        self.dark.groupdq = np.zeros_like(ramp,dtype=np.uint8)

        if len(self.dark.err.shape) == 3:
            self.dark.err = np.expand_dims(self.dark.err,axis=0)

        if len(self.dark.groupdq.shape) == 3:
            self.dark.groupdq = np.expand_dims(self.dark.groupdq,axis=0)

        #if saving the zeroth frame is requested, insert into the model instance
        if zeroframe is not None:
            #if the zeroframe is a 2D image, then add a dimension,
            #as the model expects 3D
            if len(zeroframe.shape) == 2:
                zeroframe = np.expand_dims(zeroframe,0)
                #place the zero frame into the model
                self.dark.zeroframe = zeroframe

        #EXPTYPE OPTIONS
        #exptypes = ['NRC_IMAGE','NRC_GRISM','NRC_TACQ','NRC_CORON','NRC_DARK']
        if self.params['Inst']['mode'].lower() == 'imaging':
            self.dark.meta.exposure.type = 'NRC_IMAGE'
        else:
            print('EXPTYPE mapping not complete for this!!! FIX ME!')
            sys.exit()

        #update various header keywords
        dims = self.dark.data.shape
        dtor = math.radians(1.)
        pixelsize = self.pixscale[0] / 3600.0

        current_time = datetime.datetime.utcnow()
        reference_time = datetime.datetime(2000,1,1,0,0,0)
        delta = current_time-reference_time
        et = delta.total_seconds()
        hmjd = 51544.5 + et/86400.0
        self.dark.meta.date = current_time.strftime('%Y-%m-%dT%H:%M:%S')
        self.dark.meta.telescope = 'JWST'

        self.dark.meta.instrument.name = self.params['Inst']['instrument'].upper()
        self.dark.meta.coordinates.reference_frame = 'ICRS'

        self.dark.meta.origin = 'STScI'
        self.dark.meta.filename = filename
        self.dark.meta.filetype = 'raw'
        self.dark.meta.observation.obs_id = '0'
        self.dark.meta.observation.visit_number = '0'
        self.dark.meta.observation.program = '0'
        self.dark.meta.observation.observation_number = '0'
        self.dark.meta.observation.visit_number = '0'
        self.dark.meta.observation.visit_group = '0'
        self.dark.meta.observation.sequence_id = '0'
        self.dark.meta.observation.activity_id ='0'
        self.dark.meta.observation.exposure_number = '0'
    
        self.dark.meta.program.pi_name = 'UNKNOWN'
        self.dark.meta.program.title = 'UNKNOWN'
        self.dark.meta.program.category = 'UNKNOWN'
        self.dark.meta.program.sub_category = 'UNKNOWN'
        self.dark.meta.program.science_category = 'UNKNOWN'
        self.dark.meta.program.continuation_id = 0

        self.dark.meta.target.catalog_name = 'UNKOWN'

        self.dark.meta.wcsinfo.wcsaxes = 2
        self.dark.meta.wcsinfo.crval1 = self.ra
        self.dark.meta.wcsinfo.crval2 = self.dec
        self.dark.meta.wcsinfo.crpix1 = self.refpix_pos['x']+1. 
        self.dark.meta.wcsinfo.crpix2 = self.refpix_pos['y']+1.
        self.dark.meta.wcsinfo.ctype1 = 'RA--TAN'
        self.dark.meta.wcsinfo.ctype2 = 'DEC--TAN'
        self.dark.meta.wcsinfo.cunit1 = 'deg' 
        self.dark.meta.wcsinfo.cunit2 = 'deg'


        #ra = self.params['Telescope']['ra']
        #colon = self.ra.find(':')
        #ranum = int(ra[0:colon]) + float(ra[colon+1:colon+3])/60. + float(ra[colon+4:])/3600.
        self.dark.meta.target.ra = self.ra
        #dec = self.parameters["dec"] 
        #colon = dec.find(':')
        #decnum = int(dec[0:colon]) + float(dec[colon+1:colon+3])/60. + float(dec[colon+4:])/3600.
        self.dark.meta.target.dec = self.dec

        ramptime = self.frametime*(2+self.params['Readout']['ngroup']*self.params['Readout']['nframe'])
        #Add time for the reset frame....
        rampexptime = self.frametime * (self.params['Readout']['ngroup']*self.params['Readout']['nframe'])

        # elapsed time from the end and from the start of the supposid ramp, in seconds
        # put the end of the ramp 1 second before the time the file is written
        # these only go in the fake ramp, not in the signal images....
        deltatend = datetime.timedelta(0,1.)
        deltatstart = datetime.timedelta(0,1.+ramptime)
        tend = current_time - deltatend
        tstart = current_time - deltatstart
        self.dark.meta.observation.date = tstart.strftime('%Y-%m-%d')
        str1 = str(int(tstart.microsecond/1000.))
        str2 = tstart.strftime('%H:%M:%S.')+str1
        self.dark.meta.observation.time = str2

        str1 = str(int(tend.microsecond/1000.))
        str2 = tend.strftime('%H:%M:%S.')+str1

        #Hmmm. Do we need a file with filter/pupil combos? Or should we rely
        #on the user to enter the correct values in the parameter file?
        if self.runStep['fwpw']:
            fwpw = ascii.read(self.params['Reffiles']['filtpupilcombo'])
        else:
            print("WARNING: Filter wheel element/pupil wheel element combo reffile not specified. Proceeding by")
            print("saving {} in FILTER keyword, and {} in PUPIL keyword".format(self.params['Readout']['filter'],self.params['Readout']['pupil']))
            fwpw = Table()
            fwpw['filter_wheel'] = self.params['Readout']['filter']
            fwpw['pupil_wheel'] = self.params['Readout']['pupil']

        #get the proper filter wheel and pupil wheel values for the header
        if self.params['Inst']['mode'].lower() == 'imaging':
            mtch = fwpw['filter'] == self.params['Readout']['filter'].upper()
            fw = str(fwpw['filter_wheel'].data[mtch][0])
            pw = str(fwpw['pupil_wheel'].data[mtch][0])

        #grism element
        if self.params['Inst']['mode'] == 'imaging':
            grism='N/A'
        else:
            grism=fw

        #since we are saving in DMS format, we can use the RampModel instance
        #that is self.dark, modify values, and save
        self.dark.meta.instrument.filter = fw
        self.dark.meta.instrument.pupil = pw
        self.dark.meta.instrument.grating = grism

        self.dark.meta.dither.primary_type = 'NONE'
        self.dark.meta.dither.position_number = 1
        self.dark.meta.dither.total_points = '1'
        self.dark.meta.dither.pattern_size = 0.0
        self.dark.meta.dither.subpixel_type = 'UNKNOWN'
        self.dark.meta.dither.subpixel_number = 1
        self.dark.meta.dither.subpixel_total_points = 1
        self.dark.meta.dither.xoffset = 0.0
        self.dark.meta.dither.yoffset = 0.0

        # pixel coordinates in FITS header start from 1 not from 0
        xc=(self.subarray_bounds[2]+self.subarray_bounds[0])/2.+1.
        yc=(self.subarray_bounds[3]+self.subarray_bounds[1])/2.+1.

        self.dark.meta.exposure.readpatt = self.params['Readout']['readpatt']

        #The subarray name needs to come from the "Name" column in the 
        #subarray definitions dictionary
        mtch = self.subdict["AperName"] == self.params["Readout"]['array_name']
        self.dark.meta.subarray.name = str(self.subdict["Name"].data[mtch][0])
        #self.dark.meta.subarray.name = self.params['Readout']['array_name']
        
        #subarray_bounds indexed to zero, but values in header should be
        #indexed to 1.
        self.dark.meta.subarray.xstart = self.subarray_bounds[0]+1
        self.dark.meta.subarray.ystart = self.subarray_bounds[1]+1
        self.dark.meta.subarray.xsize = self.subarray_bounds[2]-self.subarray_bounds[0]+1
        self.dark.meta.subarray.ysize = self.subarray_bounds[3]-self.subarray_bounds[1]+1

        nlrefpix = max(4-self.subarray_bounds[0],0)
        nbrefpix = max(4-self.subarray_bounds[1],0)
        nrrefpix=max(self.subarray_bounds[2]-(self.ffsize-4),0)
        ntrefpix=max(self.subarray_bounds[3]-(self.ffsize-4),0)

        self.dark.meta.exposure.nframes = self.params['Readout']['nframe']
        self.dark.meta.exposure.ngroups = self.params['Readout']['ngroup']
        self.dark.meta.exposure.nints = 1

        self.dark.meta.exposure.sample_time = 10
        self.dark.meta.exposure.frame_time = self.frametime 
        self.dark.meta.exposure.group_time = self.frametime*self.params['Readout']['nframe']
        self.dark.meta.exposure.groupgap = 0

        self.dark.meta.exposure.nresets_at_start = 2
        self.dark.meta.exposure.nresets_between_ints = 1
        self.dark.meta.exposure.integration_time = rampexptime
        self.dark.meta.exposure.exposure_time = rampexptime #assumes only 1 integration per exposure
        self.dark.meta.exposure.duration = ramptime

        self.dark.save(filename)
        print("Final output integration saved to {}".format(filename))
        return

    def readCRFiles(self):
        #read in the 10 files that comprise the cosmic ray library
        self.cosmicrays=[]
        self.cosmicraysheader=[]
        for i in range(10):
            str1='_%2.2d_IPC.fits' % (i)
            name = self.crfile + str1
            #name = self.params['cosmicRay']['path'] + self.fileList['cosmicrays'] + str1
            im,head = self.readCalFile(name)
            self.cosmicrays.append(im)
            self.cosmicraysheader.append(head)

    
    def frameToRamp(self,sigimage):
        #previously adjustSignals
        
        #Once we have a frame that contains the signal added from external sources, 
        #we need to adjust these signals to account for detector effects (poisson
        #noise, gain, and we need to add cosmic rays if requested. 
        #Using the base image, multiply by exposure time and generate the collection
        #of frames/groups that will form the final integration

        yd,xd = sigimage.shape
        
        #outframes will store all of the frames that will go into the output integration
        #averaging down frames into groups will happen in a later step
        #outallframes = np.zeros((self.params['Readout']['nframe']*self.params['Readout']['ngroup'],yd,xd))
        outimage = np.zeros((self.params['Readout']['ngroup'],yd,xd),dtype=np.float)
        totalsignalimage = np.zeros_like(sigimage,dtype=np.float)
        #oldsignalimage = np.zeros_like(sigimage,dtype=np.float)

        #read in gain map to be used below
        if self.runStep['gain']:
            gainim,gainhead = self.readCalFile(self.params['Reffiles']['gain'])
            #set any NaN's to 1.0
            bad = ~np.isfinite(gainim)
            gainim[bad] = 1.0

            #Pixels that have a gain value of 0 will be reset to have values of 1.0
            zeros = gainim == 0
            gainim[zeros] = 1.0

        #set up functions to apply cosmic rays later
        #Need the total number of active pixels in the output array to multiply the CR rate by
        if self.runStep['cosmicray']:
            npix = int(sigimage.shape[0]*sigimage.shape[1]+0.02)
            crhits,crs_perframe = self.CRfuncs(npix)

            #open output file to contain the list of cosmic rays
            crlistout = self.params['Output']['file'][0:-5] + '_cosmicrays.list'
            self.openCRListFile(crlistout,crhits)

            #counter for use in cosmic ray addition while looping over frames
            framenum = 0

        #difference between the latest outimage frame and the latest newsignalimage frame
        #This is important when nframe>1
        delta = 0.
        
        #container for zeroth frame
        zeroframe = None

        #Loop over each group
        for i in range(self.params['Readout']['ngroup']):
            accumimage = np.zeros_like(sigimage,dtype=np.int32)

            #Loop over frames within each group if necessary
            #create each frame
            for j in range(self.params['Readout']['nframe']+self.params['Readout']['nskip']):

                #add poisson noise 
                workimage = self.doPoisson(sigimage)

                #xxx=1800
                #yyy=800
                #print('after poisson',sigimage[yyy,xxx],workimage[yyy,xxx])

                #add cosmic rays
                if self.runStep['cosmicray']:
                    workimage = self.doCosmicRays(workimage,i,j,self.params['Readout']['nframe'],crs_perframe[framenum])
                    framenum = framenum + 1

                #print('after CR',workimage[yyy,xxx])

                #apply the inverse gain to go from electrons to ADU
                if self.runStep['gain']:
                    invgainimage = 1./gainim
                    workimage = workimage * invgainimage

                #Keep track of the total signal in the ramp, so that we don't neglect
                #signal which comes in during the frames that are skipped.
                totalsignalimage = totalsignalimage + workimage

                #print('after inv gain',workimage[yyy,xxx],totalsignalimage[yyy,xxx])


                # now round off and truncate to integers, simulating the A/D conversion
                # workimage is the signal accumulated in the current frame
                #NOTE: any NaN values will be translated into -2147483648
                workimage = np.around(workimage)
                workimage = workimage.astype(np.int32)
                    
                #if the frame is not one to be skipped (according to nskip)
                #then add it to the output group here
                if j >= self.params['Readout']['nskip']:

                    #if nframe is > 1, then we need to average the nframe frames and
                    #place that into the group.
                    #print('j={}'.format(j))
                    if self.params['Readout']['nframe'] != 1:
                        print('averaging frame {}'.format(i*(self.params['Readout']['nframe']+self.params['Readout']['nskip'])+j))
                        #accumimage = accumimage + workimage/self.params['Readout']['nframe']
                        accumimage = accumimage + ((self.params['Readout']['nframe']-(j-self.params['Readout']['nskip']))*1./self.params['Readout']['nframe'])*workimage
                    else:
                        #if nframe=1, then no averaging is necessary
                        #Is doing this faster than just using the averaging line above for nframe=1?
                        print('adding frame {}'.format(i*(self.params['Readout']['nframe']+self.params['Readout']['nskip'])+j))
                        accumimage = accumimage + workimage
                else:
                    #frames that are skipped, according to nskip. The signal from these frames 
                    #still must be included in the ramp, but not averaged into the group in the 
                    #case where nframe is more than 1.
                    print('skipping frame {}'.format(i*(self.params['Readout']['nframe']+self.params['Readout']['nskip'])+j))
                    accumimage = accumimage + workimage

                #print(workimage[yyy,xxx],accumimage[yyy,xxx])


                #if this is the zeroth frame of the integration, save for possible output
                if ((i == 0) & (j == 0) & (self.dark.meta.exposure.readpatt == 'RAPID')):
                    zeroframe = workimage 

            #Now put the signal group into the output cube
            accumimage = accumimage.astype(np.float)
            if i > 0:
                outimage[i,:,:] = outimage[i-1,:,:] + accumimage + delta
            else:
                outimage[i,:,:] = accumimage
            #print('final output signal: ',outimage[i,799,803])
            print("Group {} has been generated.".format(i+1))

            #Calculate the delta between outimage and the last iteration of newsignal. This needs to be
            #added to the next iteration of outimage. (This is important for cases where nframe>1)
            delta = totalsignalimage - outimage[i,:,:]
            #print('i = {}, delta = {}'.format(i+1,delta[799,803]))

        if self.runStep['cosmicray']:
            #close the cosmic ray list file
            self.cosmicraylist.close()

        return outimage,zeroframe


    def TSOframeToRamp(self,sigimage):
        #Once we have a frame that contains the signal added from external sources, 
        #we need to adjust these signals to account for detector effects (poisson
        #noise, gain, and we need to add cosmic rays if requested. For time series
        #observations, we need to shift and add signal in order to mimic the slew of
        #the telescope.
        #Using the base image, multiply by exposure time and generate the collection
        #of frames/groups that will form the final integration.
        yd,xd = sigimage.shape
    
        #outframes will store all of the frames that will go into the output integration
        #averaging down frames into groups will happen in a later step
        #outallframes = np.zeros((self.params['Readout']['nframe']*self.params['Readout']['ngroup'],yd,xd))
        outimage = np.zeros((self.params['Readout']['ngroup'],yd,xd),dtype=np.float)
        totalsignalimage = np.zeros_like(sigimage,dtype=np.float)
        #oldsignalimage = np.zeros_like(sigimage,dtype=np.float)

        #read in gain map to be used below
        if self.runStep['gain']:
            gainim,gainhead = self.readCalFile(self.params['Reffiles']['gain'])
            #set any NaN's to 1.0
            bad = ~np.isfinite(gainim)
            gainim[bad] = 1.0

            #Pixels that have a gain value of 0 will be reset to have values of 1.0
            zeros = gainim == 0
            gainim[zeros] = 1.0

        #set up functions to apply cosmic rays later
        #Need the total number of active pixels in the output array to multiply the CR rate by
        if self.runStep['cosmicray']:
            npix = int(sigimage.shape[0]*sigimage.shape[1]+0.02)
            crhits,crs_perframe = self.CRfuncs(npix)

            #open output file to contain the list of cosmic rays
            crlistout = self.params['Output']['file'][0:-5] + '_cosmicrays.list'
            self.openCRListFile(crlistout,crhits)

            #counter for use in cosmic ray addition while looping over frames
            framenum = 0

        #difference between the latest outimage frame and the latest newsignalimage frame
        #This is important when nframe>1
        delta = 0.
        
        #container for zeroth frame
        zeroframe = None

        #Loop over each group
        for i in range(self.params['Readout']['ngroup']):
            accumimage = np.zeros_like(sigimage,dtype=np.int32)

            #Loop over frames within each group if necessary
            #create each frame
            for j in range(self.params['Readout']['nframe']+self.params['Readout']['nskip']):

                #add poisson noise 
                workimage = self.doPoisson(sigimage)

                #xxx=1800
                #yyy=800
                #print('after poisson',sigimage[yyy,xxx],workimage[yyy,xxx])

                #add cosmic rays
                if self.runStep['cosmicray']:
                    workimage = self.doCosmicRays(workimage,i,j,self.params['Readout']['nframe'],crs_perframe[framenum])
                    framenum = framenum + 1

                #print('after CR',workimage[yyy,xxx])

                #apply the inverse gain to go from electrons to ADU
                if self.runStep['gain']:
                    invgainimage = 1./gainim
                    workimage = workimage * invgainimage

                #Keep track of the total signal in the ramp, so that we don't neglect
                #signal which comes in during the frames that are skipped.
                totalsignalimage = totalsignalimage + workimage

                #print('after inv gain',workimage[yyy,xxx],totalsignalimage[yyy,xxx])


                # now round off and truncate to integers, simulating the A/D conversion
                # workimage is the signal accumulated in the current frame
                #NOTE: any NaN values will be translated into -2147483648
                workimage = np.around(workimage)
                workimage = workimage.astype(np.int32)
                    
                #if the frame is not one to be skipped (according to nskip)
                #then add it to the output group here
                if j >= self.params['Readout']['nskip']:

                    #if nframe is > 1, then we need to average the nframe frames and
                    #place that into the group.
                    #print('j={}'.format(j))
                    if self.params['Readout']['nframe'] != 1:
                        print('averaging frame {}'.format(i*(self.params['Readout']['nframe']+self.params['Readout']['nskip'])+j))
                        #accumimage = accumimage + workimage/self.params['Readout']['nframe']
                        accumimage = accumimage + ((self.params['Readout']['nframe']-(j-self.params['Readout']['nskip']))*1./self.params['Readout']['nframe'])*workimage
                    else:
                        #if nframe=1, then no averaging is necessary
                        #Is doing this faster than just using the averaging line above for nframe=1?
                        print('adding frame {}'.format(i*(self.params['Readout']['nframe']+self.params['Readout']['nskip'])+j))
                        accumimage = accumimage + workimage
                else:
                    #frames that are skipped, according to nskip. The signal from these frames 
                    #still must be included in the ramp, but not averaged into the group in the 
                    #case where nframe is more than 1.
                    print('skipping frame {}'.format(i*(self.params['Readout']['nframe']+self.params['Readout']['nskip'])+j))
                    accumimage = accumimage + workimage

                #print(workimage[yyy,xxx],accumimage[yyy,xxx])


                #if this is the zeroth frame of the integration, save for possible output
                if ((i == 0) & (j == 0) & (self.dark.meta.exposure.readpatt == 'RAPID')):
                    zeroframe = workimage 

            #Now put the signal group into the output cube
            accumimage = accumimage.astype(np.float)
            if i > 0:
                outimage[i,:,:] = outimage[i-1,:,:] + accumimage + delta
            else:
                outimage[i,:,:] = accumimage
            #print('final output signal: ',outimage[i,799,803])
            print("Group {} has been generated.".format(i+1))

            #Calculate the delta between outimage and the last iteration of newsignal. This needs to be
            #added to the next iteration of outimage. (This is important for cases where nframe>1)
            delta = totalsignalimage - outimage[i,:,:]
            #print('i = {}, delta = {}'.format(i+1,delta[799,803]))

        if self.runStep['cosmicray']:
            #close the cosmic ray list file
            self.cosmicraylist.close()

        return outimage,zeroframe



    def openCRListFile(self,filename,hits):
        #open a file and print header info for the file that will contain
        #the list and positions of inserted cosmic rays
        self.cosmicraylist = open(filename,"w")
        self.cosmicraylist.write("# Cosmic ray list (file set %s random seed %d)\n" % (self.crfile,self.params['cosmicRay']['seed']))
        self.cosmicraylist.write('# Cosmic ray rate per frame: %13.6e (scale factor %f)\n' % (hits,self.params['cosmicRay']['scale']))
        self.cosmicraylist.write('Image_x    Image_y    Group   Frame   CR_File_Index   CR_file_frame   Max_CR_Signal\n')


    def addSyntheticToDark(self,synthetic,dark,syn_zeroframe=None):
        #ASSUMES INPUT DARK IS A RAMPMODEL INSTANCE

        #if zeroframe is provided, the function uses that to create the
        #dark+synthetic zeroframe that is returned. If not provided, the
        #function attempts to use the 0th frame of the input synthetic ramp
        
        #Combine the cube of synthetic signals to the real dark current ramp.
        #Be sure to adjust the dark current ramp if nframe/nskip is different
        #than the nframe/nskip values that the dark was taken with.

        #Only RAPID darks will be re-averaged into different readout patterns
        #But a BRIGHT2 dark can be used to create a BRIGHT2 simulated ramp

        #Get the info for the dark integration
        darkpatt = dark.meta.exposure.readpatt
        dark_nframe = dark.meta.exposure.nframes
        mtch = self.readpatterns['name'].data == darkpatt
        dark_nskip = self.readpatterns['nskip'].data[mtch][0]

        #We can only keep a zero frame around if the input dark
        #is RAPID. Otherwise that information is lost.
        zeroframe = None
        if (darkpatt == 'RAPID'):
            darkzero = dark.data[0,0,:,:]
            if syn_zeroframe is not None:
                zeroframe = darkzero + syn_zeroframe
            #else:
            #    zeroframe = darkzero + synthetic[0,:,:]
        else:
            print("Unable to save the zeroth frame because the input dark current ramp is not RAPID.")
            zeroframe = None
            
        #We have already guaranteed that either the readpatterns match
        #or the dark is RAPID, so no need to worry about checking for 
        #other cases here.

        #if readpatt of dark  == RAPID and readpatt of output != RAPID: then averaging.
        #else, no averaging

        #print('IN ADDSYNTHETICTODARK, DARKPATTERN IS {} AND REQ PATTERN IS {}'.format(darkpatt,self.params['Readout']['readpatt']))
        
        if ((darkpatt == 'RAPID') and (self.params['Readout']['readpatt'] != 'RAPID')): 

            deltaframe = self.params['Readout']['nskip']+self.params['Readout']['nframe']
            frames = np.arange(self.params['Readout']['nskip'],deltaframe)
            accumimage = np.zeros_like(synthetic[0,:,:],dtype=np.int32)

            #Loop over groups
            for i in range(self.params['Readout']['ngroup']):
                #average together the appropriate frames, skipe the appropriate frames
                print('Averaging dark current ramp. Frames {}, to become group {}'.format(frames,i))

                #If averaging needs to be done
                if self.params['Readout']['nframe'] > 1:
                    accumimage = np.mean(dark.data[0,frames,:,:],axis=0)
                    errimage = np.mean(dark.err[0,frames,:,:],axis=0)
                    gdqimage = dark.groupdq[0,frames[-1],:,:]

                    #If no averaging needs to be done
                else:
                    accumimage = dark.data[0,frames[0],:,:]
                    errimage = dark.err[0,frames[0],:,:]
                    gdqimage = dark.groupdq[0,frames[0],:,:]

                #now add the averaged dark frame to the synthetic data, which has already been
                #placed into the correct readout pattern
                synthetic[i,:,:] += accumimage
                
                #increment the frame indexes
                frames = frames + deltaframe

        else:
            #if the input dark is not RAPID, or if the readout pattern of the input dark and
            #the output ramp match, then no averaging needs to be done and we can simply add
            #the synthetic groups to the dark current groups.
            synthetic = synthetic + dark.data[0,0:self.params['Readout']['ngroup'],:,:]

        return synthetic,zeroframe



    def reorderDark(self,dark):
        #ASSUMES INPUT DARK IS A RAMPMODEL INSTANCE

        #Reorder the input dark ramp using the requested readout pattern (nframe,nskip).
        #If the initial dark ramp is RAPID, then save and return the 0th frame.

        #Get the info for the dark integration
        darkpatt = dark.meta.exposure.readpatt
        dark_nframe = dark.meta.exposure.nframes
        mtch = self.readpatterns['name'].data == darkpatt
        dark_nskip = self.readpatterns['nskip'].data[mtch][0]

        nint,ngroup,yd,xd = dark.data.shape
        outdark = np.zeros((self.params['Readout']['ngroup'],yd,xd))
        outerr = np.zeros((self.params['Readout']['ngroup'],yd,xd))
        outgdq = np.zeros((self.params['Readout']['ngroup'],yd,xd))
  
        #We can only keep a zero frame around if the input dark
        #is RAPID. Otherwise that information is lost.
        darkzero = None
        if (darkpatt == 'RAPID'):
            darkzero = dark.data[0,0,:,:]
        else:
            print("Unable to save the zeroth frame because the input dark current ramp is not RAPID.")
            
        #We have already guaranteed that either the readpatterns match
        #or the dark is RAPID, so no need to worry about checking for 
        #other cases here.

        if ((darkpatt == 'RAPID') and (self.params['Readout']['readpatt'] != 'RAPID')): 

            deltaframe = self.params['Readout']['nskip']+self.params['Readout']['nframe']
            frames = np.arange(self.params['Readout']['nskip'],deltaframe)
            accumimage = np.zeros_like(outdark[0,:,:],dtype=np.int32)

            #Loop over groups
            for i in range(self.params['Readout']['ngroup']):
                #average together the appropriate frames, skipe the appropriate frames
                print('Averaging dark current ramp. Frames {}, to become group {}'.format(frames,i))

                #If averaging needs to be done
                if self.params['Readout']['nframe'] > 1:
                    accumimage = np.mean(dark.data[0,frames,:,:],axis=0)
                    errimage = np.mean(dark.err[0,frames,:,:],axis=0)
                    gdqimage = dark.groupdq[0,frames[-1],:,:]

                    #If no averaging needs to be done
                else:
                    accumimage = dark.data[0,frames[0],:,:]
                    errimage = dark.err[0,frames[0],:,:]
                    gdqimage = dark.groupdq[0,frames[0],:,:]

                #now add the averaged dark frame to the synthetic data, which has already been
                #placed into the correct readout pattern
                outdark[i,:,:] += accumimage
                outerr[i,:,:] += errimage
                outgdq[i,:,:] = gdqimage

                #increment the frame indexes
                frames = frames + deltaframe

        elif (self.params['Readout']['readpatt'] == darkpatt):
            #if the input dark is not RAPID, or if the readout pattern of the input dark and
            #the output ramp match, then no averaging needs to be done and we can simply add
            #the synthetic groups to the dark current groups.
            outdark = dark.data[0,0:self.params['Readout']['ngroup'],:,:]
            outerr = dark.err[0,0:self.params['Readout']['ngroup'],:,:]
            outgdq = dark.groupdq[0,0:self.params['Readout']['ngroup'],:,:]
        else:
            #This check should already have been done, but just to be sure...
            print("WARNING: dark current readout pattern is {} and requested output is {}.".format(darkpatt,self.params['Readout']['readpatt']))
            print("Cannot convert between the two.")
            sys.exit()


        #Now place the reorganized dark into the model instance and update the appropriate metadata
        dark.data = np.expand_dims(outdark,axis=0)
        dark.meta.exposure.readpatt = self.params['Readout']['readpatt'] 
        dark.meta.exposure.nframes = self.params['Readout']['nframe']
        dark.meta.exposure.nskip = self.params['Readout']['nskip']
        dark.err = np.expand_dims(outerr,axis=0)
        dark.groupdq = np.expand_dims(outgdq,axis=0)
        
        return dark,darkzero


    def doNonLin(self,image,coeffs,sat):
        #insert non-linearity into the linear synthetic sources

        #if the sizes of the satmap or coeffs are different than the data, return an error
        #This shouldn't happen since all reference files are being opened by readCalFile
        if sat.shape != image.shape[-2:]:
            print("WARNING: in doNonLin, input image shape is {}, but input saturation map shape is {}".format(image.shape[-2:],sat.shape))
            sys.exit()


        #sat is the original saturation map data for non-linear ramps. Translate to saturation maps
        #for the linear ramps here so that we can pay attention only to non-saturated pixels in
        #the input linear image
        lin_satmap = self.nonLinFunc(sat,coeffs,sat)

        #xx=803
        #yy=799
        #print('pixel {},{}, satmap {}, lin_satmap {}'.format(xx,yy,sat[yy,xx],lin_satmap[yy,xx]))

        
        #find pixels with "good" signals, to have the nonlin applied. Negative pix or pix with
        #signals above the requested max value will not be changed.
        x = np.copy(image)
        i1 = np.where((image > 0.) & (image < lin_satmap))
        dev = np.zeros_like(image,dtype=float)
        dev[i1] = 1.
        i2 = np.where((image <= 0.) | (image >= lin_satmap))
        numhigh = np.where(image >= lin_satmap)
        #print('Number of saturated pixels: {}'.format(len(numhigh[0])))
        i = 0

        #initial run of the nonlin function - when calling the non-lin function, give the
        #original satmap for the non-linear signal values
        val = self.nonLinFunc(image,coeffs,sat)
        val[i2]=1.


        #print inputs to see what's going on
        #xxx=500
        #yyy=500
        #print('Initially: lin img: {}, val {}'.format(x[xxx,yyy],val[xxx,yyy]))
        
        if self.params['nonlin']['robberto']:            
            x = image * val
        else:
            #y1 = np.copy(image)
            x[i1] = (image[i1]+image[i1]/val[i1]) / 2. #I added the i1 used here...


            #print('y1: {}, x: {}'.format(y1[xxx,yyy],x[xxx,yyy]))
            
            #print('Huge signal values are coming from the solution not converging')
            #print('We need to adjust the initial guess, like Kevin suggested')
            while i < self.params['nonlin']['maxiter']:
                i=i+1
                #print('i is {}. x[500,500] is {}'.format(i,x[500,500]))
                #if i == self.params['nonlin']['maxiter']:
                    #print('max iterations reached, but {} pix have not met accuracy goal.'.format(len(inds[0])))
                    #print(inds)
                    #tmp=image[i1]
                    #print(tmp[inds])
                    #tmp2=lin_satmap[i1]
                    #print(tmp2[inds])
                    #tmp3=sat[i1]
                    #print(tmp3[inds])
                    #tmp4=dev[i1]
                    #print(tmp4[inds])
                    #sys.exit()
                val = self.nonLinFunc(x,coeffs,sat)
                val[i2]=1.
                #print('i is {}, val: {}'.format(i,val[xxx,yyy]))
                #y1 = x * val
                #print('y1: {}'.format(y1[xxx,yyy]))
                #dev[i1] = abs(image[i1]/y1[i1]-1.)
                dev[i1] = abs(image[i1]/val[i1]-1.)
                inds = np.where(dev[i1] > self.params['nonlin']['accuracy'])
                #print('i={}. Number of pixels not below accuracy limit {}'.format(i,inds[0].size))
                if inds[0].size < 1:
                    break
                val1 = self.nonLinDeriv(x,coeffs,sat)
                val1[i2] = 1.
                #x[i1] = x[i1] + (image[i1]-y1[i1])/val1[i1]
                x[i1] = x[i1] + (image[i1]-val[i1])/val1[i1]
                #print('dev {}, val1: {}, x: {}'.format(dev[xxx,yyy],val1[xxx,yyy],x[xxx,yyy]))


                #if we max out the number of iterations, save the array of accuracy values
                #Spot checks reveal the pix that don't meet the accuracy reqs are randomly
                #located on the detector, and don't seem to be correlated with point source
                #locations.
                if i == self.params['nonlin']['maxiter']:
                    ofile = self.params['Output']['file'][0:-5] + '_doNonLin_accuracy.fits'
                    devcheck = np.copy(dev)
                    devcheck[i2] = -1.
                    h0 = fits.PrimaryHDU()
                    h1 = fits.ImageHDU(devcheck)
                    hl = fits.HDUList([h0,h1])
                    hl.writeto(ofile,clobber=True)
                
            #test output accuracy
            #testout = self.nonLinFunc(x,coeffs,sat)
            #diff = image-testout
            #print(np.nanmean(diff[i1]),np.nanmax(diff[i1]),np.nanmin(diff[i1]),np.nanmean(diff[i1]/image[i1]))
            #print('pixel {},{}, added non-lin signal: {}'.format(xx,yy,x[yy,xx]))


            #Check for pixels with nonsense values that come from Newton's method not converging.
            #Set these pixels equal to zero.
            #bad = ((x < -100) | (x > 1e5))
            #numbad = np.sum(bad)
            #print("In doNonLin {} pixels failed to converge, and will be set to zero.".format(numbad))
            #x[bad] = 0.

            return x

    def removeNonLin(self,data,coeffs):

        #CURRENTLY NOT USED
        
        #remove non-linearity from the input data
        #this works in the same direction as the SSB pipeline's linearity step
        dims = data.shape
        coeff_dims = coeffs.shape[0]

        outdata = np.zeros_like(data)
        if len(dims) == 2:

            if self.params['nonlin']['robberto'] == False:
                #THIS MATCHES THE JWST NON-LIN CORRECTION STEP
                #for i in xrange(dims[0]):
                #    for j in xrange(dims[1]):
                #        val = 0
                #        for m in xrange(coeff_dims):
                #            val = val + coeffs[m,i,j]*data[i,j]**m
                #        outdata[i,j] = val

                #if the user gave a saturation map, then use that as the non-linearity limit
                #if self.runStep['saturation_lin_limit']:
                #    satmap,sathead = self.readCalFile(self.params['Reffiles']['saturation'])
                
                values = np.copy(image)
                print('adjust values line below for for nonlin max map')
                sys.exit()
                values[values > self.params['nonlin']['limit']] = self.params['nonlin']['limit'] 
                ncoeff = coeffs.shape[0]
                t = np.copy(coeffs[-1,:,:])
                for i in range(ncoeff-2,-1,-1):
                    t = coeffs[i,:,:] + values*t

            else:
                for i in xrange(dims[0]):
                    for j in xrange(dims[1]):
                        b = coeffs[1,i,j]
                        val = coeffs[0,i,j] + b*data[i,j]
                        for m in xrange(2,coeff_dims):
                            val = val + coeffs[m,i,j]*(b*data[i,j])**m
                        outdata[i,j] = val

        return outdata


#CHECK THIS FUNCTION AND THE DERIV FUNCTION BELOW. DO I NEED TO EDIT TO MATCH JWST NLIN REFERENCE FILE FORMAT? SEE COMMENT ABOVE
    def orignonLinFunc(self,image,coeffs):
        values = np.copy(image)
        values[values > self.params['nonlin']['limit']] = self.params['nonlin']['limit'] #- adjust this for nonlin max map
        ncoeff = coeffs.shape[0]
        t = np.copy(coeffs[-1,:,:])
        for i in range(ncoeff-2,-1,-1):
            t = coeffs[i,:,:] + values*t
        t=1. + values*t
        return t

    def nonLinFunc(self,image,coeffs,limits):
        #adjust definition so it lines up with the JWST definition
        #by commenting out the last line (t=1+values*t), I think
        #we get it.
        values = np.copy(image)
        #values[values > self.params['nonlin']['limit']] = self.params['nonlin']['limit'] - adjust this for nonlin max map

        bady = 0
        badx = 1
        if len(image.shape) == 3:
            bady = 1
            badx = 2

        bad = np.where(values > limits)
        values[bad] = limits[bad[bady],bad[badx]]
        ncoeff = coeffs.shape[0]
        t = np.copy(coeffs[-1,:,:])
        for i in range(ncoeff-2,-1,-1):
            t = coeffs[i,:,:] + values*t
        return t


    def orignonLinDeriv(self,image,coeffs):
        values = np.copy(image)
        values[values > self.params['nonlin']['limit']]=self.params['nonlin']['limit']
        ncoeff = coeffs.shape[0]
        t = (ncoeff+1) * np.copy(coeffs[-1,:,:])
        for i in range(ncoeff-2,-1,-1):
            t = (i+2) * coeffs[i,:,:] + values*t
        t = 1. + values*t
        return t

    def nonLinDeriv(self,image,coeffs,limits):
        values = np.copy(image)

        bady = 0
        badx = 1
        if len(image.shape) == 3:
            bady = 1
            badx = 2
        
        #values[values > self.params['nonlin']['limit']]=self.params['nonlin']['limit']
        bad = np.where(values > limits)
        values[bad] = limits[bad[bady],bad[badx]]
        ncoeff = coeffs.shape[0]
        t = (ncoeff-1) * np.copy(coeffs[-1,:,:])
        for i in range(ncoeff-3,-1,-1):
            t = (i+1) * coeffs[i+1,:,:] + values*t
        #t = coeffs[0,:,:] + values*t
        return t


    def doPoisson(self,signalimage):
        #add poisson noise to an input image
        newimage=np.zeros_like(signalimage,dtype=np.float)
        ndim=signalimage.shape

        #Find the appropriate quantum yield value for the filter
        if self.params['simSignals']['photonyield']:
            try:
                pym1=self.qydict[self.params['Readout']['filter']] - 1.
            except:
                pym1=0.

        #Add poisson noise to each pixel
        for i in range(ndim[0]):
            for j in range(ndim[1]):
                try:
                    newimage[i,j]=np.random.poisson(signalimage[i,j])
                except:
                    print("Error: bad signal value at pixel (x,y)=({},{}) = {}".format(j,i,signalimage[i,j]))
                    sys.exit()

                if self.params['simSignals']['photonyield'] and pym1 > 0.000001 and newimage[i,j] > 0:
                    if self.params['simSignals']['pymethod']:
                        # calculate the values to make the poisson results the same with/without photon 
                        # yield (but not for pymethod true and false)...use yield -1 because the value 
                        # cannot be less than 1
                        values = np.random.poisson(pym1,newimage[i,j])
                        newimage[i,j] = newimage[i,j] + values.sum()
                    else:
                        newimage[i,j] = newimage[i,j] * self.qydict[self.params['Readout']['filter']]
                        fract = newimage[i,j] - int(newimage[i,j])
                        if self.generator2.random() < fract:
                            newimage[i,j] = newimage[i,j] + 1
        return newimage

    
    def doCosmicRays(self,image,ngroup,iframe,nframe,ncr):
        #add cosmic rays to a frame
        nray = int(ncr)
        #if nray > 10000:
        #    self.cosmicraylist.write("# Warning: number of cosmic rays = %d, more than the size of the library.\n" % nray)

        i=0
        dims=image.shape
        while i < nray:
            i=i+1
            j=int(self.generator1.random()*dims[0])
            k=int(self.generator1.random()*dims[1])
            n=int(self.generator1.random()*10.0)
            m=int(self.generator1.random()*1000.0)
            crimage=np.copy(self.cosmicrays[n][m,:,:])
            i1=max(j-10,0)
            i2=min(j+11,dims[0])
            j1=max(k-10,0)
            j2=min(k+11,dims[1])
            k1=10-(j-i1)
            k2=10+(i2-j)
            l1=10-(k-j1)
            l2=10+(j2-k)

            image[i1:i2,j1:j2]=image[i1:i2,j1:j2]+crimage[k1:k2,l1:l2]

            #if nframe > 1:
            #    self.cosmicraylist.write("%4d %4d %4d %1d %3s %1d %4d %15.1f\n" % (k+1,j+1,ngroup,iframe,'   ',n,m+1,np.max(crimage)))
            #else:
            #    self.cosmicraylist.write("%4d %4d %4d %5s     %1d %4d %15.1f\n" % (k+1,j+1,ngroup,'     ',n,m+1,np.max(crimage)))

            self.cosmicraylist.write("{} {} {} {} {} {} {}\n".format((j2-j1)/2+j1,(i2-i1)/2+i1,ngroup,iframe,n,m,np.max(crimage[k1:k2,l1:l2])))
        return image


    #def dumpDicts(self):
    #    print("self.runPars:")
    #    for key in self.runPars.keys():
    #        print("  %s = %s" % (key,self.runPars[key]))
    #    print("self.fileList:")
    #    for key in self.fileList.keys():
    #        print("  %s = %s" % (key,self.fileList[key]))
    #    print("self.flags:")
    #    for key in self.flags.keys():
    #        print("  %s = %s" % (key,self.flags[key]))
    #    print("\nInput parameters:")
    #    for i in range(len(self.parameters)):
    #        print("%3d %s" % (i,self.parameters[i]))



    def getBaseDark(self):
        #read in the dark current ramp that will serve as the
        #base for the simulated ramp
        self.dark = RampModel(self.params['Reffiles']['dark'])

        #We assume that the input dark current integration is raw, which means
        #the data are in the original ADU measured by the detector. So the
        #data range should be no larger than 65536
        darkrange = self.dark.data.max() - self.dark.data.min()
        if darkrange > 65535.:
            print("WARNING: Range of data values in the input dark is too large.")
            print("We assume the input dark is raw ADU values, with a range no more than 65536.")
            sys.exit()

        #If the inputs are signed integers, change to unsigned.
        if self.dark.data.min() < 0.:
            self.dark.data += 32768

        #If the input is any readout pattern other than RAPID, then
        #make sure that the output readout patten matches. Only RAPID
        #can be averaged and transformed into another readout pattern
        if self.dark.meta.exposure.readpatt != 'RAPID':
            if self.params['Readout']['readpatt'].upper() != self.dark.meta.exposure.readpatt:
                print("WARNING: cannot transform input {} integration into output {} integration. Only RAPID inputs can be translated to a different readout pattern".format(self.dark.meta.exposure.readpatt,self.params['Readout']['readpatt']))
                sys.exit()

        #Finally, collect information about the detector, which will be needed for astrometry later
        self.detector = self.dark.meta.instrument.detector
        self.instrument = self.dark.meta.instrument.name

    def cropDark(self,model):
        #cut the dark current array down to the size dictated by the
        #subarray bounds
        nint,ngrp,yd,xd = model.data.shape
        if ((self.subarray_bounds[0] != 0) or (self.subarray_bounds[2] != (xd-1)) or (self.subarray_bounds[1] != 0) or (self.subarray_bounds[3] != (yd-1))):
            print("Information: a full frame dark ramp was provided as input, but a subarray was specified as output. Extracting the appropriate sub-array area.")

            model.data = model.data[0:1,:,self.subarray_bounds[1]:self.subarray_bounds[3]+1,self.subarray_bounds[0]:self.subarray_bounds[2]+1]
            model.err = model.err[0:1,:,self.subarray_bounds[1]:self.subarray_bounds[3]+1,self.subarray_bounds[0]:self.subarray_bounds[2]+1]
            model.pixeldq = model.pixeldq[self.subarray_bounds[1]:self.subarray_bounds[3]+1,self.subarray_bounds[0]:self.subarray_bounds[2]+1]
            model.groupdq = model.groupdq[0:1,:,self.subarray_bounds[1]:self.subarray_bounds[3]+1,self.subarray_bounds[0]:self.subarray_bounds[2]+1]

        #make sure that if the output is supposedly a 4-amplifier file, that the number of
        #pixels in the x direction is a multiple of 4.
        nfast=self.subarray_bounds[2]-self.subarray_bounds[0]+1
        nslow=self.subarray_bounds[3]-self.subarray_bounds[1]+1
        nramp=(self.params['Readout']['nframe']+self.params['Readout']['nskip'])*self.params['Readout']['ngroup']

        if self.params['Readout']['namp'] != 4 and self.params['Readout']['namp'] != 1:
            print('ERROR: amplifier mode specified ({}) is not allowed'.format(self.params['Readout']['namp']))
            sys.exit()

        if self.params['Readout']['namp'] == 4:
            n=int(nfast/4)*4
            if n != nfast:
                print('ERROR: 4 amplifier mode specified but the number of pixels in the fast\nread direction ({}) is not a multiple of 4.'.format(nfast))
                sys.exit()

        return model


    def dataVolumeCheck(self):
        #make sure that the input integration has enough frames/groups to create the requested
        #number of frames/groups of the output
        ngroup = int(self.params['Readout']['ngroup'])
        nframe = int(self.params['Readout']['nframe'])
        nskip = int(self.params['Readout']['nskip'])

        inputframes = self.dark.data.shape[1]
        if ngroup*(nskip+nframe) > inputframes:
            print("WARNING: Not enough frames in the input integration to create the requested number of output groups. Input has {} frames. Requested output is {} groups each created from {} frames plus skipping {} frames between groups.".format(inputframes,ngroup,nframe,nskip))
            print("for a total of {} frames.".format(ngroup*(nframe+nskip)))
            print("Making copies of {} dark current frames and adding them to the end of the dark current integration.".format(ngroup*(nskip+nframe) - inputframes))
            #sys.exit()

            #figure out how many more frames we need, in terms of how many copies of hte original dark
            div = (ngroup*(nskip+nframe)) / inputframes
            mod = (ngroup*(nskip+nframe)) % inputframes
            
            #if more frames are needed than there are frames in the original dark, then make copies of the entire
            #thing as many times as necessary, adding the signal from the previous final frame to each.
            for ints in xrange(div-1):
                extra_frames = np.copy(self.dark.data[0,:,:,:])
                extra_frames = np.expand_dims(extra_frames,axis=0)
                extra_err_frames = np.copy(self.dark.err[0,:,:,:])
                extra_err_frames = np.expand_dims(extra_err_frames,axis=0)
                extra_dq_frames = np.copy(self.dark.groupdq[0,:,:,:])
                extra_dq_frames = np.expand_dims(extra_dq_frames,axis=0)
                self.dark.data = np.hstack((self.dark.data,extra_frames+self.dark.data[0,-1,:,:]))
                self.dark.err = np.hstack((self.dark.err,extra_err_frames+self.dark.err[0,-1,:,:]))
                self.dark.groupdq = np.hstack((self.dark.groupdq,extra_dq_frames+self.dark.groupdq[0,-1,:,:]))
            
            #At this point, if more frames are needed, but fewer than an entire copy of self.dark.data,
            #then add the appropriate number of frames here.
            extra_frames = np.copy(self.dark.data[0,1:mod+1,:,:]) - self.dark.data[0,0,:,:]
            extra_frames = np.expand_dims(extra_frames,axis=0)
            extra_err_frames = np.sqrt(np.copy(self.dark.err[0,1:mod+1,:,:])**2 + self.dark.err[0,-1,:,:]**2)
            extra_err_frames = np.expand_dims(extra_err_frames,axis=0)
            extra_dq_frames = np.copy(self.dark.groupdq[0,1:mod+1,:,:])
            extra_dq_frames = np.expand_dims(extra_dq_frames,axis=0)
            self.dark.data = np.hstack((self.dark.data,extra_frames+self.dark.data[0,-1,:,:]))
            self.dark.err = np.hstack((self.dark.err,extra_err_frames))
            self.dark.groupdq = np.hstack((self.dark.groupdq,extra_dq_frames+self.dark.groupdq[0,-1,:,:]))

    def CRfuncs(self,npix):
        #set up functions that will be used to generate cosmic ray hits
        crhits = npix * self.crrate * self.params['cosmicRay']['scale'] * self.frametime
        np.random.seed(self.params['cosmicRay']['seed'])
        self.generator1 = random.Random()
        self.generator1.seed(self.params['cosmicRay']['seed'])
        #Need a set of CRs for all frames, including those that are skipped, in order for the rate of CRs to be consistent.
        crs_perframe = np.random.poisson(crhits,self.params['Readout']['ngroup'] * (self.params['Readout']['nframe']+self.params['Readout']['nskip']))
        np.random.seed(self.params['simSignals']['poissonseed'])
        self.generator2 = random.Random()
        self.generator2.seed(self.params['simSignals']['poissonseed'])
        return crhits,crs_perframe 


    def makeFilterTable(self):
        #Create the table that contains the possible filter list, quantum yields, and countrates for a
        #star with vega magnitude of 15 in each filter. Do this by reading in phot_file
        #listed in the parameter file. 

        #FUTURE WORK: If the countrates are left as 0, then pysynphot will
        #be used to calculate them later
        try:
            cvals_tab = ascii.read(self.params['Reffiles']['phot'])
            instrumentfilternames = cvals_tab['filter'].data
            stringcountrates = cvals_tab['countrate_for_vegamag15'].data
            instrumentmag15countrates = [float(s) for s in stringcountrates]
            strinstrumentqy = cvals_tab['quantum_yield'].data
            qy = [float(s) for s in strinstrumentqy]
            self.countvalues=dict(zip(instrumentfilternames,instrumentmag15countrates))
            self.qydict=dict(zip(instrumentfilternames,qy))

        except:
            print("WARNING: Unable to read in {}.".format(self.params['Reffiles']['phot']))
            sys.exit()


    def readCalFile(self,filename):
        #read in the specified calibration file
        try:
            with fits.open(filename) as h:
                image = h[1].data
                header = h[0].header
        except:
            print("WARNING: Unable to open {}")
            sys.exit()

        #extract the appropriate subarray if necessary
        if ((self.subarray_bounds[0] != 0) or (self.subarray_bounds[2] != (self.ffsize-1)) or (self.subarray_bounds[1] != 0) or (self.subarray_bounds[3] != (self.ffsize-1))):

            if len(image.shape) == 2:
                image = image[self.subarray_bounds[1]:self.subarray_bounds[3]+1,self.subarray_bounds[0]:self.subarray_bounds[2]+1]

            if len(image.shape) == 3:
                image = image[:,self.subarray_bounds[1]:self.subarray_bounds[3]+1,self.subarray_bounds[0]:self.subarray_bounds[2]+1]

        return image,header


    def readCRFile(self,filename):
        #read in a CR library file
        try:
            with fits.open(filename) as h:
                image = h[1].data
                header = h[0].header
        except:
            print("WARNING: Unable to open {}")
            sys.exit

        return image,header


    def addedSignals(self):
        #generate a signal rate image from input sources
        signalimage = np.zeros_like(self.dark.data[0,0,:,:],dtype=np.float) 
        yd,xd = signalimage.shape

        if self.params['Output']['grism_source_image'] or self.params['Inst']['mode'] == 'tso':
            signalimage = np.zeros((np.int(yd*self.coord_adjust['y']),np.int(xd*self.coord_adjust['x'])),dtype=np.float)
        
        #original version
        #if self.params['Output']['grism_source_image'] == True:
        #    signalimage = np.zeros((np.int(yd*self.grism_direct_factor),np.int(xd*self.grism_direct_factor)),dtype=np.float)

        arrayshape = signalimage.shape

        #MASK IMAGE
        #Create a mask so that we don't add signal to masked pixels
        #Initially this includes only the reference pixels
        #Keep the mask image equal to the true subarray size, since this 
        #won't be used to make a requested grism source image
        maskimage = np.zeros((self.ffsize,self.ffsize),dtype=np.int)
        maskimage[4:self.ffsize-4,4:self.ffsize-4] = 1.
        
        #crop the mask to match the requested output array
        if self.params['Readout']['array_name'] != "FULL":
            maskimage = maskimage[self.subarray_bounds[1]:self.subarray_bounds[3]+1,self.subarray_bounds[0]:self.subarray_bounds[2]+1]


        #get the name of the PSF file that will be used for convolving with sources other than the point sources
        centerpsffile = self.psfname + '_0p0_0p0.fits'
        centerpsf = fits.getdata(centerpsffile)

        #crop the psf such that it is centered in its array
        centerpsf = self.cropPSF(centerpsf)

        #normalize the PSF to a total signal of 1.0
        totalsignal = np.sum(centerpsf)
        centerpsf = centerpsf / totalsignal
        
        #POINT SOURCES
        #Read in the list of point sources to add
        #Adjust point source locations using astrometric distortion
        #Translate magnitudes to counts in a single frame
        if self.runStep['pointsource'] == True:
            pslist = self.getPointSourceList()

            #translate the point source list into an image
            psfimage = self.makePointSourceImage(pslist)
            
            #save the point source image for examination by user
            if self.params['Output']['save_intermediates'] == True:
                psfImageName = self.params['Output']['file'][0:-5] + '_pointSourceRateImage_elec_per_sec.fits'
                self.saveSingleFits(psfimage,psfImageName)
                print("Point source image saved as {}".format(psfImageName))
                
            #Add the point source image to the overall image
            signalimage = signalimage + psfimage
            #print('after point sources',np.max(signalimage),np.min(signalimage))

        #Simulated galaxies
        #Read in the list of galaxy positions/magnitudes to simulate
        #and create a countrate image of those galaxies.
        if self.runStep['galaxies'] == True:
            galaxyCRImage = self.makeGalaxyImage(centerpsf)

            #save the galaxy image for examination by the user
            if self.params['Output']['save_intermediates'] == True:
                galImageName = self.params['Output']['file'][0:-5] + '_galaxyRateImage_elec_per_sec.fits'
                self.saveSingleFits(galaxyCRImage,galImageName)
                print("Simulated galaxy image saved as {}".format(galImageName))

            #add the galaxy image to the signalimage
            signalimage = signalimage + galaxyCRImage

        #EXTENDED SOURCES
        #print("Currently, getImage assumes that inputs are oriented north up and rotates by the rotation angle")
        #print("in the parameter file. would be nice to make this smarter. Look for rotation angle within the")
        #print("input image?")

        #read in extended signal image and add the image to the overall image
        if self.runStep['extendedsource'] == True:
            print("Reading in extended image from {} and adding to the simulated data rate image.".format(self.params['simSignals']['extended']))

            #Assume that the extendedCenter coordinates in the parameter
            #file are in the coordinate system of the standard output
            #signal rate image, NOT the extended range grism source image
            self.params['simSignals']['extendedCenter'] += np.array([self.coord_adjust['xoffset'],self.coord_adjust['yoffset']],dtype=np.int)

            extendedimage,extendedheader = self.getImage(self.params['simSignals']['extended'],arrayshape,True,self.params['Telescope']['rotation'],self.params['simSignals']['extendedCenter'])


            #convolve the extended image with the PSF if requested
            if self.params['simSignals']['PSFConvolveExtended'] == True:
                extendedimage = s1.fftconvolve(extendedimage,centerpsf,mode='same')

            #add the extended image to the synthetic signal rate image
            signalimage = signalimage + extendedimage*self.params['simSignals']['extendedscale']


        #ZODIACAL LIGHT
        if self.runStep['zodiacal'] == True:
            zodiangle = self.eclipticangle() - self.params['Telescope']['rotation']
            zodiacalimage,zodiacalheader = self.getImage(self.params['simSignals']['zodiacal'],arrayshape,True,zodiangle,arrayshape/2)
            signalimage = signalimage + zodiacalimage*self.params['simSignals']['zodiscale']


        #SCATTERED LIGHT - no rotation here. 
        if self.runStep['scattered']:
            scatteredimage,scatteredheader = self.getImage(self.params['simSignals']['scattered'],arrayshape,False,0.0,arrayshape/2)
            signalimage = signalimage + scatteredimage*self.params['simSignals']['scatteredscale']


        #CONSTANT BACKGROUND
        signalimage = signalimage + self.params['simSignals']['bkgdrate']


        #Save the image containing all of the added sources from the 'sky'
        if self.params['Output']['save_intermediates'] == True:
            sourcesImageName = self.params['Output']['file'][0:-5] + '_AddedSourcesRateImage_elec_per_sec.fits'
            self.saveSingleFits(signalimage,sourcesImageName)
            print("Image of added sources from the 'sky' saved as {}".format(sourcesImageName))

        #ILLUMINATION FLAT
        if self.runStep['illuminationflat']:
            illuminationflat,illuminationflatheader = self.readCalFile(self.params['Reffiles']['illumflat'])
            signalimage = signalimage * illuminationflat

        #PIXEL FLAT
        if self.runStep['pixelflat']:
            pixelflat,pixelflatheader = self.readCalFile(self.params['Reffiles']['pixelflat'])
            signalimage = signalimage * pixelflat

        #IPC EFFECTS
        if self.runStep['ipc']:
            ipcimage,ipcimageheader = self.readCalFile(self.params['Reffiles']['ipc'])
                
            #Assume that the IPC kernel is designed for the removal of IPC, which means we need to
            #invert it.
            if self.params['Reffiles']['invertIPC']:
                print("Iverting IPC kernel prior to convolving with image")
                yk,xk = ipcimage.shape
                newkernel = 0. - ipcimage
                newkernel[(yk-1)/2,(xk-1)/2] = 1. - (ipcimage[1,1]-np.sum(ipcimage))
                ipcimage = newkernel
                print(ipcimage)

            signalimage = s1.fftconvolve(signalimage,ipcimage,mode="same")

        #CROSSTALK
        if self.runStep['crosstalk']:
            if self.params['Readout']['namp'] == 4:
                #Only sources on the detector will create crosstalk. If signalimage is larger than full frame
                #because we are creating a grism image, then extract the pixels corresponding to the actual
                #detector, and only create crosstalk values for those.
                sigshape = signalimage.shape
                xtinput = signalimage
                xs = 0
                xe = sigshape[1]
                ys = 0
                ye = sigshape[0]
                if self.params['Output']['grism_source_image']:
                    xs,xe,ys,ye = self.extractFromGrismImage(signalimage)  
                    xtinput = signalimage[ys:ye,xs:xe]
                  
                xtcoeffs = self.readCrossTalkFile(self.params['Reffiles']['crosstalk'],self.detector[3:5].upper())
                xtimage = self.crossTalkImage(xtinput,xtcoeffs)
                
                #Now add the crosstalk image to the signalimage
                signalimage[ys:ye,xs:xe] += xtimage

            else:
                print("Crosstalk calculation requested, but the chosen subarray is read out using only 1 amplifier.")
                print("Therefore there will be no crosstalk. Skipping this step.")

        #OCCULTING SPOT - don't rotate the input image
        if self.runStep['occult']:
            occultimage,occultheader = self.getImage(self.params['Reffiles']['occult'],arrayshape,False,0.0)
            signalimage = signalimage * occultimage


        #If the desired output is a countrate image to be used to create a model dispersed image, 
        #then signalimage here is what you want. It contains all of the modeled input signals, along
        #with the effects of the flat fields and IPC. So, output this signalrate image to a fits file.
        if self.params['Output']['grism_source_image'] == True:
            grismDirectName = self.params['Output']['file'][0:-5] + '_GrismDirectImage_elec_per_sec.fits'
            xcent_fov = arrayshape[1] / 2
            ycent_fov = arrarshape[1] / 2
            kw = {}
            kw['xcenter'] = xcent_fov
            kw['ycenter'] = ycent_fov
            self.saveSingleFits(signalimage,grismDirectName,key_dict=kw)
            print("Image to be used as input to make dispersed grism image saved as {}".format(grismDirectName))

            if not self.params['Output']['grism_input_only']:
                #Now, continue on and create the full ramp. First, cut down the extra pixels associated with the 
                #input for the grism modeling, and push on with the nominal-sized output array
                xs,xe,ys,ye = self.extractFromGrismImage(signalimage) 
                signalimage = signalimage[ys:ye,xs:xe]
                print('Signalimage cropped back down to nominal shape ({})'.format(signalimage.shape))
            else:
                print("grism_input_only set to True in {}. Quitting.".format(self.paramfile))
                sys.exit()

        #Apply the mask image to make sure the refrence pixels haven't had any signal added
        signalimage = signalimage * maskimage


        #if self.params['Output']['save_intermediates'] == True:
        #    sourcesImageName = self.params['Output']['file'][0:-5] + '_TEMPmaskedSigRateImg.fits'
        #    self.saveSingleFits(signalimage,sourcesImageName)
        #    print("Image of added sources from the 'sky' saved as {}".format(sourcesImageName))

        #Apply the pixel area map to introduce distortion
        if self.runStep['pixelAreaMap']:
            pixAreaMap = self.simpleGetImage(self.params['Reffiles']['pixelAreaMap'])
            print("NEED TO FIGURE OUT IF WE ARE MULT OR DIVIDING THE PAM!!!")
            print("Refpix are set to 0 in the PAM, so if we divide, we need to deal with that.")
            signalimage = signalimage * pixAreaMap

        #Save the final rate image of added signals
        if self.params['Output']['save_intermediates'] == True:
            rateImageName = self.params['Output']['file'][0:-5] + '_AddedSourcesPlusDetectorEffectsRateImage_elec_per_sec.fits'
            self.saveSingleFits(signalimage,rateImageName)
            print("Signal rate image of all added sources (plus flats and IPC applied if requested) saved as {}".format(rateImageName))

        #Multiply the signal rate image by the calculated frame time in order to get an 
        #image of the signal for a single frame
        signalimage = signalimage * self.frametime

        #print('after all simulated signals, times frametime',np.max(signalimage),np.min(signalimage))

        return signalimage

    def extractFromGrismImage(self,array):
        #return the indexes that will allow you to extract the nominal output
        #image from the extra-large grism source image
        arrayshape = array.shape
        nominaly,nominalx = self.dark.data[0,0,:,:].shape
        diffx = (arrayshape[1] - nominalx) / 2
        diffy = (arrayshape[0] - nominaly) / 2
        x1 = diffx
        y1 = diffy
        x2 = arrayshape[1] - diffx
        y2 = arrayshape[0] - diffy
        return x1,x2,y1,y2

    def cropPSF(self,psf):
        '''take an array containing a psf and crop it such that the brightest
        pixel is in the center of the array'''
        nyshift,nxshift = np.where(psf == np.max(psf))
        nyshift = nyshift[0]
        nxshift = nxshift[0]
        py,px = psf.shape

        xl = nxshift - 0
        xr = px - nxshift - 1
        if xl <= xr:
            xdist = xl
        if xr < xl:
            xdist = xr

        yl = nyshift - 0
        yr = py - nyshift - 1
        if yl <= yr:
            ydist = yl
        if yr < yl:
            ydist = yr

        return psf[nyshift-ydist:nyshift+ydist+1,nxshift-xdist:nxshift+xdist+1]
    

    def simpleGetImage(self,name):
        '''read in an array from a fits file and crop using subarray_bounds'''
        try:
           image,header = fits.getdata(name,header=True)
        except:
            print('WARNING: unable to read in {}'.format(name))
            sys.exit() 

        #assume that the input is 2D, since we are using it to build a signal rate frame
        imageshape=image.shape
        if len(imageshape) != 2:
            self.printfunc("Error: image %s is not two-dimensional" % (name))
            return None,None

        imageshape = image.shape

        try:
            image = image[self.subarray_bounds[1]:self.subarray_bounds[3]+1,self.subarray_bounds[0]:self.subarray_bounds[2]+1]
        except:
            print("Unable to crop image from {}".format(name))
            sys.exit

        return image
            
        
            
    def getImage(self,name,arrayshape,rotateflag,angle,place=[0,0]):
        #Read in a countrate image, rotate/crop, and return
        #Assume that the center of the image will align with
        #the reference pixel location on the array to be simulated

        #read in input array from fits file
        try:
            image,header = fits.getdata(name,header=True)
        except:
            print('WARNING: unable to read in {}'.format(name))
            sys.exit()

        # check for negative signal values which are not allowed.
        if (np.min(image) < 0.):
            print("WARNING: negative signal values in the input image {}".format(name))
            #return None,None

        #assume that the input is 2D, since we are using it to build a signal rate frame
        imageshape=image.shape
        if len(imageshape) != 2:
            self.printfunc("Error: image %s is not two-dimensional" % (name))
            return None,None

        # rotate the image if required
        if rotateflag and angle != 0.0:
            image=self.rotate(image,0.-angle)

        # extract the proper subarray if necessary,
        # if it is larger than the nominal full frame image size (which would be required 
        # if it is to be rotated properly and still fill the field of view
        imageshape = image.shape
        #nominalarrayshape = np.array([self.subarray_bounds[3]-self.subarray_bounds[1]+1,self.subarray_bounds[2]-self.subarray_bounds[0]+1])

        #print('before cropping:',arrayshape,imageshape)

        #Crop the input image to match the output array shape
        #place = [placex,placey]
        #for i in xrange(2):
        if imageshape != arrayshape:
            #image = self.cropImageDimension(name,image,arrayshape,i,place[i])
            image = self.cropImage(name,image,arrayshape,place)
            #print('after cropping in {} dimension, {},{}'.format(i,image.shape,arrayshape))

        #print(image.shape,arrayshape)
        #sys.exit()

        return image,header


    def cropImage(self,name,image,arrayshape,places):
        #crop a given image to match the dimensions of arrayshape.
        #Center the image at the coordinates places within the array represented
        #by arrayshape
        imageshape = image.shape

        imagecenterx = imageshape[1]/2
        imagecentery = imageshape[0]/2

        arrayplacex = places[0]
        arrayplacey = places[1]

        #offsets between coordinate systems
        deltacenterx = imagecenterx - arrayplacex
        deltacentery = imagecentery - arrayplacey
        
        #find the edges of the image in array coords
        minimagex_inarr = 0 - deltacenterx
        maximagex_inarr = imageshape[1] - deltacenterx
        minimagey_inarr = 0 - deltacentery
        maximagey_inarr = imageshape[0] - deltacentery

        #find the edges of the array in image coords
        minarrx_inim = 0 + deltacenterx
        maxarrx_inim = arrayshape[1] + deltacenterx
        minarry_inim = 0 + deltacentery
        maxarry_inim = arrayshape[0] + deltacentery
        
        #array to be returned
        outarr = np.zeros(arrayshape)

        #array indexes that correspond to the edges
        #of the image
        #minarrx = minimagex_inarr
        #if minimagex_inarr < 0:
        #    minarrx = 0 

        #maxarrx = maximagex_inarr
        #if maximagex_inarr > arrayshape[1]:
        #    maxarrx = arrayshape[1]

        minarrx = np.max([minimagex_inarr,0])
        maxarrx = np.min([maximagex_inarr,arrayshape[1]])
            
        #minarry = minimagey_inarr
        #if minimagey_inarr < 0:
        #    minarry = 0 

        #maxarry = maximagey_inarr
        #if maximagey_inarr > arrayshape[0]:
        #    maxarry = arrayshape[0]

        minarry = np.max([minimagey_inarr,0])
        maxarry = np.min([maximagey_inarr,arrayshape[0]])
   
        #image indexes that correspond to the section to be added
        #minimx = 0
        #if minarrx_inim > 0:
        #    minimx = minarrx_inim

        minimx = np.max([0,minarrx_inim])
        maximx = np.min([imageshape[1],maxarrx_inim])
   
        #maximx = imageshape[1]
        #if maxarrx_inim < imageshape[1]:
        #    maximx = maxarrx_inim

        minimy = np.max([0,minarry_inim])
        maximy = np.min([imageshape[0],maxarry_inim])
        
        #minimy = 0
        #if minarry_inim > 0:
        #    minimy = minarry_inim

        #maximy = imageshape[0]
        #if maxarry_inim < imageshape[0]:
        #    maximy = maxarry_inim

        print('CHECKING CROPIMAGE RESULTS:')
        print(imageshape,arrayshape)
        print(minarry,maxarry,minarrx,maxarrx,minimy,maximy,minimx,maximx)
        outarr[minarry:maxarry,minarrx:maxarrx] = image[minimy:maximy,minimx:maximx]
        return outarr



    def cropImageDimension(self,name,image,arrayshape,axis,placey):
        #crop a given image in one dimension
        imageshape = image.shape
        nominalarrayshape = np.array([self.subarray_bounds[3]-self.subarray_bounds[1]+1,self.subarray_bounds[2]-self.subarray_bounds[0]+1])

        if axis == 0:
            sb = 1
        elif axis == 1:
            sb = 0
        else:
            print("WARNING: axis needs to be either 0 (for y dimension) or 1 (for x dimension)")
            sys.exit()

        #if imageshape != arrayshape:

        #print('inside getimage {} {}'.format(imageshape,arrayshape))
        if (imageshape[axis] > arrayshape[axis]): #image read in is larger than eventual signal rate image

            #min and max coordinates used in array. placeholders for later.
            yminarray = 0.
            ymaxarray = arrayshape[axis]

            #CASE: where the input image is larger than or equal to a full frame
            if (imageshape[axis] >= self.ffsize):
                sub_width = self.subarray_bounds[sb+2]-self.subarray_bounds[sb]+1

                if (nominalarrayshape[axis] <= self.ffsize):
                    #First, get indexes to extract full frame from this extra large image
                    ymin = imageshape[axis]/2 - self.ffsize/2
                    ymax = ymin + self.ffsize
                    #print('first cut, ymin, ymax {}, {}'.format(ymin,ymax))

                if nominalarrayshape[axis] < self.ffsize:    
                    #Now adjust the indexes to be able to extract the subarray from the extra large image
                    ymin = ymin + self.subarray_bounds[sb]
                    #sub_width = self.subarray_bounds[3]-self.subarray_bounds[1]
                    ymax = ymin + sub_width
                    #print('second cut, ymin,ymax {},{}'.format(ymin,ymax))

                #Now, if a grism source image is requested, adjust the indexes to account
                #for the extra rows and columns
                if self.params['Output']['grism_source_image'] == True:
                    #print(ymin,sub_width,np.int(sub_width*(self.grism_direct_factor-1.)),np.int(np.int(sub_width*(self.grism_direct_factor-1.))/2.))
                    ymin = ymin - np.int(np.int(sub_width*(self.grism_direct_factor-1.))/2.)
                    ymax = ymax + np.int(np.int(sub_width*(self.grism_direct_factor-1.))/2.)
                    #print('grism expansion ymin, ymax {}, {}'.format(ymin,ymax))
                 
                    if axis == 0:
                        top = 'top'
                        bottom = 'bottom'
                    else:
                        top = 'right'
                        bottom = 'left'

                    if ymin < 0:
                        print("WARNING: Location and size of proposed subarray extends beyond the {} boundary of the input image {}. ".format(bottom,name))
                        yminarray = 0 - ymin
                        ymin = 0
                    if ymax > imageshape[axis]:
                        print("WARNING: Location and size of proposed subarray extends beyond the {} boundary of the input image {}. ".format(top,name))
                        #ymaxarray = ymax - imageshape[axis]
                        ymaxarray = yminarray + imageshape[axis] - (ymax-arrayshape[axis])
                        ymax = imageshape[axis]
                        #print('now ymaxarray is {}'.format(ymaxarray))

            #Here is the case where the input image is larger than the output array
            #but it is smaller than full frame size. The best we can do here is extract
            #the center of the input image and place it in th eoutput array
            else:
                ymin = imageshape[axis]/2 - arrayshape[axis]/2
                ymax = ymin + arrayshape[axis]


            #if the input image wasn't large enough to cover the proposed output array (such as for a subarray in the corner,
            #and a requested grism source image, which extends the output array beyond the edges of the full frame, but the 
            #input image is only a full frame-sized image.) then we need to add some blank pixels to the edge of the cropped
            #image, so that the returned image is the same size as the output array


        #CASE: where input image is smaller than the output image
        elif (imageshape[axis] < arrayshape[axis]):
            if axis == 0:
                ax = 'y'
            else:
                ax = 'x'
            print("Image within {} is smaller than the proposed output array size of {} in the {}-direction.".format(name,arrayshape[axis],ax))
            print("Centering the image at pixel location {}.".format(placey))

            ymin = 0
            ymax = imageshape[axis]
            yminarray = placey - imageshape[axis]/2
            ymaxarray = yminarray + imageshape[axis]
            if ymaxarray > arrayshape[axis]:
                print("CAUTION: Proposed location of extended image center will cause it to fall off the top side of the output array.")
                ymax = ymax - (ymaxarray-arrayshape[axis])
                ymaxarray = arrayshape[axis]-1
            if yminarray < 0:
                print("CAUTION: Proposed location of extended image center will cause it to fall off the bottom side of the output array.")
                ymin = ymin - yminarray
                yminarray = 0

        #extract the appropriate y range from the input image
        print('just before cropping, ymin, ymax {},{}, image shape is {}'.format(ymin,ymax,image.shape))
        if axis == 0:
            image = image[ymin:ymax,:]
        else:
            image = image[:,ymin:ymax]
        #print('after cropping',image.shape)

        #hh = fits.PrimaryHDU(image)
        #hl = fits.HDUList([hh])
        #hl.writeto('testing_cropped.fits',clobber=True)

        #in the case where the image fell off the edge of the detector, place the image in a blank array so 
        #that the result is the same size as the output array
        if ((yminarray != 0) | (ymaxarray != arrayshape[axis])):
            if axis == 0:
                fullimage = np.zeros((arrayshape[0],image.shape[1]))
                #print(image.shape,yminarray,ymaxarray)
                fullimage[yminarray:ymaxarray,:] = image
            else:
                fullimage = np.zeros((image.shape[0],arrayshape[1]))
                fullimage[:,yminarray:ymaxarray] = image

            image = fullimage

        hh = fits.PrimaryHDU(image)
        hl = fits.HDUList([hh])
        hl.writeto('testing.fits',clobber=True)


        return image
                           

    def rotate(self,image,angle):
        newangle=-angle
        # interpolation.angle measures angles clockwise, so have to invert the angle
        newimage=interpolation.rotate(image,newangle,reshape=False,mode="constant",cval=image.min(),order=5)
        newimage[np.where(newimage < 0.)]=np.min(image)
        return newimage


    #def saveSingleFits(self,image,name):
    #    #save an array into the first extension of a fits file
    #    h0 = fits.PrimaryHDU()
    #    h1 = fits.ImageHDU(image)
    #    hdulist = fits.HDUList([h0,h1])
    #    hdulist.writeto(name,clobber=True)

    def saveSingleFits(self,image,name,key_dict=None):
        #save an array into the first extension of a fits file
        h0 = fits.PrimaryHDU()
        h1 = fits.ImageHDU(image)

        #if a keyword dictionary is provided, put the 
        #keywords into the 0th and 1st extension headers
        if key_dict is not None:
            for key in key_dict:
                h0.header[key] = key_dict[key]
                h1.header[key] = key_dict[key]

        hdulist = fits.HDUList([h0,h1])
        hdulist.writeto(name,clobber=True)


    def dist(self,pos1,pos2):
        #calculates distance between two points
        ang=self.posang(pos1,pos2)
        if ang is None:
            return None,None
        if ang > 360.:
            ang=ang-360.
        if ang < 0.:
            ang=ang+360.
        dtor=math.radians(1.)
        arcdist=math.sin(pos1[1]*dtor)*math.sin(pos2[1]*dtor)+math.cos(pos1[1]*dtor)*math.cos(pos2[1]*dtor)*math.cos(dtor*(pos1[0]-pos2[0]))
        if abs(arcdist) > 1.:
            return 180.,ang
        else:
            arcdist=math.acos(arcdist)/dtor
            if arcdist < 0.:
                arcdist=arcdist+180.
            return arcdist,ang


#    def orig_skytopixel(self,xreal,yreal):
#        if len(self.xpow) < 6:
#            return xreal,yreal
#        else:
#            a1=0.
#            a2=0.
#            for k in range(len(self.xpow)):
#                t1=xreal**self.xpow[k]
#                t2=yreal**self.ypow[k]
#                a1=a1+self.xtrans2[k]*t1*t2
#                a2=a2+self.ytrans2[k]*t1*t2
#            return a1,a2
#
#    def scitoideal(self,xreal,yreal,xref,yref,degree,coeffs):
#        #originally pixeltosky(?????)
#        #translate an input pixel coordinate from distorted
#        #to undistorted coordinates - see the first two 
#        #equations in JWST-STScI-001550
#        counter = 1
#        xideal = 0
#        yideal = 0
#        for i in xrange(1,degree+1):
#            for j in xrange(i+1):
#                xideal = xideal + coeffs[0,counter] * (xreal-xref)**(i-j) * (yreal-yref)**j
#                yideal = yideal + coeffs[1,counter] * (xreal-xref)**(i-j) * (yreal-yref)**j
#                counter += 1
#        return xideal,yideal
#
#
#    def idealtosci(self,x1,y1,xref,yref,degree,coeffs):
#        #ideal coord system is in units of arcsecs!!!
#        #previously skytopixel(????)
#        #translate an input pixel coordinate from undistorted
#        #to distorted coordinates. See the second two equations
#        #in JWST-STScI-001550
#        deltax = 0.
#        deltay = 0.
#        counter = 1
#        for i in xrange(1,degree+1):
#            for j in xrange(i+1):
#                deltax = deltax + coeffs[0,counter]*x1**(i-j)*y1**j
#                #print(i,j,x1,(i-j),x1**(i-j),y1,j,y1**j,x1**(i-j)*y1**j)
#                deltay = deltay + coeffs[1,counter]*x1**(i-j)*y1**j
#                counter += 1
#        return deltax + xref, deltay + yref
#
#
#    def orig_pixeltosky(self,x1,y1):
#        if len(self.xpow) < 3:
#            return x1,y1
#        else:
#            a1=0.
#            a2=0.
#            for k in range(len(self.xpow)):
#                t1=x1**self.xpow[k]
#                t2=y1**self.ypow[k]
#                a1=a1+self.xtrans1[k]*t1*t2
#                a2=a2+self.ytrans1[k]*t1*t2
#            return a1,a2

    def posang(self,pos1,pos2):
        #calculate the position angle between two points
        dtor=math.radians(1.)
        if abs(pos1[1]) > 90. or abs(pos2[1]) > 90.:
            return None
        else:
            if pos1[1] == pos2[1] and pos1[0] == pos2[0]:
                return 0.0
            else:
                angle=math.atan2(math.sin((pos2[0]-pos1[0])*dtor),math.cos(pos1[1]*dtor)*math.tan(pos2[1]*dtor)-math.sin(pos1[1]*dtor)*math.cos((pos2[0]-pos1[0])*dtor))/dtor
            return angle


    def makePos(self,alpha1,delta1):
        #given a numerical RA/Dec pair, convert to string
        #values hh:mm:ss
        if alpha1 < 0.: 
            alpha1=alpha1+360.
        if delta1 < 0.: 
            sign="-"
            d1=abs(delta1)
        else:
            sign="+"
            d1=delta1
        decd=int(d1)
        value=60.*(d1-float(decd))
        decm=int(value)
        decs=60.*(value-decm)
        a1=alpha1/15.0
        radeg=int(a1)
        value=60.*(a1-radeg)
        ramin=int(value)
        rasec=60.*(value-ramin)
        alpha2="%2.2d:%2.2d:%7.4f" % (radeg,ramin,rasec)
        delta2="%1s%2.2d:%2.2d:%7.4f" % (sign,decd,decm,decs)
        alpha2=alpha2.replace(" ","0")
        delta2=delta2.replace(" ","0")
        return alpha2,delta2


    def getDistortionCoeffs(self,inst,det):
        #choose the appropriate distortion coefficients for the instrument/detector
        apname = det + '_' + self.params['Readout']['array_name']

        #match using instrument name and aperture name (which is detector + subarray name)
        match = ((self.astrom_coeffs['InstrName'] == inst) & (self.astrom_coeffs['AperName'] == apname))
        if sum(match) == 1:
            #get the reference pixel position (units are in full frame coords)
            #These are also indexed to 1, not zero. Be careful when using 
            #with idealtosky where we need it indexed to 1, versus other python
            #applications where it should be indexed to zero.
            xdetref = self.astrom_coeffs['XSciRef'].data[match] 
            ydetref = self.astrom_coeffs['YSciRef'].data[match] 
            
            #Extract the pixel scale in x and y
            xscale = self.astrom_coeffs['XSciScale'].data[match].data[0]
            yscale = self.astrom_coeffs['YSciScale'].data[match].data[0]

            #polynomial degree of the polynomials to be used
            deg = self.astrom_coeffs['Sci2IdlDeg'].data[match]

            #lists to hold the coefficients
            xidl2sci = []
            yidl2sci = []
            xsci2idl = []
            ysci2idl = []
            
            for xpow in xrange(deg+1):
                for ypow in xrange(xpow+1):
                    xsci2idl.append(self.astrom_coeffs['Sci2IdlX'+str(xpow)+str(ypow)].data[match])
                    ysci2idl.append(self.astrom_coeffs['Sci2IdlY'+str(xpow)+str(ypow)].data[match])
                    xidl2sci.append(self.astrom_coeffs['Idl2SciX'+str(xpow)+str(ypow)].data[match])
                    yidl2sci.append(self.astrom_coeffs['Idl2SciY'+str(xpow)+str(ypow)].data[match])

            #extract the actual values from the masked array
            xs2i = [l.data[0] for l in xsci2idl]
            ys2i = [l.data[0] for l in ysci2idl]
            xi2s = [l.data[0] for l in xidl2sci]
            yi2s = [l.data[0] for l in yidl2sci]

            #return one list for each direction. (sci -> ideal and ideal -> sci)
            #Each list will be 2D and created from one list for X's and one for Y's
            idl2sci = np.stack([np.array(xi2s),np.array(yi2s)],axis=0)
            sci2idl = np.stack([np.array(xs2i),np.array(ys2i)],axis=0)

            return deg[0],xdetref[0],ydetref[0],xscale,yscale,idl2sci,sci2idl

        else:
            if sum(match) == 0:
                print('WARNING: distortion coefficients for {} not found in {}'.format(apname,self.params['Reffiles']['astrometric']))
            if sum(match) > 1:
                print('WARNING: more than one set of matching distortion coefficients for {} found in {}.'.format(apname,self.params['Reffiles']['astrometric']))
            sys.exit()
            

    def readCrossTalkFile(self,file,detector):
        #read in crosstalk coefficients file
        #read in appropriate line from the xtalk file and return the coeffs
        xtcoeffs = ascii.read(file,header_start=0)

        coeffs = []
        mtch = xtcoeffs['Det'] == detector.upper()
        if np.any(mtch) == False:
            print('Detector {} not found in xtalk file {}'.format(detector,file))
            sys.exit()

        return xtcoeffs[mtch]


    def crossTalkImage(self,orig,coeffs):
        #using Xtalk coefficients, generate an image of the crosstalk signal
        xtalk_corr_im = np.zeros_like(orig)
        subamp_shift = {"0":1,"1":-1,"2":1,"3":-1}

        #List of starting columns for all quadrants.
        #This should be the same for all instruments, right?
        xtqstart = [0,512,1024,1536,2048]

        for amp in xrange(4):
            to_mult = orig[:,xtqstart[amp]:xtqstart[amp+1]]
            receivers = []
            for i in xrange(4):
                if i != amp:
                    receivers.append(i)
            #reverse the values to multply if the amps being used are adjacent or 3 amps apart
            for subamp in receivers:
                index = 'xt'+str(amp+1)+str(subamp+1)
                if ((np.absolute(amp-subamp) == 1) | (np.absolute(amp-subamp) == 3)):
                    corr_amp = np.fliplr(to_mult) * coeffs[index]
                if (np.absolute(amp-subamp) == 2):
                    corr_amp = to_mult * coeffs[index]
            
                xtalk_corr_im[:,xtqstart[subamp]:xtqstart[subamp+1]] += corr_amp 

            #per Armin's instructions, now repeat the process using his xt??post coefficients, but shift the arrays
            #by one pixel according to readout direction.
            #to_mult = xtalk_corr_im[group,:,qstart[amp]:qstart[amp+1]]
            for subamp in receivers:
                index = 'xt'+str(amp+1)+str(subamp+1)+'post'
                if ((np.absolute(amp-subamp) == 1) | (np.absolute(amp-subamp) == 3)):
                    corr_amp = np.fliplr(to_mult) * coeffs[index] 
                    corr_amp = np.roll(corr_amp,subamp_shift[str(subamp)],axis=1)
                if (np.absolute(amp-subamp) == 2):
                    corr_amp = to_mult * coeffs[index]
                    corr_amp = np.roll(corr_amp,subamp_shift[str(subamp)])

                xtalk_corr_im[:,xtqstart[subamp]:xtqstart[subamp+1]] += corr_amp
                    
        #save the crosstalk correction image
        if self.params['Output']['save_intermediates'] == True:
            phdu = fits.PrimaryHDU(xtalk_corr_im)
            phdu.writeto('xtalk_correction_image.fits',clobber=True)

        return xtalk_corr_im


    def getPointSourceList(self):
        #read in the list of point sources to add, and adjust the
        #provided positions for astrometric distortion
        dummypos=' 00:00:00.00'
        dummydeg=0.00

        #find the array sizes of the PSF files in the library. Assume they are all the same.
        #We want the distance from the PSF peak to the edge, assuming the peak is centered
        if self.params['simSignals']['psfwfe'] != 0:
            numstr = str(self.params['simSignals']['psfwfe'])
        else:
            numstr = 'zero'
        psflibfiles = glob.glob(self.params['simSignals']['psfpath'] +'*')


        #If a PSF library is specified, then just get the dimensions from one of the files
        if self.params['simSignals']['psfpath'] != None:
            h = fits.open(psflibfiles[0])
            edgex = h[0].header['NAXIS1'] / 2 - 1
            edgey = h[0].header['NAXIS2'] / 2 - 1
            self.psfhalfwidth = np.array([edgex,edgey])
            h.close()
        else:
            #if no PSF library is specified, then webbpsf will be creating the PSF on the 
            #fly. In this case, we assume webbpsf's default output size of 301x301 pixels?
            edgex = int(301 / 2)
            edgey = int(301 / 2)
            print("INFO: no PSF library specified, but point sources are to be added to")
            print("the output. PSFs will be generated by WebbPSF on the fly")
            print("Not yet implemented.")
            sys.exit()


        #print('initially, edgex and y are {}, {}'.format(edgex,edgey))

        #If you are making a signal rate image for grism simulator input, then save
        #all the sources that would fall on the extended image. Add the number
        #of "extra" pixels added to the borders of the image to the already-computed
        #value of edge.
        #if self.params['Output']['grism_source_image']:
        #    print("before accounting for the extra large grism array size, edgex and edgey are {},{}".format(edgex,edgey))
        #    yd,xd = self.dark.data[0,0,:,:].shape
        #    edgex = edgex + np.int((np.int(xd*self.grism_direct_factor) - xd) / 2)
        #    edgey = edgey + np.int((np.int(yd*self.grism_direct_factor) - yd) / 2)
        #    print('edgex,edgey are {},{}'.format(edgex,edgey))

        # Read in the point source list
        #self.pointSourceList=[]

        pointSourceList = Table(names=('pixelx','pixely','RA','Dec','RA_degrees','Dec_degrees','magnitude','countrate_e/s','counts_per_frame_e'),dtype=('f','f','S14','S14','f','f','f','f','f'))

        try:
            filename = self.params['simSignals']['pointsource']
            #psfile = open(filename,"r")
            #lines = psfile.readlines()
            lines,pixelflag = self.readPointSourceFile(filename)
            if pixelflag:
                print("Point source list input positions assumed to be in units of pixels.")
            else:
                print("Point list input positions assumed to be in units of RA and Dec.") 
            #psfile.close()
        except:
            print("WARNING: Unable to open the point source list file {}".format(filename))
            sys.exit()

        #File to save adjusted point source locations
        pslist = open(self.params['Output']['file'][0:-5] + '_pointsources.list','w')
        #pslist = open("pointsources.list","w")

        dtor = math.radians(1.)
        nx = (self.subarray_bounds[2]-self.subarray_bounds[0])+1
        ny = (self.subarray_bounds[3]-self.subarray_bounds[1])+1
        xc = (self.subarray_bounds[2]+self.subarray_bounds[0])/2.
        yc = (self.subarray_bounds[3]+self.subarray_bounds[1])/2.

        #Location of the subarray's reference pixel. 
        xrefpix = self.refpix_pos['x']
        yrefpix = self.refpix_pos['y']

        # center positions, sub-array sizes in pixels
        # now offset the field center to array center for astrometric distortion corrections
        coord_transform = None
        if self.runStep['astrometric']:

            #Read in the CRDS-format distortion reference file
            with AsdfFile.open(self.params['Reffiles']['astrometric']) as dist_file:
                coord_transform = dist_file.tree['model']

        #Using the requested RA,Dec of the reference pixel, along with the 
        #V2,V3 of the reference pixel, and the requested roll angle of the telescope
        #create a matrix that can be used to translate between V2,V3 and RA,Dec
        #for any pixel.
        #v2,v3 need to be in arcsec, and RA, Dec, and roll all need to be in degrees
        attitude_matrix = rotations.attitude(self.refpix_pos['v2'],self.refpix_pos['v3'],self.ra,self.dec,self.params['Telescope']["rotation"])
      
        #Define the min and max source locations (in pixels) that fall onto the subarray
        #Inlude the effects of a requested grism_direct image, and also keep sources that
        #will only partially fall on the subarray
        #pixel coords here can still be negative and kept if the grism image is being made

        #First, coord limits for just the subarray
        miny = 0
        maxy = self.subarray_bounds[3] - self.subarray_bounds[1] 
        minx = 0
        maxx = self.subarray_bounds[2] - self.subarray_bounds[0] 
        #print('before adjusting for grism, min/max x {} {}, min/max y {} {}'.format(minx,maxx,miny,maxy))
        
        #Expand the limits if a grism direct image is being made
        if self.params['Output']['grism_source_image'] == True:
            #extrapixyold = np.int((maxy+1)/2 * (self.grism_direct_factor - 1.))
            extrapixy = np.int((maxy+1)/2 * (self.coord_adjust['y'] - 1.))
            miny -= extrapixy
            maxy += extrapixy
            extrapixx = np.int((maxx+1)/2 * (self.coord_adjust['x'] - 1.))
            #extrapixxold = np.int((maxx+1)/2 * (self.grism_direct_factor - 1.))
            #print(extrapixy,extrapixx)
            minx -= extrapixx
            maxx += extrapixx

        #print('updated for grism, min/max x {} {}, min/max y {} {}'.format(minx,maxx,miny,maxy))
        #print('extrapixx new and old {} {}, extrapixy new and old {} {}'.format(extrapixx,extrapixxold,extrapixy,extrapixyold))

        #Now, expand the dimensions again to include point sources that fall only partially on the 
        #subarray
        miny -= edgey
        maxy += edgey
        minx -= edgex
        maxx += edgex

        #print('updated to include sources that only partially intersect the subarray,  min/max x {} {}, min/max y {} {}'.format(minx,maxx,miny,maxy))

        #Write out the RA and Dec of the field center to the output file
        #Also write out column headers to prepare for source list
        pslist.write("# Field center (degrees): %13.8f %14.8f y axis rotation angle (degrees): %f  image size: %4.4d %4.4d\n" % (self.ra,self.dec,self.params['Telescope']['rotation'],nx,ny))
        pslist.write('#\n')
        pslist.write("#    RA_(hh:mm:ss)   DEC_(dd:mm:ss)   RA_degrees      DEC_degrees     pixel_x   pixel_y    magnitude   counts/sec    counts/frame\n")
        #pixelflag=False


        #if the top line of the source list file contains '# pixel'
        #the the script assumes the input values are x,y pixels
        #If '# pixel' is not present, it assumes columns 1 and 2 are
        #RA and Dec values
        #if 'position_pixel' in lines[0]:
        #    pixelflag=True
        #    print("Point source list input positions assumed to be in units of pixels.")
        #else:
        #    print("Point list input positions assumed to be in units of RA and Dec.") 

        #Loop over input lines in the source list 
        for values in lines:
            #try:
            #line below (if 1>0) used to keep the block of code below at correct indent for the try: above
            #the try: is commented out for code testing.
            if 1>0:
                try:
                    entry0 = float(values[0])
                    entry1 = float(values[1])
                    print(entry0,entry1,pixelflag)
                    if not pixelflag:
                        ra_str,dec_str = self.makePos(entry0,entry1)
                        #dec_str = self.makePos(entry0,entry1)
                        ra = entry0
                        dec = entry1
                except:
                    #if inputs can't be converted to floats, then 
                    #assume we have RA/Dec strings. Convert to floats.
                    ra_str = values[0]
                    dec_str = values[1]
                    ra,dec = self.parseRADec(ra_str,dec_str)

                #Case where point source list entries are given with RA and Dec
                if not pixelflag:

                    #If distortion is to be included - either with or without the full set of coordinate
                    #translation coefficients
                    if self.runStep['astrometric']:
                        pixelx,pixely = self.RADecToXY_astrometric(ra,dec,attitude_matrix,coord_transform)
                    else:
                        #No distortion at all - "manual mode"
                        pixelx,pixely = self.RADecToXY_manual(ra,dec)

                else:
                    #Case where the point source list entry locations are given in units of pixels
                    #In this case we have the source position, and RA/Dec are calculated only so 
                    #they can be written out into the output source list file.

                    #Assume that the input x and y values are coordinate values
                    #WITHIN THE SPECIFIED SUBARRAY. So for example, a source in the file
                    #at 0,0 when you are making a SUB160 ramp will fall on the lower left
                    #corner of the SUB160 subarray, NOT the lower left corner of the full
                    #frame. 

                    pixelx = entry0
                    pixely = entry1

                    ra,dec,ra_str,dec_str = self.XYToRADec(pixelx,pixely,attitude_matrix,coord_transform)
                            

                #Get the input magnitude of the point source
                mag=float(values[2])

                #Keep only sources within the appropriate bounds
                if pixely > miny and pixely < maxy and pixelx > minx and pixelx < maxx:
                            
                    #set up an entry for the output table
                    entry = [pixelx,pixely,ra_str,dec_str,ra,dec,mag]

                    #translate magnitudes to countrate
                    scale = 10.**(0.4*(15.0-mag))

                    #get the countrate that corresponds to a 15th magnitude star for this filter
                    cval = self.countvalues[self.params['Readout']['filter']]

                    #DEAL WITH THIS LATER, ONCE PYSYNPHOT IS INCLUDED WITH PIPELINE DIST?
                    if cval == 0:
                        print("Countrate value for {} is zero in {}.".format(self.params['Readout']['filter'],self.parameters['phot_file']))
                        print("Eventually attempting to calculate value using pysynphot.")
                        print("but pysynphot is not present in jwst build 6, so pushing off to later...")
                        sys.exit()
                        cval = self.findCountrate(self.params['Readout']['filter'])

                    #translate to counts in single frame at requested array size
                    framecounts = scale*cval*self.frametime
                    countrate = scale*cval

                    #add the countrate and the counts per frame to pointSourceList
                    #since they will be used in future calculations
                    #entry.append(scale)
                    entry.append(countrate)
                    entry.append(framecounts)

                    #add the good point source, including location and counts, to the pointSourceList
                    #self.pointSourceList.append(entry)
                    pointSourceList.add_row(entry)

                    #write out positions, distances, and counts to the output file
                    pslist.write("%s %s %14.8f %14.8f %9.3f %9.3f  %9.3f  %13.6e   %13.6e\n" % (ra_str,dec_str,ra,dec,pixelx,pixely,mag,countrate,framecounts))
                #except:
                #    print("ERROR: bad point source line %s. Skipping." % (line))
        print("Number of point sources found within the requested aperture: {}".format(len(pointSourceList)))
        #close the output file
        pslist.close()
        
        #print("testing distortion solutions")
        #sys.exit()

        #If no good point sources were found in the requested array, alert the user
        if len(pointSourceList) < 1:
            print("Warning: no point sources within the requested array.")
            print("The point source image option is being turned off")
            self.runStep['pointsource']=False
            if self.runStep['extendedsource'] == False and self.runStep['cosmicray'] == False:
                print("Error: no input point sources, extended image, nor cosmic rays specified")
                print("Exiting...")
                sys.exit()
        
        return pointSourceList


    def makePointSourceImage(self,pointSources):
        dims = np.array(self.dark.data[0,0,:,:].shape)

        #offset that needs to be applied to the x,y positions of the
        #source list to account for case where we make a point
        #source image that is extra-large, to be used as a grism 
        #direct image
        deltax = 0
        deltay = 0
        #psfimage = np.zeros_like(self.dark.data[0,0,:,:])

        #If a grism or TSO source image is requested, then we make the image
        #larger by the appropraite factor, and the coordinates
        #of the sources need to be adjusted accordingly.
        #xfactor = 1
        #yfactor = 1
        #if self.params['Output']['grism_source_image'] == True:
        #    xfactor = self.grism_direct_factor
        #    yfactor = self.grism_direct_factor
        #elif self.params['Inst']['mode'] == 'tso':
        #    xfactor = self.tso_factor['x']
        #    yfactor = self.tso_factor['y']

        #newdims = np.array(dims * self.grism_direct_factor,dtype=np.int)
        #deltax = np.int((newdims[1] - dims[1]) / 2)
        #deltay = np.int((newdims[0] - dims[0]) / 2)
        ##dims = newdims

        #print('previously deltax,deltay,newdims {} {} {}'.format(deltax,deltay,newdims))

        newdimsx = np.int(dims[1] * self.coord_adjust['x'])
        newdimsy = np.int(dims[0] * self.coord_adjust['y'])
        deltax = self.coord_adjust['xoffset']
        deltay = self.coord_adjust['yoffset']
        dims = np.array([newdimsy,newdimsx])

        #print('new deltax,deltay,newdims {} {} {}'.format(deltax,deltay,dims))
        #print('deltax,deltay',deltax,deltay)

        #create the empty image
        psfimage = np.zeros((dims[0],dims[1]))
        #dims=psfimage.shape

        #if the mode is spectroscopic, then return zeros
        #if self.params['Inst']['mode'] not in pointSourceModes:
        #    return psfimage


        #pointSourceList = Table(names=('pixelx','pixely','RA','Dec','RA_degrees','Dec_degrees','Ang_dist_arcsec','Position_angle_deg','magnitude','countrate_e/s','counts_per_frame_e'))


        #Loop over the entries in the point source list
        for entry in pointSources:
            #adjust x,y position if the grism output image is requested
            xpos = entry['pixelx'] + deltax
            ypos = entry['pixely'] + deltay

            #desired counts per second in the point source
            counts = entry['countrate_e/s'] #/ self.frametime

            #find sub-pixel offsets in position from the center of the pixel
            xoff = math.floor(xpos)
            yoff = math.floor(ypos)
            xfract = abs(xpos-xoff)
            yfract = abs(ypos-yoff)

            #Now we need to determine the proper PSF file to read in from the library
            #This depends on the sub-pixel offsets above
            #a=0.1*int(10.*xfract+0.5)-0.5 - original lines, assuming one file per 0.1 pix interval
            #b=0.1*int(10.*yfract+0.5)-0.5
            interval = self.params['simSignals']['psfpixfrac']
            numperpix = int(1./interval)
            a = interval * int(numperpix*xfract + 0.5) - 0.5
            b = interval * int(numperpix*yfract + 0.5) - 0.5

            if a < 0:
                astr = str(a)[0:4]
            else:
                astr = str(a)[0:3]
            if b < 0:
                bstr = str(b)[0:4]
            else:
                bstr = str(b)[0:3]

            #generate the psf file name based on the center of the point source
            #in units of fraction of a pixel
            frag = astr + '_' + bstr
            frag = frag.replace('-','m')
            frag = frag.replace('.','p')
            
            #now create the PSF image. If no PSF library is supplied
            #then webbpsf will be called to create a PSF. In that case, return
            #zeros right now for the PSF
            if self.params['simSignals']['psfpath'] is None:
                webbpsfimage = self.psfimage
            else:
                #case where PSF library location is specified. 
                #Read in the appropriate PSF file
                try:
                    psffn = self.psfname+'_'+frag+'.fits'
                    webbpsfimage = fits.getdata(psffn)
                except:
                    print("ERROR: Could not load PSF file {} from library".format(psffn))
                    sys.exit()

            #Extract the appropriate subarray from the PSF image if necessary
            #Assume that the brightest pixel corresponds to the peak of the psf
            nyshift,nxshift = np.where(webbpsfimage == np.max(webbpsfimage))
            nyshift = nyshift[0]
            nxshift = nxshift[0]

            psfdims = webbpsfimage.shape
            nx = int(xoff)
            ny = int(yoff)
            i1 = max(nx-nxshift,0)
            i2 = min(nx+1+nxshift,dims[1])
            j1 = max(ny-nyshift,0)
            j2 = min(ny+1+nyshift,dims[0])
            k1 = nxshift-(nx-i1)
            k2 = nxshift+(i2-nx)
            l1 = nyshift-(ny-j1)
            l2 = nyshift+(j2-ny)

            #if the cutout for the psf is larger than
            #the psf array, truncate it, along with the array
            #in the source image where it will be placed
            if l2 > psfdims[0]:
                l2 = psfdims[0]
                j2 = j1 + (l2-l1)

            if k2 > psfdims[1]:
                k2 = psfdims[1]
                i2 = i1 + (k2-k1)

            #At this point coordinates are in the final output array coordinate system, so there
            #should be no negative values, nor values larger than the output array size
            if j1 < 0 or i1<0 or l1<0 or k1<0:
                print(j1,i1,l1,k1)
                print('bad low')
                #sys.exit()
            if j2>(dims[0]+1) or i2>(dims[1]+1) or l2>(psfdims[1]+1) or k2>(psfdims[1]+1):
                print(j2,i2,l2,k2)
                print('bad high')
                #sys.exit()

            #print(psfimage.shape,'imx',j1,j2,'imy',i1,i2,webbpsfimage.shape,'psfx',l1,l2,'psfy',k1,k2)
            #print(yoff,ny,nyshift)
            psfimage[j1:j2,i1:i2] = psfimage[j1:j2,i1:i2] + webbpsfimage[l1:l2,k1:k2]*counts

            #print(np.max(webbpsfimage[l1:l2,k1:k2]*counts),np.max(psfimage[j1:j2,i1:i2]))

        return psfimage


    def makeGalaxyImage(self,psf):
        #Using the entries in the 'simSignals' 'galaxyList' file, create a countrate image
        #of model galaxies (sersic profile)

        #Read in the list of galaxies (positions and magnitides)
        glist, pixflag, radflag = self.readGalaxyFile(self.params['simSignals']['galaxyListFile'])
        if pixflag:
            print("Galaxy list input positions assumed to be in units of pixels.")
        else:
            print("Galaxy list input positions assumed to be in units of RA and Dec.") 

        if radflag:
            print("Galaxy list input radii assumed to be in units of pixels.")
        else:
            print("Galaxy list input radii assumed to be in units of RA and Dec.") 


        #Extract and save only the entries which will land (fully or partially) on the
        #aperture of the output
        galaxylist = self.filterGalaxyList(glist,pixflag,radflag)

        #galaxylist is a table with columns:
        #'pixelx','pixely','RA','Dec','RA_degrees','Dec_degrees','radius','ellipticity','pos_angle','sersic_index','magnitude','countrate_e/s','counts_per_frame_e'

        #final output image
        origyd,origxd = self.dark.data[0,0,:,:].shape
        yd = origyd
        xd = origxd
        
        #expand if a grism source image is being made
        xfact = 1
        yfact = 1
        if self.params['Output']['grism_source_image']:
            #xfact = self.grism_direct_factor
            #yfact = self.grism_direct_factor
            #elif 
            yd = np.int(origyd * self.coord_adjust['y'])
            xd = np.int(origxd * self.coord_adjust['x'])

        #create the final galaxy countrate image
        galimage = np.zeros((yd,xd))
        dims = galimage.shape
        #print('DIMENSIONS OF GALAXY OUTPUT: {}'.format(dims))

        #Adjust the coordinate system of the galaxy list if working with a grism direct image output
        deltax = 0
        deltay = 0
        if self.params['Output']['grism_source_image']:
            deltax = np.int((dims[1] - origxd) / 2)
            deltay = np.int((dims[0] - origyd) / 2)

        #For each entry, create an image, and place it onto the final output image
        for entry in galaxylist:
            
            #first create the galaxy image
            stamp = self.create_galaxy(entry['radius'],entry['ellipticity'],entry['sersic_index'],entry['pos_angle'],entry['counts_per_frame_e'])

            #convolve the galaxy with the NIRCam PSF
            stamp = s1.fftconvolve(stamp,psf,mode='same')
            
            #Now add the stamp to the main image
            #Extract the appropriate subarray from the galaxy image if necessary
            galdims = stamp.shape

            print('requested radius: {}  stamp size: {}'.format(entry['radius'],galdims))

            nyshift = galdims[0] / 2
            nxshift = galdims[1] / 2

            nx = int(entry['pixelx']+deltax)
            ny = int(entry['pixely']+deltay)
            i1 = max(nx-nxshift,0)
            i2 = min(nx+1+nxshift,dims[1])
            j1 = max(ny-nyshift,0)
            j2 = min(ny+1+nyshift,dims[0])
            k1 = nxshift-(nx-i1)
            k2 = nxshift+(i2-nx)
            l1 = nyshift-(ny-j1)
            l2 = nyshift+(j2-ny)

            #if the cutout for the psf is larger than
            #the psf array, truncate it, along with the array
            #in the source image where it will be placed
            if l2 > galdims[0]:
                l2 = galdims[0]
                j2 = j1 + (l2-l1)

            if k2 > galdims[1]:
                k2 = galdims[1]
                i2 = i1 + (k2-k1)

            #At this point coordinates are in the final output array coordinate system, so there
            #should be no negative values, nor values larger than the output array size
            if j1 < 0 or i1<0 or l1<0 or k1<0:
                print(j1,i1,l1,k1)
                print('bad low')
                #sys.exit()
            if j2>(dims[0]+1) or i2>(dims[1]+1) or l2>(galdims[1]+1) or k2>(galdims[1]+1):
                print(j2,i2,l2,k2)
                print('bad high')
                #sys.exit()

            #if j1 == 2949:
            #print(j1,j2,i1,i2,l1,l2,k1,k2)
            #print(entry)
            #print(galimage.shape)
            #print(stamp.shape)
            #print(ny,nyshift,dims)
            #print(entry['pixely'],deltay)
            #print(entry['RA'],entry['Dec'])
            
            #print(psfimage.shape,'imx',j1,j2,'imy',i1,i2,webbpsfimage.shape,'psfx',l1,l2,'psfy',k1,k2)
            #print(yoff,ny,nyshift)
            #if entry['pixelx'] > 2289.31 and entry['pixelx'] < 2289.32 and entry['pixely'] < -310.26 and entry['pixely'] > -310.27:
            #    print('input indexes, orig {},{}, adj for grism {} {}'.format(entry['pixelx'],entry['pixely'],entry['pixelx']+deltax,entry['pixely']+deltay))
            #    print('full indexes {},{} and {},{}.'.format(j1,j2,i1,i2))
            #    print('stamp indexes {},{}, and {},{}.'.format(l1,l2,k1,k2))

            if ((j2 > j1) and (i2 > i1) and (l2 > l1) and (k2 > k1) and (j1 < dims[0]) and (i1 < dims[0])):
                galimage[j1:j2,i1:i2] = galimage[j1:j2,i1:i2] + stamp[l1:l2,k1:k2]
            else:
                print("Source located entirely outside the field of view. Skipping.")

        return galimage


    def filterGalaxyList(self,galaxylist,pixelflag,radiusflag):
        #given a list of galaxies (location, size, orientation, magnitude)
        #keep only those which will fall fully or partially on the output array
        
        filteredList = Table(names=('pixelx','pixely','RA','Dec','RA_degrees','Dec_degrees','radius','ellipticity','pos_angle','sersic_index','magnitude','countrate_e/s','counts_per_frame_e'),dtype=('f','f','S14','S14','f','f','f','f','f','f','f','f','f'))

        #each entry in galaxylist is:
        #x_or_RA  y_or_Dec  radius  ellipticity  pos_angle  sersic_index  magnitude
        #remember that x/y are interpreted as coordinates in the output subarray
        #NOT full frame coordinates. This is the same as the point source list coords

        #First, begin to define the pixel limits beyond which a galaxy will be completely
        #outside of the field of view
        #First, coord limits for just the subarray
        miny = 0
        maxy = self.subarray_bounds[3] - self.subarray_bounds[1] 
        minx = 0
        maxx = self.subarray_bounds[2] - self.subarray_bounds[0] 
        ny = self.subarray_bounds[3] - self.subarray_bounds[1]
        nx = self.subarray_bounds[2] - self.subarray_bounds[0]
        print('before adjusting for grism, min/max x {} {}, min/max y {} {}'.format(minx,maxx,miny,maxy))
        
        #Expand the limits if a grism direct image is being made
        if self.params['Output']['grism_source_image'] == True:
            extrapixy = np.int((maxy+1)/2 * (self.grism_direct_factor - 1.))
            miny -= extrapixy
            maxy += extrapixy
            extrapixx = np.int((maxx+1)/2 * (self.grism_direct_factor - 1.))
            minx -= extrapixx
            maxx += extrapixx

            nx = np.int(nx * self.grism_direct_factor)
            ny = np.int(ny * self.grism_direct_factor)

        print('updated for grism, min/max x {} {}, min/max y {} {}'.format(minx,maxx,miny,maxy))
        #print('final update, to account for partially overlapping sources, will be done for each source individually.')

        #print(galaxylist)

            
        #Create transform matrix for galaxy sources
        #Read in the CRDS-format distortion reference file
        coord_transform = None
        if self.runStep['astrometric']:
            with AsdfFile.open(self.params['Reffiles']['astrometric']) as dist_file:
                coord_transform = dist_file.tree['model']

        #Using the requested RA,Dec of the reference pixel, along with the 
        #V2,V3 of the reference pixel, and the requested roll angle of the telescope
        #create a matrix that can be used to translate between V2,V3 and RA,Dec
        #for any pixel
        #v2,v3 need to be in arcsec, and RA, Dec, and roll all need to be in degrees
        attitude_matrix = rotations.attitude(self.refpix_pos['v2'],self.refpix_pos['v3'],self.ra,self.dec,self.params['Telescope']["rotation"])

        #Loop over galaxy sources
        for source in galaxylist:

            #If galaxy radii are given in units of arcseconds, translate to pixels 
            if radiusflag == False:
                source['radius'] /= self.pixscale[0]

            #how many pixels beyond the nominal subarray edges can a source be located and
            #still have it fall partially on the subarray? Galaxy stamps are nominally set to
            #have a length and width equal to 100 times the requested radius.
            edgex = source['radius'] * 100 / 2 - 1
            edgey = source['radius'] * 100 / 2 - 1

            #reset the field of view limits for the size of the current stamp image
            outminy = miny - edgey
            outmaxy = maxy + edgey
            outminx = minx - edgex
            outmaxx = maxx + edgex
        
            try:
                entry0 = float(source['x_or_RA'])
                entry1 = float(source['y_or_Dec'])
                if not pixelflag:
                    ra_str,dec_str = self.makePos(entry0,entry1)
                    #dec_str = self.makePos(entry0,entry1)
                    ra = entry0
                    dec = entry1
            except:
                #if inputs can't be converted to floats, then 
                #assume we have RA/Dec strings. Convert to floats.
                ra_str = source['x_or_RA']
                dec_str = source['y_or_Dec']
                ra,dec = self.parseRADec(ra_str,dec_str)

            #case where point source list entries are given with RA and Dec
            if not pixelflag:

                #if distortion is to be included
                if self.runStep['astrometric']:

                    pixelx,pixely = self.RADecToXY_astrometric(ra,dec,attitude_matrix,coord_transform)

                else:
                    #No distortion. Fall back to "manual" calculations
                    pixelx,pixely = self.RADecToXY_manual(ra,dec)
                    
            else:
                #case where the point source list entry locations are given in units of pixels
                #In this case we have the source position, and RA/Dec are calculated only so 
                #they can be written out into the output source list file.

                #Assume that the input x and y values are coordinate values
                #WITHIN THE SPECIFIED SUBARRAY. So for example, a source in the file
                #at 0,0 when you are making a SUB160 ramp will fall on the lower left
                #corner of the SUB160 subarray, NOT the lower left corner of the full
                #frame. 

                pixelx = entry0
                pixely = entry1

                ra,dec,ra_str,dec_str = self.XYToRADec(pixelx,pixely,attitude_matrix,coord_transform)


            #print(ra_str,dec_str)
            #print(outminy,outmaxy,outminx,outmaxx)
            #print(pixelx,pixely)
            #sys.exit()
            #only keep the source if the peak will fall within the subarray
            if pixely > outminy and pixely < outmaxy and pixelx > outminx and pixelx < outmaxx:

                entry = [pixelx,pixely,ra_str,dec_str,ra,dec,source['radius'],source['ellipticity'],source['pos_angle'],source['sersic_index']]

                #Now look at the input magnitude of the point source
                #append the mag and pixel position to the list of ra,dec
                mag = float(source['magnitude'])
                entry.append(mag)
    
                #translate magnitudes to countrate
                scale = 10.**(0.4*(15.0-mag))

                #get the countrate that corresponds to a 15th magnitude star for this filter
                cval = self.countvalues[self.params['Readout']['filter']]

                #DEAL WITH THIS LATER, ONCE PYSYNPHOT IS INCLUDED WITH PIPELINE DIST?
                if cval == 0:
                    print("Countrate value for {} is zero in {}.".format(self.params['Readout']['filter'],self.parameters['phot_file']))
                    print("Eventually attempting to calculate value using pysynphot.")
                    print("but pysynphot is not present in jwst build 6, so pushing off to later...")
                    sys.exit()
                    cval = self.findCountrate(self.params['Readout']['filter'])

                #translate to counts in single frame at requested array size
                framecounts = scale*cval*self.frametime
                rate = scale*cval

                #add the countrate and the counts per frame to pointSourceList
                #since they will be used in future calculations
                entry.append(rate)
                entry.append(framecounts)

                #add the good point source, including location and counts, to the pointSourceList
                filteredList.add_row(entry)


        #Write the results to a file
        filteredList.meta['comments'] = ["Field center (degrees): %13.8f %14.8f y axis rotation angle (degrees): %f  image size: %4.4d %4.4d\n" % (self.ra,self.dec,self.params['Telescope']['rotation'],nx,ny)]
        filteredList.write('galaxySources.list',format='ascii',overwrite=True)

        print(filteredList)

        return filteredList

        
    def create_galaxy(self,radius,ellipticity,sersic,posang,totalcounts):
        #given relevent parameters, create a model sersic image with a given radius, eccentricity, 
        #position angle, and total counts.

        #create the grid of pixels
        meshmax = np.min([np.int(self.ffsize*self.coord_adjust['y']),radius*100.])
        #meshmax = np.min([self.ffsize,radius*100.])
        x,y = np.meshgrid(np.arange(meshmax), np.arange(meshmax))

        #center the galaxy in the array
        xc = meshmax/2
        yc = meshmax/2
        
        #create model
        mod = Sersic2D(amplitude = 1,r_eff = radius, n=sersic, x_0=xc, y_0=yc, ellip=ellipticity, theta=posang)

        #create instance of model
        img = mod(x, y)

        #check to see if you've cropped too small and there is still significant signal
        #at the edges
        
        mxedge = np.max(np.array([np.max(img[:,-1]),np.max(img[:,0]),np.max(img[0,:]),np.max(img[-1,:])]))
        if mxedge > 0.001:
            print('Too small!')

        #scale such that the total number of counts in the galaxy matches the input
        summedcounts = np.sum(img)
        factor = totalcounts / summedcounts
        img = img * factor
        return img

    def readGalaxyFile(self,filename):
        # Read in the galaxy source list
        try:
            #read table
            gtab = ascii.read(filename,comment='#')

            #Look at the header lines to see if inputs
            #are in units of pixels or RA,Dec
            pflag = False
            rpflag = False
            try:
                if 'position_pixel' in gtab.meta['comments'][0]:
                    pflag = True
            except:
                pass
            try:
                if 'radius_pixel' in gtab.meta['comments'][1]:
                    rpflag = True
            except:
                pass

        except:
            print("WARNING: Unable to open the galaxy source list file {}".format(filename))
            sys.exit()

        return gtab,pflag,rpflag


    def readPointSourceFile(self,filename):
        # Read in the point source list
        try:
            #read table
            gtab = ascii.read(filename,comment='#')
            print(gtab)
            #Look at the header lines to see if inputs
            #are in units of pixels or RA,Dec
            pflag = False
            try:
                if 'position_pixel' in gtab.meta['comments'][0]:
                    pflag = True
            except:
                pass

        except:
            print("WARNING: Unable to open the point source list file {}".format(filename))
            sys.exit()

        return gtab,pflag


    def readMTFile(self,file):
        #read in moving target list file
        mtlist = ascii.read(file,comment='#')

        #check to see whether the position is in x,y or ra,dec
        pixelflag = False
        try:
            if 'position_pixels' in mtlist.meta['comments'][0]:
                pixelflag = True
        except:
            pass

        #if present, check whether the velocity entries are pix/sec
        #or arcsec/sec.
        pixelvelflag = False
        try:
            if 'velocity_pixels' in mtlist.meta['comments'][1]:
                pixelvelflag = True
        except:
            pass

        return mtlist,pixelflag,pixelvelflag


    def RADecToXY_astrometric(self,ra,dec,attitude_matrix,coord_transform):
        #Translate backwards, RA,Dec to V2,V3
        pixelv2,pixelv3 = rotations.getv2v3(attitude_matrix,ra,dec)

        if self.runStep['distortion_coeffs']:
            #If the full set of distortion coefficients are provided, then
            #use those to make the exact transformation from the 'ideal'
            #to 'science' coordinate systems

            #Now V2,V3 to undistorted angular distance from the reference pixel
            xidl = self.v2v32idlx(pixelv2-self.v2_ref,pixelv3-self.v3_ref)
            yidl = self.v2v32idly(pixelv2-self.v2_ref,pixelv3-self.v3_ref)
                                
            #Finally, undistorted distances to distorted pixel values
            deltapixelx, deltapixely, err, iter = polynomial.invert(self.x_sci2idl,self.y_sci2idl,xidl,yidl,5)

        else:
            #If the full set of distortion coefficients are not provided,
            #then we fall back to the coordinate transform provided by the
            #distortion reference file. These results are not exact, and
            #become less accurate the farther the source is from the center
            #of the detector. Results can be incorrect by ~20 pixels in the
            #corners of the detector.
    
            #Now go backwards from V2,V3 to distorted pixels
            deltapixelx,deltapixely = coord_transform.inverse(pixelv2-self.refpix_pos['v2'],pixelv3-self.refpix_pos['v3'])

        pixelx = deltapixelx + self.refpix_pos['x']
        pixely = deltapixely + self.refpix_pos['y']

        return pixelx,pixely


    def RADecToXY_manual(self,ra,dec):
        #In this case, the sources are provided as an RA,Dec list, 
        #but no astrometry information is provided. So assume an average
        #pixel scale and calculate the pixel position of the source from that.
        #This obviously does not include distortion, and is kind of a last
        #resort.
        ra_source = ra * 3600.
        dec_source = dec * 3600.
        
        dist_between,deltaang = self.dist([self.ra,self.dec],[ra_source,dec_source])

        #Now translate to deltax and deltay if the 
        #position angle is non-zero
        tot_ang = deltaang + (0. - self.params['Telescope']['rotation'] * np.pi / 180.)

        deltax = dist_between * np.sin(tot_ang) / self.pixscale[0]
        deltay = dist_between * np.cos(tot_ang) / self.pixscale[0]
                            
        pixelx = self.refpix_pos['x'] + deltax
        pixely = self.refpix_pos['y'] + deltay

        return pixelx,pixely


    def XYToRADec(self,pixelx,pixely,attitude_matrix,coord_transform):
        #Translate a given x,y location on the detector
        #to RA,Dec

        #If distortion is to be included
        #if self.runStep['astrometric']:
        if coord_transform is not None:
            #Transform distorted pixels to V2,V3
            deltav2,deltav3 = coord_transform(pixelx-self.refpix_pos['x'],pixely-self.refpix_pos['y'])
            pixelv2 = deltav2 + self.refpix_pos['v2']
            pixelv3 = deltav3 + self.refpix_pos['v3']

            #Now translate V2,V3 to RA,Dec
            ra,dec = rotations.pointing(attitude_matrix,pixelv2,pixelv3)

        else:
            #Without including distortion. 
            #Fall back to "manual" calculations
            dist_between = np.sqrt((pixelx-self.refpix_pos['x'])**2 + (pixely-self.refpix_pos['y'])**2)
            deltaang = np.arctan2(pixely,pixelx)

            tot_ang = deltaang + (self.parms['Telescope']['rotation'] * np.pi / 180.)

            deltara = dist_between * np.sin(tot_ang) / self.pixoscale[0]
            deltadec = dist_between * np.cos(tot_ang) / self.pixscale[0]
                            
            ra = self.ra + deltara
            dec = self.dec + deltadec
                            
        #Translate the RA/Dec floats to strings
        ra_str,dec_str = self.makePos(ra,dec)

        return ra,dec,ra_str,dec_str


    def readSubarrayDefinitionFile(self):
        #read in the file that contains a list of subarray names and positions on the detector
        try:
            self.subdict = ascii.read(self.params['Reffiles']['subarray_defs'],data_start=1,header_start=0)
        except:
            print("Error: could not read in subarray definitions file.")
            sys.exit()


    def readParameterFile(self):
        #read in the parameter file
        try:
            with open(self.paramfile,'r') as infile:
                self.params = yaml.load(infile)
        except:
            print("WARNING: unable to open {}".format(self.paramfile))
            sys.exit()
        

        #to make things easier to read, break up the yaml dictionary into 
        #a separate dictionary for each group

        #DOES IT MAKE SENSE TO DO THIS?
        #self.inst_params = self.params['Inst']
        #self.read_params = self.params['Readout']
        #self.reffiles = self.params['Reffiles']


    def instrument_specific_dicts(self,instrument):
        #get instrument-specific values for things that
        #don't need to be in the parameter file

        #array size of a full frame image
        self.ffsize = full_array_size[instrument]

        #pixel scale - return as a 2-element list, with pixscale for x and y.
        if instrument.lower() == 'nircam':
            filt = self.params['Readout']['filter']
            fnum = int(filt[1:4])
            if fnum < 230:
                channel = 'sw'
            else:
                channel = 'lw'
            self.pixscale = [pixelScale[instrument][channel],pixelScale[instrument][channel]]
        else:
            self.pixscale = [pixelScale[instrument],pixelScale[instrument]]

    def checkParams(self):
        #check instrument name
        if self.params['Inst']['instrument'].lower() not in inst_list:
            print("WARNING: {} instrument not implemented within ramp simulator")
            sys.exit()

        #check entred mode: 
        possibleModes = modes[self.params['Inst']['instrument'].lower()]
        self.params['Inst']['mode'] = self.params['Inst']['mode'].lower()
        if self.params['Inst']['mode'] in possibleModes:
            pass
        else:
            print("WARNING: unrecognized mode {} for {}. Must be one of: {}".format(self.params['Inst']['mode'],self.params['Inst']['instrument'],possibleModes))
            sys.exit()

        #make sure nframe,nskip,ngroup are all integers
        try:
            self.params['Readout']['nframe'] = int(self.params['Readout']['nframe'])
        except:
            print("WARNING: Input value of nframe is not an integer.")
            sys.exit

        try:
            self.params['Readout']['nskip'] = int(self.params['Readout']['nskip'])
        except:
            print("WARNING: Input value of nskip is not an integer.")
            sys.exit

        try:
            self.params['Readout']['ngroup'] = int(self.params['Readout']['ngroup'])
        except:
            print("WARNING: Input value of ngroup is not an integer.")
            sys.exit


        #check the number of amps. Full frame data will always be collected using 4 amps. 
        #Subarray data will always be 1 amp, except for the grism subarrays which span the
        #entire width of the detector. Those can be read out using 1 or 4 amps.
        

        #Make sure that the requested number of groups is less than or equal to the maximum
        #allowed. If you're continuing on with an unknown readout pattern (not recommended)
        #then assume a max of 10 groups.
        #For science operations, ngroup is going to be limited to 10 for all readout patterns
        #except for the DEEP patterns, which can go to 20.
        match = self.readpatterns['name'] == self.params['Readout']['readpatt'].upper()
        if sum(match) == 1:
            maxgroups = self.readpatterns['maxgroups'].data[match][0]
        if sum(match) == 0:
            print("Unrecognized readout pattern {}. Assuming a maximum allowed number of groups of 10.".format(self.params['Readout']['readpatt']))
            maxgroups = 10

        if (self.params['Readout']['ngroup'] > maxgroups):
            print("WARNING: {} is limited to a maximum of {} groups. Proceeding with ngroup = {}.".format(self.params['Readout']['readpatt'],maxgroups,maxgroups))
            self.params['Readout']['readpatt'] = maxgroups

        #check for entries in the parameter file that are None or blank,
        #indicating the step should be skipped. Create a dictionary of steps 
        #and populate with True or False
        self.runStep = {}
        self.runStep['superbias'] = self.checkRunStep(self.params['Reffiles']['superbias'])
        self.runStep['nonlin'] = self.checkRunStep(self.params['Reffiles']['linearity'])
        self.runStep['gain'] = self.checkRunStep(self.params['Reffiles']['gain'])
        self.runStep['phot'] = self.checkRunStep(self.params['Reffiles']['phot'])
        self.runStep['pixelflat'] = self.checkRunStep(self.params['Reffiles']['pixelflat'])
        self.runStep['illuminationflat'] = self.checkRunStep(self.params['Reffiles']['illumflat'])
        self.runStep['astrometric'] = self.checkRunStep(self.params['Reffiles']['astrometric'])
        self.runStep['distortion_coeffs'] = self.checkRunStep(self.params['Reffiles']['distortion_coeffs'])
        self.runStep['ipc'] = self.checkRunStep(self.params['Reffiles']['ipc'])
        self.runStep['crosstalk'] = self.checkRunStep(self.params['Reffiles']['crosstalk'])
        self.runStep['occult'] = self.checkRunStep(self.params['Reffiles']['occult'])
        self.runStep['pointsource'] = self.checkRunStep(self.params['simSignals']['pointsource'])
        self.runStep['galaxies'] = self.checkRunStep(self.params['simSignals']['galaxyListFile'])
        self.runStep['extendedsource'] = self.checkRunStep(self.params['simSignals']['extended'])
        self.runStep['movingTargets'] = self.checkRunStep(self.params['simSignals']['movingTargetList'])
        self.runStep['movingTargetsExtended'] = self.checkRunStep(self.params['simSignals']['movingTargetExtended'])
        self.runStep['zodiacal'] = self.checkRunStep(self.params['simSignals']['zodiacal'])
        self.runStep['scattered'] = self.checkRunStep(self.params['simSignals']['scattered'])
        self.runStep['linearity'] = self.checkRunStep(self.params['Reffiles']['linearity'])
        self.runStep['cosmicray'] = self.checkRunStep(self.params['cosmicRay']['path'])
        self.runStep['saturation_lin_limit'] = self.checkRunStep(self.params['Reffiles']['saturation'])
        self.runStep['fwpw'] = self.checkRunStep(self.params['Reffiles']['filtpupilcombo'])
        self.runStep['linearized_darkfile'] = self.checkRunStep(self.params['newRamp']['linearized_darkfile'])
        self.runStep['hotpixfile'] = self.checkRunStep(self.params['Reffiles']['hotpixmask'])
        self.runStep['pixelAreaMap'] = self.checkRunStep(self.params['Reffiles']['pixelAreaMap'])

        #create table that will contain filters/quantum yield/and vegamag=15 countrates
        self.makeFilterTable()

        #make sure the requested filter is allowed. For imaging, all filters are allowed.
        #In the future, other modes will be more restrictive
        if self.params['Readout']['filter'] not in self.qydict:
            print("WARNING: requested filter {} is not in the list of possible filters.".format(self.params['Readout']['filter']))
            sys.exit()


        #COSMIC RAYS: 
        #generate the name of the actual CR file to use
        if self.params['cosmicRay']['path'] is None:
            self.crfile = None
        else:
            if self.params['cosmicRay']['path'][-1] != '/':
                self.params['cosmicRay']['path'] += '/'
            if self.params['cosmicRay']["library"].upper() in ["SUNMAX","SUNMIN","FLARES"]:
                self.crfile=self.params['cosmicRay']['path'] + "CRs_MCD1.7_"+self.params['cosmicRay']["library"].upper()
            else:
                self.crfile=None
                print("Warning: unrecognised cosmic ray library {}".format(self.params['cosmicRay']["library"]))
                sys.exit()


        #PSF: generate the name of the PSF file to use
        #if the psf path has been left blank or set to 'None'
        #then assume the user does not want to add point sources
        if self.params['simSignals']['psfpath'] is not None:
            if self.params['simSignals']['psfpath'][-1] != '/':
                self.params['simSignals']['psfpath']=self.params['simSignals']['psfpath']+'/'

            wfe = self.params['simSignals']['psfwfe']
            wfegroup = self.params['simSignals']['psfwfegroup']
            basename = self.params['simSignals']['psfbasename'] + '_'
            if wfe == 0:
                psfname=basename+self.params['simSignals']["filter"].lower()+'_zero'
                self.params['simSignals']['psfpath']=self.params['simSignals']['psfpath']+self.params['simSignals']['filter'].lower()+'/zero/'
            else:
                if wfe in [123,136,155] and wfegroup > -1 and wfegroup < 10:
                    psfname=basename+self.params['Readout']['filter'].lower()+"_"+str(wfe)+"_"+str(wfegroup)
                    self.params['simSignals']['psfpath']=self.params['simSignals']['psfpath']+self.params['Readout']['filter'].lower()+'/'+str(wfe)+'/'
            self.psfname = self.params['simSignals']['psfpath'] + psfname

        else:
            #case where psfPath is None. In this case, create a PSF on the fly to use
            #for adding sources
            print("update this to include a call to WebbPSF????????")
            self.psfimage=np.zeros((5,5),dtype=np.float32)
            sum1=0
            for i in range(5):
                for j in range(5):
                    self.psfimage[i,j]=(0.02**(abs((i-2)))*(0.02**abs(j-2)))
                    sum1=sum1+self.psfimage[i,j]
            self.psfimage=self.psfimage/sum1
            self.psfname = None
                

        #NON-LINEARITY
        #make sure the input accuracy is a float with reasonable bounds
        self.params['nonlin']['accuracy'] = self.checkParamVal(self.params['nonlin']['accuracy'],'nlin accuracy',1e-12,1e-6,1e-6)
        self.params['nonlin']['maxiter'] = self.checkParamVal(self.params['nonlin']['maxiter'],'nonlin max iterations',5,40,10)
        self.params['nonlin']['limit'] = self.checkParamVal(self.params['nonlin']['limit'],'nonlin max value',30000.,1.e6,66000.)
    
        #Combining the base dark ramp and the simulated signal ramp
        comb_types = ['STANDARD','HIGHSIG','PROPER']
        if self.params['newRamp']['combine_method'].upper() not in comb_types:
            print("WARNING: unrecognized method for combining the dark and simulated signal: {}.".format(self.params['newRamp']['combine_method']))
            print("Acceptible values are {}".format(comb_types))
            sys.exit()

        self.runStep['better_combine'] = False
        if self.params['newRamp']['combine_method'].upper() != 'STANDARD':
            self.runStep['better_combine'] = True

        #If the use of the better combination method is requested, check what type of pixels it will be used on.
        better_method = ['HOTPIX','COSMICRAYS','BOTH']
        self.runStep['better_combine'] = False
        if self.params['newRamp']['combine_method'].upper() != 'STANDARD' and self.params['newRamp']['proper_combine'].upper() not in better_method:
            print('WARNING: set of pixels on which to use the proper combination technique is not defined {}.'.format(self.params['newRamp']['proper_combine']))
            print("Acceptible values are {}".format(better_method))
            sys.exit()

            
        #If the pipeline is going to be used to create the linearized dark current ramp, make sure the specified 
        #configuration files are present.
        if self.runStep['better_combine'] and self.runStep['linearized_dark'] == False:
            dqcheck = self.checkRunStep(self.params['newRamp']['dq_configfile'])
            if dqcheck == False:
                print("WARNING: DQ pipeline step configuration file not provided. This file is needed to run the pipeline.")
                sys.exit()
            satcheck = self.checkRunStep(self.params['newRamp']['sat_configfile'])
            if satcheck == False:
                print("WARNING: Saturation pipeline step configuration file not provided. This file is needed to run the pipeline.")
                sys.exit()
            sbcheck = self.checkRunStep(self.params['newRamp']['superbias_configfile'])
            if sbcheck == False:
                print("WARNING: Superbias pipeline step configuration file not provided. This file is needed to run the pipeline.")
                sys.exit()
            refpixcheck = self.checkRunStep(self.params['newRamp']['refpix_configfile'])
            if refpixcheck == False:
                print("WARNING: Refpix pipeline step configuration file not provided. This file is needed to run the pipeline.")
                sys.exit()
            lincheck = self.checkRunStep(self.params['newRamp']['lin_configfile'])
            if lincheck == False:
                print("WARNING: Linearity pipeline step configuration file not provided. This file is needed to run the pipeline.")
                sys.exit()
            


        #make sure the CR random number seed is an integer
        try:
            self.params['cosmicRay']['seed'] = int(self.params['cosmicRay']['seed'])
        except:
            self.params['cosmicRay']['seed'] = 66231289
            print("ERROR: cosmic ray random number generator seed is bad. Using the default value of {}.".format(self.params['cosmicRay']['seed']))
         

        #also make sure the poisson random number seed is an integer
        try:
            self.params['simSignals']['poissonseed'] = int(self.params['simSignals']['poissonseed'])
        except:
            self.params['simSignals']['poissonseed'] = 815813492
            print("ERROR: cosmic ray random number generator seed is bad. Using the default value of {}.".format(self.params['simSignals']['poissonseed']))
         

        #ASTROMETRY
        #Read in the distortion coefficients file if present. These will provide a more exact
        #transform from RA,Dec to x,y than the astrometric distortion reference file above.
        #The file above can be off by ~20 pixels in the corners of the array. This file will give
        #exact answers
        if self.runStep['distortion_coeffs'] == True:
            if os.path.isfile(self.params['Reffiles']['distortion_coeffs']):
                distortionTable = ascii.read(self.params['Reffiles']['distortion_coeffs'],header_start=1)
            else:
                print("WARNING: Input distortion coefficients file {} does not exist.".format(self.params['Reffile']['distortion_coeffs']))
                sys.exit()

            #read in coefficients for the forward 'science' to 'ideal' coordinate transformation.
            #'science' is in units of distorted pixels, while 'ideal' is the undistorted
            #angular distance from the reference pixel
            #ap_name = inst_abbrev[self.params['Inst']['instrument'].lower()] + self.params['Inst']['detector'].upper() + '_' + self.params['Readout']['array_name'].upper()
            ap_name = self.params['Readout']['array_name']

            self.x_sci2idl,self.y_sci2idl,self.v2_ref,self.v3_ref = self.getDistortionCoefficients(distortionTable,'science','ideal',ap_name)
            
            #Generate the coordinate transform for V2,V3 to 'ideal'
            self.v2v32idlx, self.v2v32idly = read_siaf_table.get_siaf_v2v3_transform(self.params['Reffiles']['distortion_coeffs'],ap_name,to_system='ideal')

            
        #convert the input RA and Dec of the pointing position into floats
        #check to see if the inputs are in decimal units or hh:mm:ss strings
        try:
            self.ra = float(self.params['Telescope']['ra'])
            self.dec = float(self.params['Telescope']['dec'])
        except:
            self.ra,self.dec=self.parseRADec(self.params['Telescope']['ra'],self.params['Telescope']['dec'])

        if abs(self.dec) > 90. or self.ra < 0. or self.ra > 360. or self.ra is None or self.dec is None:
            print("WARNING: bad requested RA and Dec {} {}".format(self.ra,self.dec))
            sys.exit()

        #make sure the rotation angle is a float
        try:
            self.params['Telescope']["rotation"]=float(self.params['Telescope']["rotation"])
        except:
            print("ERROR: bad rotation value {}, setting to zero.".format(self.params['Telescope']["rotation"]))
            self.params['Telescope']["rotation"]=0.


        #check that the various scaling factors are floats and within a reasonable range
        self.params['cosmicRay']['scale'] = self.checkParamVal(self.params['cosmicRay']['scale'],'cosmicRay',0,100,1)
        self.params['simSignals']['extendedscale'] = self.checkParamVal(self.params['simSignals']['extendedscale'],'extendedEmission',0,10000,1)
        self.params['simSignals']['zodiscale'] = self.checkParamVal(self.params['simSignals']['zodiscale'],'zodi',0,10000,1)
        self.params['simSignals']['scatteredscale'] = self.checkParamVal(self.params['simSignals']['scatteredscale'],'scatteredLight',0,10000,1)


        #check the hot pixel signal limit. Defulat to 5000 if the value in the parameter file is bad.
        self.params['newRamp']['proper_signal_limit'] = self.checkParamVal(self.params['newRamp']['proper_signal_limit'],'signal limit for proper combine',0.,1.e6,5000.)

        #make sure the requested output format is an allowed value
        if self.params['Output']['format'] not in allowedOutputFormats:
            print("WARNING: unsupported output format {} requested. Possible options are {}.".format(self.params['Output']['format'],allowedOutputFormats))
            sys.exit()

        #Entries for creating the grims input image
        if not isinstance(self.params['Output']['grism_source_image'],bool):
            if self.params['Output']['grism_source_image'].lower() == 'none':
                self.params['Output']['grism_source_image'] = False
            else:
                print("WARNING: grism_source_image needs to be True or False")
                sys.exit()

        if not isinstance(self.params['Output']['grism_input_only'],bool):
            if self.params['Output']['grism_source_image'].lower() == 'none':
                self.params['Output']['grism_input_only'] = False
            else:
                print("WARNING: grism_input_only needs to be True or False")
                sys.exit()

        #Location of extended image on output array, pixel x,y values.
        try:
            self.params['simSignals']['extendedCenter'] = np.fromstring(self.params['simSignals']['extendedCenter'], dtype=int, sep=",")
        except:
            print("WARNING: not able to parse the extendedCenter list {}. It should be a comma-separated list of x and y pixel positions.".format(self.params['simSignals']['extendedCenter']))
            sys.exit()

        #Time series settings
        if self.params['Inst']['mode'] == 'tso':
            
            #make sure slew rate and angle are floats
            try:
                self.params['Telescope']['slewRate'] = np.float(self.params['Telescope']['slewRate'])
            except:
                print("WARNING: input slew rate {} is not an integer or float.".format(self.params['Telescope']['slewRate']))
                sys.exit()

            try:
                self.params['Telescope']['slewAngle'] = np.float(self.params['Telescope']['slewAngle'])
            except:
                print("WARNING: input slew angle {} is not an integer or float.".format(self.params['Telescope']['slewAngle']))
                sys.exit()

                
    def getDistortionCoefficients(self,table,from_sys,to_sys,aperture):
        '''from the table of distortion coefficients, get the coeffs that correspond
        to the requested transformation and return as a list for x and another for y
        '''
        match = table['AperName'] == aperture
        if np.any(match) == False:
            print("Aperture name {} not found in input CSV file.".format(aperture))
            sys.exit()

        row = table[match]

        if ((from_sys == 'science') & (to_sys == 'ideal')):
            label = 'Sci2Idl'
        elif ((from_sys == 'ideal') & (to_sys == 'science')):
            label = 'Idl2Sci'
        else:
            print("WARNING: from_sys of {} and to_sys of {} not a valid transformation.".format(from_sys,to_sys))
            sys.exit()
        
        #get the coefficients, return as list
        X_cols = [c for c in row.colnames if label+'X' in c]
        Y_cols = [c for c in row.colnames if label+'Y' in c]
        x_coeffs = [row[c].data[0] for c in X_cols]
        y_coeffs = [row[c].data[0] for c in Y_cols]

        #Also get the V2,V3 values of the reference pixel
        v2ref = row['V2Ref'].data[0]
        v3ref = row['V3Ref'].data[0]

        return x_coeffs,y_coeffs,v2ref,v3ref
                

    def getCRrate(self):
        #get the base cosmic ray impact probability
        #these numbers are from Kevin's original script. Not sure where
        #they come from or their units
        self.crrate=0.
        if "SUNMAX" in self.params["cosmicRay"]["library"]:
            self.crrate=1.6955e-04
        if "SUNMIN" in self.params["cosmicRay"]["library"]:
            self.crrate=6.153e-05
        if "FLARES" in self.params["cosmicRay"]["library"]:
            self.crrate=0.10546

        self.crrate=self.crrate/self.frametime
        #print('self.crrate is {}'.format(self.crrate))
        if self.crrate > 0.:
            print("Base cosmic ray probability per pixel per second: {}".format(self.crrate))

            
    def parseRADec(self,rastr,decstr):
        #convert the input RA and Dec strings to floats
        try:
            rastr=rastr.lower()
            rastr=rastr.replace("h",":")
            rastr=rastr.replace("m",":")
            rastr=rastr.replace("s","")
            decstr=decstr.lower()
            decstr=decstr.replace("d",":")
            decstr=decstr.replace("m",":")
            decstr=decstr.replace("s","")

            values=rastr.split(":")
            ra0=15.*(int(values[0])+int(values[1])/60.+float(values[2])/3600.)

            values=decstr.split(":")
            if "-" in values[0]:
                sign=-1
                values[0]=values[0].replace("-"," ")
            else:
                sign=+1
            dec0=sign*(int(values[0])+int(values[1])/60.+float(values[2])/3600.)
            return ra0,dec0
        except:
            print("Error parsing RA,Dec strings: {} {}".format(rastr,decstr))
            sys.exit()
            #return None,None


    def checkRunStep(self,filename):
        #check to see if a filename exists in the parameter file.
        if ((len(filename) == 0) or (filename.lower() == 'none')):
            return False
        else:
            return True

    def readpatternCheck(self):
        '''check the readout pattern that's entered and set nframe and nskip 
           accordingly'''
        self.params['Readout']['readpatt'] = self.params['Readout']['readpatt'].upper()

        #read in readout pattern definition file 
        #and make sure the possible readout patterns are in upper case
        self.readpatterns = ascii.read(self.params['Reffiles']['readpattdefs'])
        self.readpatterns['name'] = [s.upper() for s in self.readpatterns['name']]

        #if the requested readout pattern is in the table of options,
        #then adopt the appropriate nframe and nskip
        if self.params['Readout']['readpatt'] in self.readpatterns['name']:
            mtch = self.params['Readout']['readpatt'] == self.readpatterns['name']
            self.params['Readout']['nframe'] = self.readpatterns['nframe'][mtch].data[0]
            self.params['Readout']['nskip'] = self.readpatterns['nskip'][mtch].data[0]
            print('Requested readout pattern {} found in {}. Using the nframe and nskip'.format(self.params['Readout']['readpatt'],self.params['Reffiles']['readpattdefs']))
            print('values from the definition file. nframe = {} and nskip = {}.'.format(self.params['Readout']['nframe'],self.params['Readout']['nskip']))
        else:
            #if readpatt is not present in the definition file but the nframe/nskip combo is, then reset 
            #readpatt to the appropriate value from the definition file
            readpatt_nframe = self.readpatterns['nframe'].data
            readpatt_nskip = self.readpatterns['nskip'].data
            readpatt_name = self.readpatterns['name'].data
            if self.params['Readout']['nframe'] in readpatt_nframe:
                nfmtch = self.params['Readout']['nframe'] == readpatt_nframe
                nskip_subset = readpatt_nskip[nfmtch]
                name_subset = readpatt_name[nfmtch]
                if self.params['Readout']['nskip'] in nskip_subset:
                    finalmtch = self.params['Readout']['nskip'] == nskip_subset
                    finalname = name_subset[finalmtch][0]
                    print("CAUTION: requested readout pattern {} not recognized.".format(self.params['Readout']['readpatt']))
                    print('but the requested nframe/nskip combination ({},{}), matches those values for'.format(self.params['Readout']['nframe'],self.params['Readout']['nskip']))
                    print("the {} readout pattern, as listed in {}.".format(finalname,self.params['Reffiles']['readpattdefs']))
                    print('Continuing on using that as the readout pattern.')
                    self.params['Readout']['readpatt'] = finalname
                else:
                    #case where readpatt is not recognized, nframe is present in the definition file, but the nframe/nskip combination is not
                    print('Unrecognized readout pattern {}, and the input nframe/nskip combination {},{} does not'.format(self.params['Readout']['readpatt'],self.params['Readout']['nframe'],self.params['Readout']['nskip']))
                    print('match any present in {}. Continuing simulation with the input nframe/nskip, and'.format(self.params['Reffiles']['readpattdefs']))
                    print("setting the readout pattern to 'ANY' in order to allow the output file to be saved via RampModel without error.")
                    self.params['readpatt'] = 'ANY'
            else:
                #case where readpatt is not recognized, and nframe is not present in the definition file
                print('Unrecognized readout pattern {}, and the input nframe/nskip combination {},{} does not'.format(self.params['Readout']['readpatt'],self.params['Readout']['nframe'],self.params['Readout']['nskip']))
                print('match any present in {}. '.format(self.params['Reffiles']['readpattdefs']))
                print('Continuing simulation with the input nframe/nskip values, and')
                print("setting the readout pattern to 'ANY' in order to allow the output file to be saved via RampModel without error.")
                self.params['readpatt'] = 'ANY'

        
    def setNumAmps(self):
        #set the number of amps used in the output data. Full frame data always use 4 amps. Suabrray data
        #generally use 1 amp, except for the grism-related subarrays that span the entire width
        #of the detector. For those, trust that the user input is what they want.
        if "FULL" in self.params['Readout']['array_name'].upper():
            self.params['Readout']['namp'] = 4
        else:
            if self.subarray_bounds[2]-self.subarray_bounds[0] != 2047:
                self.params['Readout']['namp'] = 1


    def getSubarrayBounds(self):
        #find the bounds of the requested subarray
        if self.params['Readout']['array_name'] in self.subdict['AperName']:
            mtch = self.params['Readout']['array_name'] == self.subdict['AperName']
            self.subarray_bounds = [self.subdict['xstart'].data[mtch][0],self.subdict['ystart'].data[mtch][0],self.subdict['xend'].data[mtch][0],self.subdict['yend'].data[mtch][0]]
            self.refpix_pos = {'x':self.subdict['refpix_x'].data[mtch][0],'y':self.subdict['refpix_y'][mtch][0],'v2':self.subdict['refpix_v2'].data[mtch][0],'v3':self.subdict['refpix_v3'].data[mtch][0]}

            namps = self.subdict['num_amps'].data[mtch][0]
            if namps != 0:
                self.params['Readout']['namp'] = namps
            else:
                if ((self.params['Readout']['namp'] == 1) or (self.params['Readout']['namp'] == 4)):
                    print("CAUTION: Aperture {} can be used with either a 1-amp".format(self.subdict['AperName'].data[mtch][0]))
                    print("or a 4-amp readout. The difference is a factor of 4 in")
                    print("readout time. You have requested {} amps.".format(self.params['Readout']['namp']))
                else:
                    print("WARNING: {} requires the number of amps to be 1 or 4. You have requested {}.".format(self.params['Readout']['array_name'],self.params['Readout']['namp']))
                    sys.exit()


        else:
            print("WARNING: subarray name {} not found in the subarray dictionary {}.".format(self.params['Readout']['array_name'],self.params['Reffiles']['subarray_defs']))
            sys.exit()

        #if ',' in self.params['Readout']['subarray_bounds']:
        #    try:
        #        self.subaray_bounds = [int(s) for s in self.params['Readout']['subarray_bounds'].split(',')]
        #        if (len(self.subarray_bounds) != 4):
        #            print("WARNING: incorrect number of values in subarray_bounds. Expecting 4 values separated by commas.")
        #            sys.exit()
        

    def calcFrameTime(self):
        #calculate the exposure time of a single frame of the proposed output ramp
        #based on the size of the croped dark current integration
        numint,numgrp,yd,xd = self.dark.data.shape

        self.frametime = (yd+1)*(int(xd/self.params['Readout']['namp']+0.01)+12)/100000.

        if yd < 65 and xd < 65:
            if self.params['Readout']['namp'] == 1:
                self.frametime = (yd+1)*(xd+6)/100000.

        # if the read-out is a sub-array, add the rolling line resets time (nresetlines times 
        # the read time of 10 microseconds)...
        if yd < 2048 and xd < 2048:
            self.frametime = self.frametime+self.params['Inst']['nresetlines'] / 100000.


    def checkParamVal(self,value,typ,vmin,vmax,default):
        #make sure the input value is a float and between min and max
        try:
            value = float(value)
        except:
            print("WARNING: {} for {} is not a float.".format(value,typ))
            sys.exit()
            
        if ((value >= vmin) & (value <= vmax)):
            return value
        else:
            print("ERROR: {} for {} is not within reasonable bounds. Setting to {}".format(value,typ,default))
            return default

    def make_param(self):
        #output an example parameter file
        pfile = 'example_parameter_file.yaml'
        with open(pfile,'w') as f:
            f.write('Inst:\n')
            f.write('  instrument: NIRCam          #Instrument name\n')
            f.write('  mode: imaging                #Observation mode (e.g. imaging, WFSS, TSO)\n')
            f.write('  nresetlines: 512                        #eventially use dictionary w/in code to look this up\n')
            f.write('\n')
            f.write('Readout:\n')
            f.write('  readpatt: RAPID        #Readout pattern (RAPID, BRIGHT2, etc) overrides nframe,nskip unless it is not recognized\n')
            f.write('  nframe: 1        #Number of frames per group\n')
            f.write('  nskip: 0         #Number of skipped frames between groups\n')
            f.write('  ngroup: 5              #Number of groups in integration\n')
            f.write('  namp: 4                               #Number of amplifiers used in readout (4 for full frame, 1 for subarray)\n')
            f.write('  array_name: NRCA1_FULL    #Name of array (FULL, SUB160, SUB64P, etc) overrides subarray_bounds below\n')
            f.write('  subarray_bounds: 0, 0, 159, 159          #Coords of subarray corners. (xstart, ystart, xend, yend) Over-ridden by array_name above. Currently not used. Could be used if output saved in raw format\n')
            f.write('  filter: F090W                 #Filter of simulated data (F090W, F322W2, etc)\n')
            f.write('  pupil: CLEAR                   #Pupil element for simulated data (CLEAR, GRISMC, etc)\n')
            f.write('\n')
            f.write('Reffiles:                                 #Set to None or leave blank if you wish to skip that step\n')
            f.write('  dark: NRCNRCA1-DARK-60091434481_1_481_SE_2016-01-09T15h50m45_uncal.fits               #Dark current integration used as the base\n')
            f.write('  hotpixmask: None                        #Hot pixel mask to go with the dark integration. If none, the script will find hot pixels. Fits file. Ones are hot. Zeros not.\n')
            f.write('  superbias: A1_superbias_from_list_of_biasfiles.list.fits     #Superbias file. Set to None or leave blank if not using\n')
            f.write('  subarray_defs: NIRCam_subarray_definitions.list                #File that contains a list of all possible subarray names and coordinates\n')
            f.write('  readpattdefs: nircam_read_pattern_definitions.list           #File that contains a list of all possible readout pattern names and associated NFRAME/NSKIP values\n')
            f.write('  linearity: NRCA1_17004_LinearityCoeff_ADU0_2016-05-14_ssblinearity_DMSorient.fits     #linearity correction coefficients\n')
            f.write('  saturation: NRCA1_17004_WellDepthADU_2016-03-10_ssbsaturation_DMSorient.fits      #well depth reference files\n')
            f.write('  gain: NRCA1_17004_Gain_ISIMCV3_2016-01-23_ssbgain_DMSorient.fits                    #Gain map\n')
            f.write('  phot: nircam_mag15_countrates.list                  #File with list of all filters and associated quantum yield values and countrates for mag 15 star\n')
            f.write('  pixelflat: None \n')
            f.write('  illumflat: None                               #Illumination flat field file\n')
            f.write('  astrometric: NRCA1_FULL_distortion.asdf             #Astrometric distortion file (asdf)\n')
            f.write('  distortion_coeffs: NIRCam_SIAF_2016-09-29.csv           #CSV file containing distortion coefficients\n')
            f.write('  ipc: NRCA1_17004_IPCDeconvolutionKernel_2016-03-18_ssbipc_DMSorient.fits                        #File containing IPC kernel to apply\n')
            f.write('  invertIPC: True       #Invert the IPC kernel before the convolution. True or False. Use True if the kernel is designed for the removal of IPC effects, like the JWST reference files are.\n')
            f.write('  crosstalk: xtalk20150303g0.errorcut.txt              #File containing crosstalk coefficients\n')
            f.write('  occult: None                                    #Occulting spots correction image\n')
            f.write('  filtpupilcombo: nircam_filter_pupil_pairings.list       #File that lists the filter wheel element / pupil wheel element combinations. Used only in writing output file\n')
            f.write('  pixelAreaMap: jwst_nircam_area_0001.fits              #Pixel area map for the detector. Used to introduce distortion into the output ramp.\n')
            f.write('\n')
            f.write('nonlin:\n')
            f.write('  limit: 60000.0                           #Upper singal limit to which nonlinearity is applied (ADU)\n')
            f.write('  accuracy: 0.000001                        #Non-linearity accuracy threshold\n')
            f.write('  maxiter: 10                              #Maximum number of iterations to use when applying non-linearity\n')
            f.write('  robberto:  False                         #Use Massimo Robberto type non-linearity coefficients\n')
            f.write('\n')
            f.write('\n')
            f.write('cosmicRay:\n')
            f.write('  path: /User/myself/cosmic_ray_library/               #Path to CR library\n')
            f.write('  library: SUNMIN                   #Type of cosmic rayenvironment (SUNMAX, SUNMIN, FLARE)\n')
            f.write('  scale: 1.5                                 #Cosmic ray scaling factor\n')
            f.write('  seed: 973415286                            #Seed for random number generator\n')
            f.write('\n')
            f.write('simSignals:\n')
            f.write('  pointsource: stars_input_radec_for_mosaic.list   #File containing a list of point sources to add (x,y locations and magnitudes)\n')
            f.write('  psfpath: /User/myself/psf_files/        #Path to PSF library\n')
            f.write('  psfbasename: nircam                        #Basename of the files in the psf library\n')
            f.write('  psfpixfrac: 0.1                           #Fraction of a pixel between entries in PSF library (e.g. 0.1 = files for PSF centered at 0.1 pixel intervals within pixel)\n')
            f.write('  psfwfe: 155                               #PSF WFE value (0,115,123,132,136,150,155)\n')
            f.write('  psfwfegroup: 0                             #WFE realization group (0 to 9)\n')
            f.write('  galaxyListFile: galaxies_for_mosaic.list    #File containing a list of positions/ellipticities/magnitudes of galaxies to simulate\n')
            f.write('  extended: None                             #Extended emission count rate image file name\n')
            f.write('  extendedscale: 2.0                          #Scaling factor for extended emission image\n')
            f.write('  extendedCenter: 1024,1024                   #x,y pixel location at which to place the extended image if it is smaller than the output array size\n')
            f.write('  PSFConvolveExtended: True #Convolve the extended image with the PSF before adding to the output image (True or False)\n')
            f.write('  movingTargetList: None                   #Name of file containing a list of point source moving targets (e.g. KBOs, asteroids) to add.\n')
            f.write('  movingTargetExtended: None               #ascii file containing a list of stamp images to add as moving targets (planets, moons, etc)\n')
            f.write('  movingTargetConvolveExtended: True       #convolve the extended moving targets with PSF before adding.\n')
            f.write('  zodiacal:  None                          #Zodiacal light count rate image file \n')
            f.write('  zodiscale:  1.0                            #Zodi scaling factor\n')
            f.write('  scattered:  None                          #Scattered light count rate image file\n')
            f.write('  scatteredscale: 1.0                        #Scattered light scaling factor\n')
            f.write('  bkgdrate: 10.001                          #Constant background count rate (electrons/sec/pixel)\n')
            f.write('  poissonseed: 3124580697                   #Random number generator seed for Poisson simulation)\n')
            f.write('  photonyield: True                         #Apply photon yield in simulation\n')
            f.write('  pymethod: True                            #Use double Poisson simulation for photon yield\n')
            f.write('\n')
            f.write('Telescope:\n')
            f.write('  ra: "00:00:00.00"                      #RA of simulated pointing\n')
            f.write('  dec: "00:00:00.00"                    #Dec of simulated pointing\n')
            f.write('  rotation: 0.0                    #y axis rotation (degrees E of N)\n')
            f.write('  slewRate: 10.0                             #arcsec/sec, Used only for time series observations (TSO mode)\n')
            f.write('  slewAngle: 10.0                            #degrees anti-clockwise from vertical?\n')
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
            f.write('  file: sim_A1_F090W_pointing1_mosaic.fits    #Output filename\n')
            f.write('  format: DMS                            #Output file format Options: DMS, SSR(not yet implemented)\n')
            f.write('  save_intermediates: True               #Save intermediate products separately (point source image, etc)\n')
            f.write('  grism_source_image: False               #grism\n')
            f.write('  grism_input_only: False                  #grism\n')
            f.write('  unsigned: True                         #Output unsigned integers? (0-65535 if true. -32768 to 32768 if false)\n')
            f.write('  dmsOrient: True                        #Output in DMS orientation (vs. fitswriter orientation).\n')
        print('Example parameter file written to {}.'.format(pfile))


    def make_input_examples(self):
        #output example files for subarray definition file, readout pattern file, point source list

        from itertools import izip

        #subarray definition file
        subfile = 'example_nircam_subarray_definitions.list'
        with open(subfile,'w') as f:
            f.write('# Definitions of available NIRCam subarrays.\n')
            f.write('# x and y starting and ending coordinates, and reference pixel coordinates\n')
            f.write('# are 0-indexed.\n')
            f.write('#    \n')
            f.write('AperName Name Detector Filter xstart ystart xend yend num_amps refpix_x refpix_y refpix_v2 refpix_v3\n')
            f.write('NRCA1_FULL FULL A1 ANY 0 0 2047 2047 4 1023.5 1023.5 120.6714 -527.3877\n')
            f.write('NRCA2_FULL FULL A2 ANY 0 0 2047 2047 4 1023.5 1023.5 120.1121 -459.6806\n')
            f.write('NRCA3_FULL FULL A3 ANY 0 0 2047 2047 4 1023.5 1023.5 51.9345 -527.8034\n')
            f.write('NRCA4_FULL FULL A4 ANY 0 0 2047 2047 4 1023.5 1023.5 52.2768 -459.8097\n')
            f.write('NRCA5_FULL FULL A5 ANY 0 0 2047 2047 4 1023.5 1023.5 86.1035 -493.2275\n')
            f.write('NRCB1_FULL FULL B1 ANY 0 0 2047 2047 4 1023.5 1023.5 -120.9682 -457.7527\n')
            f.write('NRCB2_FULL FULL B2 ANY 0 0 2047 2047 4 1023.5 1023.5 -121.1443 -525.4582\n')
            f.write('NRCB3_FULL FULL B3 ANY 0 0 2047 2047 4 1023.5 1023.5 -53.1238 -457.7804\n')
            f.write('NRCB4_FULL FULL B4 ANY 0 0 2047 2047 4 1023.5 1023.5 -52.8182 -525.7273\n')
            f.write('NRCB5_FULL FULL B5 ANY 0 0 2047 2047 4 1023.5 1023.5 -89.3892 -491.444\n')
            f.write('NRCA5_GRISM_F277W FULL B5 F277W 0 0 2047 2047 4 1581.0 484.0 50.8765 -527.4932\n')
            f.write('NRCA5_GRISM_F322W2 FULL B5 F322W2 0 0 2047 2047 4 1581.0 484.0 50.8765 -527.4932\n')
            f.write('NRCA5_GRISM_F356W FULL B5 F356W 0 0 2047 2047 4 1581.0 484.0 50.8765 -527.4932\n')
            f.write('NRCA5_GRISM_F444W FULL B5 F444W 0 0 2047 2047 4 951.0 484.0 90.7682 -527.3979\n')
            f.write('NRCB1_SUB160 SUB160 B1 ANY 0 1 159 160 1 79.5 79.5 -92.0533 -487.085\n')
            f.write('NRCB2_SUB160 SUB160 B2 ANY 1 1887 160 2046 1 79.5 79.5 -91.5481 -496.3272\n')
            f.write('NRCB3_SUB160 SUB160 B3 ANY 1887 1 2046 160 1 79.5 79.5 -82.0967 -487.1041\n')
            f.write('NRCB4_SUB160 SUB160 B4 ANY 1888 1887 2047 2046 1 79.5 79.5 -82.2869 -496.2056\n')
            f.write('NRCB5_SUB160 SUB160 B5 ANY 944 944 1103 1103 1 79.5 79.5 -86.8726 -491.7008\n')
            f.write('NRCA3_SUB160P SUB160P A3 ANY 1888 0 2047 159 1 79.5 79.5 22.0441 -557.6773\n')
            f.write('NRCA5_SUB160P SUB160P A5 ANY 1888 0 2047 159 1 79.5 79.5 26.0435 -553.4513\n')
            f.write('NRCB1_SUB160P SUB160P B1 ANY 1888 1888 2047 2047 1 79.5 79.5 -149.6495 -428.4826\n')
            f.write('NRCB5_SUB160P SUB160P B5 ANY 1888 1888 2047 2047 1 79.5 79.5 -148.2935 -431.795\n')
            f.write('NRCA5_GRISM256_F356W SUB256 ALONG F356W 0 379 2047 634 1 1581.0 33.0 50.7591 -556.4226\n')
            f.write('NRCB1_SUB320 SUB320 B1 ANY 0 1 319 320 1 159.5 159.5 -94.5153 -484.5942\n')
            f.write('NRCB2_SUB320 SUB320 B2 ANY 1 1727 320 2046 1 159.5 159.5 -94.0423 -498.7958\n')
            f.write('NRCB3_SUB320 SUB320 B3 ANY 1727 1 2046 320 1 159.5 159.5 -79.6308 -484.622\n')
            f.write('NRCB4_SUB320 SUB320 B4 ANY 1728 1727 2047 2046 1 159.5 159.5 -79.8049 -498.699\n')
            f.write('NRCB5_SUB320 SUB320 B5 ANY 864 864 1183 1183 1 159.5 159.5 -86.8726 -491.7008\n')
            f.write('NRCA5_MASK335R SUB320A335R A5 ANY 491 1501 810 1820 1 160.0 159.0 107.5069 -405.5171\n')
            f.write('NRCA5_MASK430R SUB320A430R A5 ANY 813 1501 1132 1820 1 160.0 159.0 87.2474 -405.1918\n')
            f.write('NRCA5_MASKLWB SUB320ALWB A5 ANY 1454 1507 1773 1826 1 160.0 159.0 47.2109 -404.4368\n')
            f.write('NRCB5_MASK335R SUB320B335R B5 ANY 1194 1531 1513 1850 1 159.0 160.0 -107.4436 -402.0345\n')
            f.write('NRCB5_MASK430R SUB320B430R B5 ANY 872 1529 1191 1848 1 159.0 160.0 -87.2134 -402.1865\n')
            f.write('NRCA5_TAGRISMTS32 SUB32TATSGRISM A5 ANY 1235 4 1266 35 1 15.5 15.5 70.845 -541.8441\n')
            f.write('NRCA3_SUB400P SUB400P A3 ANY 1648 0 2047 399 1 199.5 199.5 25.8754 -553.8581\n')
            f.write('NRCA5_SUB400P SUB400P A5 ANY 1648 0 2047 399 1 199.5 199.5 33.7948 -545.719\n')
            f.write('NRCB1_SUB400P SUB400P B1 ANY 1848 1648 2247 2047 1 199.5 199.5 -146.0122 -432.2043\n')
            f.write('NRCB5_SUB400P SUB400P B5 ANY 1848 1648 2247 2047 1 199.5 199.5 -140.8413 -439.3816\n')
            f.write('NRCB1_SUB640 SUB640 B1 ANY 0 1 639 640 1 319.5 319.5 -99.4325 -479.616\n')
            f.write('NRCB2_SUB640 SUB640 B2 ANY 1 1407 640 2046 1 319.5 319.5 -99.0379 -503.7327\n')
            f.write('NRCB3_SUB640 SUB640 B3 ANY 1407 1 2046 640 1 319.5 319.5 -74.7053 -479.6566\n')
            f.write('NRCB4_SUB640 SUB640 B4 ANY 1408 1407 2047 2046 1 319.5 319.5 -74.8328 -503.6899\n')
            f.write('NRCB5_SUB640 SUB640 B5 ANY 704 704 1343 1343 1 319.5 319.5 -86.8726 -491.7008\n')
            f.write('NRCA2_MASK210R SUB640A210R A2 ANY 391 1185 1030 1824 1 319.5 320.0 127.2272 -405.2448\n')
            f.write('NRCA4_MASKSWB SUB640ASWB A4 ANY 170 1173 809 1812 1 319.0 320.0 67.6093 -404.6656\n')
            f.write('NRCB1_MASK210R SUB640B210R B1 ANY 928 1210 1567 1849 1 319.0 320.0 -127.1519 -401.7447\n')
            f.write('NRCB5_MASKLWB SUB640BLWB B5 ANY 536 1527 855 1846 1 159.0 160.0 -67.2724 -402.3254\n')
            f.write('NRCB3_MASKSWB SUB640BSWB B3 ANY 521 1206 1160 1845 1 319.0 320.0 -47.2745 -401.9089\n')
            f.write('NRCA3_SUB64P SUB64P A3 ANY 1984 0 2047 63 1 31.5 31.5 20.5087 -559.207\n')
            f.write('NRCA5_SUB64P SUB64P A5 ANY 1984 0 2047 63 1 31.5 31.5 22.931 -556.5529\n')
            f.write('NRCB1_SUB64P SUB64P B1 ANY 1984 1984 2047 2047 1 31.5 31.5 -151.1039 -426.9936\n')
            f.write('NRCB5_SUB64P SUB64P B5 ANY 1984 1984 2047 2047 1 31.5 31.5 -151.2737 -428.7578\n')
            f.write('NRCA3_DHSPIL_SUB96 SUB96DHSPILA A3 ANY 967 1031 1062 1126 1 48.0 47.0 52.2372 -526.058\n')
            f.write('NRCB4_DHSPIL_SUB96 SUB96DHSPILB B4 ANY 988 1031 1083 1126 1 48.0 47.0 -53.188 -523.983\n')
            f.write('NRCA2_FSTAMASK210R SUBFSA210R A2 ANY 563 1211 690 1338 1 63.5 64.0 130.5186 -412.9244\n')
            f.write('NRCA5_FSTAMASKM335R SUBFSA335R A5 ANY 587 1519 650 1582 1 32.0 31.0 113.0226 -412.6908\n')
            f.write('NRCA5_FSTAMASKM430R SUBFSA430R A5 ANY 903 1525 966 1588 1 32.0 31.0 92.8054 -412.7684\n')
            f.write('NRCA5_FSTAMASKLWB SUBFSALWB A5 ANY 1423 1526 1486 1589 1 32.0 31.5 48.2918 -412.27\n')
            f.write('NRCA4_FSTAMASKSWB SUBFSASWB A4 ANY 102 1198 229 1325 1 63.0 64.0 67.4056 -412.3789\n')
            f.write('NRCA5_GRISM128_F277W SUBGRISM128 ALONG F277W 0 443 2047 570 0 1581.0 33.0 50.7591 -556.4226\n')
            f.write('NRCA5_GRISM128_F322W2 SUBGRISM128 ALONG F322W2 0 0 2047 127 0 1581.0 33.0 50.7591 -556.4226\n')
            f.write('NRCA5_GRISM128_F356W SUBGRISM128 ALONG F356W 0 443 2047 570 0 1581.0 33.0 50.7591 -556.4226\n')
            f.write('NRCA5_GRISM128_F444W SUBGRISM128 ALONG F444W 0 443 2047 570 0 951.0 33.0 90.9086 -556.2425\n')
            f.write('NRCA5_GRISM256_F277W SUBGRISM256 ALONG F277W 0 379 2047 634 0 1581.0 33.0 50.7591 -556.4226\n')
            f.write('NRCA5_GRISM256_F322W2 SUBGRISM256 ALONG F322W2 0 0 2047 255 0 1581.0 33.0 50.7591 -556.4226\n')
            f.write('NRCA5_GRISM256_F444W SUBGRISM256 ALONG F444W 0 379 2047 634 0 951.0 33.0 90.9086 -556.2425\n')
            f.write('NRCA5_GRISM64_F277W SUBGRISM64 ALONG F277W 0 475 2047 538 0 1581.0 33.0 50.7591 -556.4226\n')
            f.write('NRCA5_GRISM64_F322W2 SUBGRISM64 ALONG F322W2 0 0 2047 63 0 1581.0 33.0 50.7591 -556.4226\n')
            f.write('NRCA5_GRISM64_F356W SUBGRISM64 ALONG F356W 0 475 2047 538 0 1581.0 33.0 50.7591 -556.4226\n')
            f.write('NRCA5_GRISM64_F444W SUBGRISM64 ALONG F444W 0 475 2047 538 0 951.0 33.0 90.9086 -556.2425\n')
            f.write('NRCA2_TAMASK210R SUBNDA210R A2 ANY 404 1211 531 1338 1 63.5 64.0 134.9439 -412.8938\n')
            f.write('NRCA5_TAMASK335R SUBNDA335R A5 ANY 458 1518 521 1581 1 32.0 31.0 117.5148 -412.6635\n')
            f.write('NRCA5_TAMASK430R SUBNDA430R A5 ANY 781 1518 844 1581 1 32.0 31.0 97.2989 -412.7576\n')
            f.write('NRCA5_TAMASKLWBL SUBNDALWBL A5 ANY 1423 1526 1486 1589 1 32.0 31.0 57.1235 -412.2851\n')
            f.write('NRCA5_TAMASKLWB SUBNDALWBS A5 ANY 1704 1525 1767 1588 1 32.0 31.0 39.4485 -412.2406\n')
            f.write('NRCA4_TAMASKSWB SUBNDASWBL A4 ANY 757 1197 884 1324 1 64.0 63.0 77.4942 -412.3504\n')
            f.write('NRCA4_TAMASKSWBS SUBNDASWBS A4 ANY 102 1198 229 1325 1 63.0 64.0 57.3069 -412.3892\n')
            f.write('NRCB1_TAMASK210R SUBNDB210R B1 ANY 1430 1225 1557 1352 1 63.0 64.0 -134.7447 -409.3042\n')
            f.write('NRCB5_TAMASK335R SUBNDB335R B5 ANY 1486 1540 1549 1603 1 31.0 32.0 -117.513 -409.5193\n')
            f.write('NRCB5_TAMASK430R SUBNDB430R B5 ANY 1163 1538 1226 1601 1 31.0 32.0 -97.3886 -409.5902\n')
            f.write('NRCB5_TAMASKLWBL SUBNDBLWBL B5 ANY 506 1532 569 1595 1 31.0 32.0 -57.0805 -409.7607\n')
            f.write('NRCB5_TAMASKLWB SUBNDBLWBS B5 ANY 829 1535 892 1598 1 31.0 32.0 -77.2508 -409.7124\n')
            f.write('NRCB3_TAMASKSWB SUBNDBSWBL B3 ANY 1099 1221 1226 1348 1 63.0 64.0 -39.6828 -409.4965\n')
            f.write('NRCB3_TAMASKSWBS SUBNDBSWBS B3 ANY 525 216 652 1343 1 63.0 64.0 -57.2978 -409.5126\n')
        print('Example subarray definition file written to {}.'.format(subfile))


        #readout pattern example file
        #currently outputs a list of the nircam readout patterns
        rpattfile = 'example_nircam_readpattern_definitions.list'
        patts = Table()
        patts['name'] = ['RAPID','BRIGHT1','BRIGHT2','SHALLOW2','SHALLOW4','MEDIUM2','MEDIUM8','DEEP2','DEEP8']
        patts['nframe'] = [1,1,2,2,4,2,8,2,8]
        patts['nskip'] = [0,1,0,3,1,8,2,18,12]
        patts['maxgroups'] = [10,10,10,10,10,10,10,20,20]
        ascii.write(patts,rpattfile)
        print("Example readout pattern definition file saved to {}.".format(rpattfile))

        #phot file example, listing filter, countrate corresponding to 15th magnitude, and quantum yield
        photfile = 'example_nircam_mag15_countrates.list'
        with open(photfile,'w') as f:
            f.write('filter      countrate_for_vegamag15  quantum_yield\n')
            f.write('F070W       49281.1447294             1.0\n')
            f.write('F090W       57680.7834349             1.0\n')
            f.write('F115W       51327.5199917             1.0\n')
            f.write('F140M       21316.1696660             1.0\n')
            f.write('F150W2      149792.82264              1.0\n')
            f.write('F150W       43154.0490416             1.0\n')
            f.write('F162M       17859.1795777             1.0\n')
            f.write('F164N       1979.27224099             1.0\n')
            f.write('F182M       20132.4577227             1.0\n')
            f.write('F187N       1756.22486199             1.0\n')
            f.write('F200W       33164.2638545             1.0\n')
            f.write('F210M       12747.5970103             1.0\n')
            f.write('F212N       1640.61371661             1.0\n')
            f.write('F250M       5821.96192824             1.0\n')
            f.write('F277W       18032.9318342             1.0\n')
            f.write('F300M       6089.30652217             1.0\n')
            f.write('F322W2      29678.092864              1.0\n')
            f.write('F323N       468.430930666             1.0\n')
            f.write('F335M       5773.41314496             1.0\n')
            f.write('F356W       11956.5024828             1.0\n')
            f.write('F360M       5219.17272450             1.0\n')
            f.write('F405N       354.845594799             1.0\n')
            f.write('F410M       4329.79822793             1.0\n')
            f.write('F430M       1861.56655907             1.0\n')
            f.write('F444W       8236.65805215             1.0\n')
            f.write('F460M       1278.18616094             1.0\n')
            f.write('F466N       220.600039945             1.0\n')
            f.write('F470N       191.102453557             1.0\n')
            f.write('F480M       1220.57811704             1.0\n')
        print("Example phot definition file saved to {}.".format(photfile))


        #filter/pupil combination list file
        fpfile = 'example_nircam_filter_pupil_pairings.list'
        with open(fpfile,'w') as f:
            f.write('filter   filter_wheel   pupil_wheel\n')
            f.write('F070W    F070W           CLEAR\n')
            f.write('F090W    F090W           CLEAR\n')
            f.write('F115W    F115W           CLEAR\n')
            f.write('F140M    F140M           CLEAR\n')
            f.write('F150W2   F150W2          CLEAR\n')
            f.write('F150W    F150W           CLEAR\n')
            f.write('F162M    F150W2          F162M\n') 
            f.write('F164N    F150W2          F164N\n')
            f.write('F182M    F182M           CLEAR\n')
            f.write('F187N    F187N           CLEAR\n')
            f.write('F200W    F200W           CLEAR\n')
            f.write('F210M    F210M           CLEAR\n')
            f.write('F212N    F212N           CLEAR\n')
            f.write('F250M    F250M           CLEAR\n')
            f.write('F277W    F277W           CLEAR\n')
            f.write('F300M    F300M           CLEAR\n')
            f.write('F322W2   F322W2          CLEAR\n')
            f.write('F323N    F322W2          F323N\n')   
            f.write('F335M    F335M           CLEAR\n')
            f.write('F356W    F356W           CLEAR\n')
            f.write('F360M    F360M           CLEAR\n')
            f.write('F405N    F444W           F405N\n')
            f.write('F410M    F410M           CLEAR\n')
            f.write('F430M    F430M           CLEAR\n')
            f.write('F444W    F444W           CLEAR\n')
            f.write('F460M    F460M           CLEAR\n')
            f.write('F466N    F444W           F466N\n')
            f.write('F470N    F444W           F470N\n')
            f.write('F480M    F480M           CLEAR\n')
        print("Example filter/pupil combination file saved to {}.".format(fpfile))

        #point source list
        x = [500.0, 600.0, 700.0, 800.0]
        y = [500.0, 600.0, 700.0, 800.0]
        mags = [15.0, 14.0, 13.0, 12.0]

        psource = 'example_stars.list'
        with open(psource,'w') as f:
            f.write('# pixel\n')
            f.write("# '# pixel' must be in the top line if the coords provided are in units of detector pixels\n")
            f.write("If '# pixel' is not present in the top line, scipt assumes input coords are RA and Dec.\n")
            f.write("If RA and Dec are provided, they can be in HH:MM:SS.SSS, DD:MM:SS.SS string format\n")
            f.write("or in decimal degrees.")
            f.write("# All lines beginning with '#' will be ignored when read in\n")
            f.write('#\n')
            f.write('#xPos    yPos    Vegamag\n')
            for xv,yv,mv in izip(x,y,mags):
                f.write('{}  {}  {}\n'.format(xv,yv,mv))
        print("Example point source list saved to {}.".format(psource))


        #galaxy list
        galaxyfile = 'example_galaxies.list'
        with open(galaxyfile,'w') as f:
            f.write('#\n')
            f.write('#\n')
            f.write("# x and y can be pixel values, or RA and Dec strings (HH:MM:SS.SS, DD:MM:SS.SS) or decimal degrees.\n")
            f.write("# To differentiate, put 'pixels' in the top line if the inputs are pixel values.\n")
            f.write("# Radius can also be in units of pixels or arcseconds. Put 'radius_pixels' in the second line above\n")
            f.write('to specify radii in pixels.\n')
            f.write('# position angle is given in degrees counterclockwise. A value of 0 will align the semimajor axis with\n')
            f.write('the x axis of the detector.\n')
            f.write('x_or_RA y_or_Dec radius ellipticity pos_angle sersic_index magnitude\n')
            f.write('359.98222 -0.0054125 1.373 0.0912 101.293 3.07 19.79\n')
            f.write('359.99151 -0.0041562 1.837 0.6898 236.011 3.35 19.02\n')
            f.write('359.99609 -0.0053726 0.428 0.4734 353.997 2.24 25.53\n')
        print("Example galaxy source list saved to {}.".format(psource))
        

    def add_options(self,parser=None,usage=None):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,description='Simulate JWST ramp')
        parser.add_argument("paramfile",help='File describing the input parameters and instrument settings to use. (YAML format).')
        parser.add_argument("--param_example",help='If used, an example parameter file is output.')
        parser.add_argument("--input_examples",help="If used, output examples of subarray bounds file, readpattern file, point source list")
        return parser


if __name__ == '__main__':

    usagestring = 'USAGE: ramp_simulator.py inputs.yaml'

    newramp = RampSim()
    parser = newramp.add_options(usage = usagestring)
    args = parser.parse_args(namespace=newramp)


    #if the user requests an example parameter file, output and then quit
    if newramp.param_example:
        newramp.make_param()

    if newramp.input_examples:
        newramp.make_input_examples()

    if ((not newramp.param_example) and (not newramp.input_examples)):
        newramp.run()
