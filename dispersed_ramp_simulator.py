#! /usr/bin/env python

'''
Simulator for dispersed ramps. Really this is just a wrapper 
around ramp_simulator.py and Nor's disperse.py

Plot summary:

input: series of yaml files for the ramp simulator that gives
relevant details for creating a series of signal rate images
for multiple filters.

yaml files -> ramp_simulator.py -> rate images

rate images -> disperser -> single dispersed rate image

dispersed rate image -> ramp_simualtor.py -> dispersed ramp

Allowed cross filters 
Any LW W2, W, or M filter.
N filters not allowed since they are all on the 
pupil wheel along with the grisms


'''

import argparse
import ramp_simulator as rampsim
import observations
import numpy as np
import copy
import sys
import yaml
from astropy.io import fits
from astropy.table import Table

class Simulator():


    def run(self):

        #read in list of input files for ramp_simulator
        yaml_files = self.read_input_list(self.yaml_list)

        directfiles = []
        filters = []
        for file in yaml_files:            

            #get the output name from the yaml files so that we
            #will know the names of the countrate images
            inParams = self.readYaml(file)
            directfiles.append(self.getCountRateName(inParams))
            filters.append(self.getFilterName(inParams))

            #Run the ramp simulator 
            newramp = rampsim.RampSim()
            newramp.paramfile = file
            newramp.param_example = False
            newramp.input_examples = False
            newramp.run()
            #print("skipping the initial run of ramp simulator for debugging")
            #print("since the direct images already exist")
            
        #Inputs to disperser code
        if self.dispdir.lower() == 'column':
            config = "NIRCAM_C.conf"
        else:
            config = "NIRCAM_R.conf"

        #This is the passband to use for the output, dispersed data
        #It should be the same for all outputs
        #passband = "We/Need/PassBand/Files/Somewhere/%s_passband.dat" % self.pairedFilter.upper()
        passband_dir = '/grp/jwst/wit/nircam/reference_files/SpectralResponse_2015-Final/Filters_ASCII-Tables/nircam_throughputs/modA/without_0.98reserve_factor/nrc_plus_ote/'
        pbandfile = "{}_nircam_plus_ote_throughput_moda_sorted.txt".format(self.pairedFilter.upper())
        passband = passband_dir + pbandfile
        
        
        #Output direct data can be a single countrate frame or a
        #group, depending on observing mode
        #Use the 'units' header keyword value to distinguish.
        #The value should be the same for all files.
        with fits.open(directfiles[0]) as h:
            units = h[1].header['UNITS']
            yd = h[1].header['NAXIS2']
            xd = h[1].header['NAXIS1']

        #if we have moving target data, where the grism direct data is a ramp,
        #then we need to split the ramp up here into separate files for each group.
        if units == 'e-':
            gfiles = []
            for i,file in enumerate(directfiles):
                #make sure the units in each file agrees. We can't
                #mix count rate images and ramps with counts
                ucheck = fits.getval(file,'units')
                if ucheck != 'e-':
                    print("WARNING: {} has units of {}, but units of 'e-' are expected!!".format(file,ucheck))
                    sys.exit()
                else:
                    splitFiles = self.splitRamp(file)
                    gfiles.append(splitFiles)

            #Now we need to collect together all of the files with 
            #containing similar group numbers, and call the disperser
            #once with each group of files
            numgroups = len(splitFiles)
            dispInt = np.zeros((numgroups,yd,xd))
            for i in range(numgroups):
                dispfiles = [x[i] for x in gfiles]
                segfile = self.findSegFile(dispfiles,filters)
                seg_map = self.makeSegmentationMap(segfile)
                
                #Set up observation object and disperse.
                #Run once for each order
                dispInt[i,:,:] = self.disperse(dispfiles,seg_map,config,passband=passband)


            print("ramp_simulator not ready for this yet!!!!!")
            sys.exit()

                
        else: #normal imaging data. countrate images can go directly to disperser
            segfile = self.findSegFile(directfiles,filters)
            seg_map = self.makeSegmentationMap(segfile)

            #save segmentation map as a check
            h0=fits.PrimaryHDU(seg_map)
            hl = fits.HDUList([h0])
            hl.writeto('test_seg.fits',overwrite=True)
            
            dispInt = self.disperse(directfiles,seg_map,config,passband=passband)

        #Save the dispersed, noiseless image/ramp
        if self.dispFileName is None:
            self.dispFileName = directfiles[0][0:-5] + '_noiselessDispersed.fits'
        self.saveDispersed(dispInt,self.dispFileName,directfiles)
                
        #Now create yaml input file for ramp simulator
        print("remember that currently the simulator cannot handle a full ramp")
        print("as an input to add to the dark. ")

        #Filename of yaml file for final ramp simulator run
        dot = yaml_files[0].rfind('.')
        final_yaml = yaml_files[0][0:dot] + '_FinalDispersedRamp.yaml'

        #Output name for final dispersed ramp
        if self.output_file is None:
            self.output_file = yaml_files[0:dot] + '_FinalDispersedRamp.fits'
                
        dispYaml = self.readYaml(yaml_files[0])
        dispYaml['Inst']['mode'] = 'wfss'
        dispYaml['simSignals']['pointsource'] = 'None'
        dispYaml['simSignals']['galaxyListFile'] = 'None'
        dispYaml['simSignals']['extendedscale'] = 1.0
        dispYaml['simSignals']['PSFConvolveExtended'] = False
        dispYaml['simSignals']['movingTargetList'] = 'None'
        dispYaml['simSignals']['movingTargetSersic'] = 'None'
        dispYaml['simSignals']['movingTargetExtended'] = 'None'
        dispYaml['simSignals']['movingTargetToTrack'] = 'None'
        dispYaml['Output']['file'] = self.output_file
        dispYaml['Output']['grism_source_image'] = False
        dispYaml['Readout']['filter'] = self.pairedFilter
        if self.dispdir.lower() == 'column':
            dispYaml['Inst']['pupil'] = 'GRISMC'
        else:
            dispYaml['Inst']['pupil'] = 'GRISMR'
            
        #Create an extended target input file using the
        #output dispersed file
        extendedListFile = self.makeExtendedList(self.dispFileName,dispYaml)
        dispYaml['simSignals']['extended'] = extendedListFile

        with open(final_yaml,'w') as fin:
            yaml.dump(dispYaml,fin,default_flow_style=False)

            
        #Run the ramp simulator
        newramp = rampsim.RampSim()
        newramp.paramfile = final_yaml
        newramp.param_example = False
        newramp.input_examples = False
        newramp.run()

        #Finished
        print("Final dispersed integration written to {}.".format(self.output_file))
        

    def makeExtendedList(self,disp_file,yamlDict):
        #make a text file that will be used as the extended source input list
        #to the ramp simulator using the dispersed file
        outfile = disp_file[0:-5] + '_extendedTargList.list'

        meta0 = ''
        meta1 = ''
        meta2 = 'x and y can be pixel values, or RA and Dec strings or floats.'
        meta3 = "To differentiate, put 'position_pixel' in the top line if the inputs"
        meta4 = 'are pixel values.'
        tab = Table()
        tab['filename'] = [disp_file]
        tab['x_or_RA'] = [yamlDict['Telescope']['ra']]
        tab['y_or_Dec'] = [yamlDict['Telescope']['dec']]
        tab['pos_angle'] = [0.]
        tab['magnitude'] = ['None'] #no rescaling of input image
        tab.meta['comments'] = [meta0,meta1,meta2,meta3,meta4]
        tab.write(outfile,format='ascii',overwrite=True)
        return outfile
        
    def saveDispersed(self,data,outname,filelist):
        #Save the noiseless dispersed image
        with fits.open(filelist[0]) as h:
            h0head = h[0].header
            h1head = h[1].header

        #Add the list of files used to create the dispersed image
        #into the headers
        h0head.add_history('Files used to construct dispersed data:')
        h1head.add_history('Files used to construct dispersed data:')
        for file in filelist:
            h0head.add_history(file)
            h1head.add_history(file)

        #Copy the headers from the first input file
        #into the dispersed image file
        h0 = fits.PrimaryHDU(np.array([]),h0head)
        h1 = fits.ImageHDU(data,h1head)
        hlist = fits.HDUList([h0,h1])
        hlist.writeto(outname,overwrite=True)
        

    def disperse(self,files,seg,config,passband):
        #Set up an observation and disperse. Run once for each order
        obs = observations.observation(files,seg,config,passband,order="+1")
        obs.disperse()
        p1 = copy.deepcopy(obs.simulated_image)        
        obs = observations.observation(files,seg,config,passband,order="+2")
        obs.disperse()
        #check
        h0=fits.PrimaryHDU(p1)
        h1=fits.ImageHDU(obs.simulated_image)
        hl = fits.HDUList([h0,h1])
        hl.writeto('test.fits',overwrite=True)


        return p1 + obs.simulated_image
        

    def makeSegmentationMap(self,file,threshold=0.3):
        #create a segmentation map of 0s and 1s for the given
        #file.
        with fits.open(file) as h:
            data = h[1].data
            tgroup = h[1].header['TGROUP']
            try:
                group = h[1].header['GRPNUM']
            except:
                group = 0

        #make sure data are only 2D
        if len(data.shape) == 3:
            data = data[0,:,:]
                
        #convert counts to countrate if necessary
        if tgroup != 0:
            data /= ((group+1)*tgroup)

        #remove any detector-wide background
        print("Segmentation map, background to remove: {}".format(np.min(data)))
        data = data - np.min(data)

        #set any pixels with signal above threshold equal to 1
        seg = np.zeros_like(data)
        seg[data >= threshold] = 1
        return seg
        
        
    def findSegFile(self,directfiles,filters):
        #find the file with the longest wavelength filter
        #and return that, as the file with which to make the
        #segmentation map
        directfiles = np.array(directfiles)
        waves = np.array([np.float(x[1:4]) for x in filters])
        maxwave = waves == np.max(waves)
        return directfiles[maxwave][0]

    
    def splitRamp(self,file):
        #split a single ramp file into separate files
        #for each group
        with fits.open(file) as h:
            data = h[1].data
            header0 = h[0].header
            header1 = h[1].header

        numgroups = data.shape[0]
        outfilelist = []
        for group in range(numgroups):
            header1['GRPNUM'] = group
            
            outfile = file[0:-5] + '_Group' + str(group) + '.fits'
            outfilelist.append(outfile)
            h0 = fits.PrimaryHDU(header0)
            h1 = fits.ImageHDU(data[group,:,:],header1)
            hlist = fits.HDUList([h0,h1])
            hlist.write(outfile,overwrite=True)
            
        return outfilelist


    def segmentationMap(self,frame,limit,outfile):
        #create and save a segmentation map of the given data frame.
        #1 for pix with signal, 0 for pix without signal
        limit = np.float(limit)
        dispPix = frame > limit
        seg_map = data * 0
        seg_map[dispPix] = 1

        #save the file
        h0 = fits.PrimaryHDU(seg_map)
        hlist = fits.HDUList([h0])
        hlist.write(outfile,overwrite=True)
        return seg_map
        
        
    def readYaml(self,file):
        #read in yaml parameter file
        try:
            with open(file,'r') as infile:
                params = yaml.load(infile)
        except:
            print("WARNING: unable to open {}".format(file))
            sys.exit()
        return params
            
        
    def getCountRateName(self,yamldict):
        #get the output grism direct image name from yaml dict
        return yamldict['Output']['file'][0:-5] + '_GrismDirectData.fits'

    
    def getFilterName(self,yamldict):
        #get the filter name from yaml dict
        return yamldict['Readout']['filter']

    
    def read_input_list(self,file):
        files = []
        with open(file) as f:
            t = f.readlines()
        t = [x.strip() for x in t if x.strip() != '']
        return t

                
    def add_options(self,parser=None,usage=None):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage,description='Create a simulated WFSS dispersed integration')
        parser.add_argument("yaml_list",help="File containing a list of yaml files that can act as input to ramp_simulator.py",type=str)
        parser.add_argument("dispdir",help="Direction to disperse sources. Use 'row' or 'column'",default='row')
        parser.add_argument("pairedFilter",help="Name of filter for observation, used in conjunction with grism")
        parser.add_argument("--dispFileName",help="Name of output dispersed noiseless image/ramp. This will be input to the ramp simulator.",default=None)
        parser.add_argument("--output_file",help="Name of the final dispersed multiaccum ramp file.",default=None)
        return parser
                        

if __name__ == '__main__':

    usagestring = "dispersed_ramp_simulator.py yaml_files.list row F430M --ouput_file 'GrismRamp_F430M.fits'"

    ramp = Simulator()
    parser = ramp.add_options(usage = usagestring)
    args = parser.parse_args(namespace = ramp)

    ramp.run()
    
