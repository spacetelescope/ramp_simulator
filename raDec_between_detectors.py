#! /usr/bin/env python

'''
Given an RA,Dec value for a given NIRCam detector and aperture,
return the RA,Dec value for a different NIRCam detector and aperture.

input:

detector name, aperture name, ra, dec, telescope rotation, desired detector, desired aperture

output:
ra, dec in desired detector at desired aperture reference location
'''

from astropy.io import ascii
import rotations
import argparse
import numpy as np
import sys

def find_radec(siaf_file,in_det,in_ap,in_ra,in_dec,rot,out_det,out_ap):

    #If the input RA and Dec are HH:MM:SS strings, convert to
    #decimal degrees
    try:
        in_ra = np.float(in_ra)
        in_dec = np.float(in_dec)
    except:
        in_ra,in_dec = parseRADec(in_ra,in_dec)

    #Read in the SIAF file
    distortionTable = ascii.read(siaf_file,header_start=1)

    #Find V2,V3 reference location values for the input and output detectors
    in_v2,in_v3 = find_v2v3(in_det,in_ap,distortionTable)
    out_v2,out_v3 = find_v2v3(out_det,out_ap,distortionTable)

    #Calculate the delta V2,V3 (for possible output)
    #delta_v2 = in_v2 - out_v2
    #delta_v3 = in_v3 - out_v3

    #Create attitude matrix for the given input RA,Dec
    attitude_matrix = rotations.attitude(in_v2,in_v3,in_ra,in_dec,rot)
    
    #Find the RA,Dec at the reference location in the output detector
    out_ra,out_dec = rotations.pointing(attitude_matrix,out_v2,out_v3)

    #Convert to RA,Dec strings for convenience
    out_ra_str,out_dec_str = raDecStrings(out_ra,out_dec)

    return out_ra,out_dec,out_ra_str,out_dec_str


    
def find_v2v3(detector,aperture,table):
    #Find V2,V3 reference location value for detector,aperture
    full_ap = detector + '_' + aperture
    
    match = table['AperName'] == full_ap
    if np.any(match) == False:
        print("Aperture name {} not found in input CSV file.".format(full_ap))
        sys.exit()

    row = table[match]

    #Get the V2,V3 values of the reference pixel
    return row['V2Ref'].data[0],row['V3Ref'].data[0]


def raDecStrings(alpha1,delta1):
    #given a numerical RA/Dec pair, convert to string
    #values hh:mm:ss
    if alpha1 < 0.: 
        alpha1=alpha1+360.
    if delta1 < 0.: 
        sign="-"
        d1=np.abs(delta1)
    else:
        sign="+"
        d1=delta1
    decd=np.int(d1)
    value=60.*(d1-np.float(decd))
    decm=np.int(value)
    decs=60.*(value-decm)
    a1=alpha1/15.0
    radeg=np.int(a1)
    value=60.*(a1-radeg)
    ramin=np.int(value)
    rasec=60.*(value-ramin)
    alpha2="%2.2d:%2.2d:%7.4f" % (radeg,ramin,rasec)
    delta2="%1s%2.2d:%2.2d:%7.4f" % (sign,decd,decm,decs)
    alpha2=alpha2.replace(" ","0")
    delta2=delta2.replace(" ","0")
    return alpha2,delta2


def parseRADec(rastr,decstr):
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
        ra0=15.*(np.int(values[0])+np.int(values[1])/60.+np.float(values[2])/3600.)

        values=decstr.split(":")
        if "-" in values[0]:
            sign=-1
            values[0]=values[0].replace("-"," ")
        else:
            sign=+1
        dec0=sign*(np.int(values[0])+np.int(values[1])/60.+np.float(values[2])/3600.)
        return ra0,dec0
    except:
        print("Error parsing RA,Dec strings: {} {}".format(rastr,decstr))
        sys.exit()


def add_options(parser=None,usage=None):
    if parser is None:
        parser = argparse.ArgumentParser(usage=usage,description='Find RA,Dec between detector reference locations for a given telescope pointing')
    parser.add_argument("siaf_file",help='File containing SIAF information')
    parser.add_argument("in_detector",help='Detector with known RA,Dec')
    parser.add_argument("in_aperture",help='Aperture used within in_detector')
    parser.add_argument("in_ra",help="RA at reference location in in_detector at reference location in in_aperture, in decimal degrees")
    parser.add_argument("in_dec",help="Dec at reference location in in_detector at reference location in in_aperture, in decimal degrees")
    parser.add_argument("JWST_rotation",help="Telescope rotation. Degrees.",type=np.float)
    parser.add_argument("out_detector",help='Detector to find RA,Dec at reference location')
    parser.add_argument("out_aperture",help='Aperture to find RA,Dec at reference location')
    return parser



if __name__ == '__main__':

    usagestring = 'USAGE: raDec_between_detectors.py detector aperture, ra, dec, rotation, outdetector, outaperture'

    
    parser = add_options(usage = usagestring)
    args = parser.parse_args()

    outra, outdec, outra_str, outdec_str = find_radec(args.siaf_file,args.in_detector,args.in_aperture,args.in_ra,args.in_dec,args.JWST_rotation,args.out_detector,args.out_aperture)

    print(outra, outdec, outra_str, outdec_str)
