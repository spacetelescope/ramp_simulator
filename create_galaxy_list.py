#! /usr/bin/env python

'''
Create an example file that lists input galaxies to be fed into the
data ramp simulator.
'''

import numpy as np
from astropy.table import Table

number_of_sources = 200
outfile = 'galaxies_single_detector_ra0dec0.list'

outtab = Table()

delta = 2048 * (np.sqrt(2) - 1) / 2
minx = 0 - delta
maxx = 2048 + delta
miny = minx
maxy = maxx

minra = 359.9917
maxra = 360.0083

mindec = -0.009
maxdec = 0.008

minmag = 27
maxmag = 18

minrad = 0.2
maxrad = 2

minellip = 0
maxellip = 0.75

minsersic = 1
maxsersic = 4

ras = np.random.uniform(minra,maxra,number_of_sources) 
over = ras >= 360.
ras[over] = ras[over] - 360.

outtab['x_or_RA'] = ras
outtab['y_or_Dec'] = np.random.uniform(mindec,maxdec,number_of_sources)
outtab['radius'] = np.random.uniform(minrad,maxrad,number_of_sources)
outtab['ellipticity'] = np.random.uniform(minellip,maxellip,number_of_sources)
outtab['pos_angle'] = np.random.uniform(0,359,number_of_sources)
outtab['sersic_index'] = np.random.uniform(minsersic,maxsersic,number_of_sources)
outtab['magnitude'] = np.random.uniform(minmag,maxmag,number_of_sources)

outtab.meta['comments'] = ["","","x and y can be pixel values, or RA and Dec strings or floats. To differentiate, put 'pixels' in the top line if the inputs are pixel values.","radius can also be in units of pixels or arcseconds. Put 'radius_pixels' at top of file to specify radii in pixels.","position angle is given in degrees counterclockwise. A value of 0 will align the semimajor axis with the x axis of the detector."]

outtab.write(outfile,format='ascii',overwrite=True)
