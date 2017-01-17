#! /usr/bin/env python

'''
Create an example file that lists input point sources to be fed into the
data ramp simulator.

We are trying to model all 10 detectors (including dithers). 
So for the SW, at 0.032"/pix, one detector is 65.5" across. 5" between
SW detectors, and 43" between modules. So all the detectors together will
cover:

4*65.5 + 43 + 5 + 5 = 315"

Include distortion, so let's bump it up to 360". 






'''

import numpy as np
from astropy.table import Table

number_of_sources = 100
outfile = 'point_sources_single_detector_ra0dec0.list'

outtab = Table()

delta = 2048 * (np.sqrt(2) - 1) / 2
minx = 0 - delta
maxx = 2048 + delta
miny = minx
maxy = maxx

minra = 359.9917
maxra = 360.0083

mindec = -0.0083
maxdec = +0.0083

minmag = 27
maxmag = 18

ras = np.random.uniform(minra,maxra,number_of_sources) 
over = ras >= 360.
ras[over] = ras[over] - 360.

outtab['x_or_RA'] = ras
outtab['y_or_Dec'] = np.random.uniform(mindec,maxdec,number_of_sources)
outtab['magnitude'] = np.random.uniform(minmag,maxmag,number_of_sources)

outtab.meta['comments'] = ["","","x and y can be pixel values, or RA and Dec strings or floats. To differentiate, put 'pixels' in the top line if the inputs are pixel values.","radius can also be in units of pixels or arcseconds. Put 'radius_pixels' at top of file to specify radii in pixels.","position angle is given in degrees counterclockwise. A value of 0 will align the semimajor axis with the x axis of the detector."]

outtab.write(outfile,format='ascii',overwrite=True)
