#! /usr/bin/env python

from distutils.core import setup

setup(name='ramp_simulator',
      version = '1.1.0',
      description='JWST Multiaccum Ramp Simulator',
      long_description='A tool to create simulated raw NIRCAm ramps based on dark current ramps from ground testing in combination with source catalogs.',
      url='https://github.com/spacetelescope/ramp_simulator',
      author='Bryan Hilbert',
      author_email='hilbert@stsci.edu',
      py_modules = ['ramp_simulator','polynomial','rotations',
                    'raDec_between_detectors','create_galaxy_list',
                    'create_point_source_list','moving_targets',
                    'dispersed_ramp_simulator','read_siaf_table',
                    'observations','read_fits',
                    'set_telescope_pointing_separated',
                    'RADec_vs_xy_map',
                    'apt_inputs','yaml_generator'],
    )
