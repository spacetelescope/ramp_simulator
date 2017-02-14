#! /usr/bin/env python

from distutils.core import setup

setup(name='ramp_simulator',
      version = '1.1',
      description='JWST Multiaccum Ramp Simulator',
      author='Bryan Hilbert',
      author_email='hilbert@stsci.edu',
      py_modules = ['ramp_simulator','polynomial','rotations',
                    'RADec_vs_xy_map','create_galaxy_list',
                    'create_point_source_list','moving_targets',
                    'dispersed_ramp_simulator','read_siaf_table',
                    'observations'])
