#! /usr/bin/env python

from distutils.core import setup

setup(name='ramp_simulator',
      version='0.1',
      description='Simulate raw JWST integrations',
      author='Bryan Hilbert',
      author_email='hilbert@stsci.edu',
      py_modules=['ramp_simulator','polynomial','rotations','moving_targets','RADec_vs_xy_map.py']
      )
