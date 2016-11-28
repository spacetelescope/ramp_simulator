# ramp_simulator
Code to create high-fidelity simulations of JWST multiaccum integrations.

This code takes an existing real, raw dark current ramp for the instrument being simulated, and adds simulated
signals on top of the dark current. Options for these simulated sources include point sources, galaxies (2D sersic models),
extended sources, and moving targets.

Sources are entered via list files, with an x,y or RA,Dec position, as well as magnitude, for each source. Example input files
are provided in the repo. 

A detailed description of how the code works, as well as the inputs required, is given in the accompanying documentation.

The script is called with a single argument; the name of the input parameter file, which is a yaml file:
python ramp_simulator.py input_params.yaml

The output is a single multiaccum integration which can be immediately run through the JWST calibration pipeline.
