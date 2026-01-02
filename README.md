he UAT (Unified Applicable Timeframe) Cosmological Framework
Author: Miguel Angel Percudani

Status: Official Implementation for Manuscript Submission

DOI: [https://doi.org/10.5281/zenodo.18125223]

Overview
This repository contains the official Python implementation of the Unified Applicable Timeframe (UAT) framework. UAT provides a self-consistent solution to the Hubble tension (H 
0
​
 ) by incorporating early-universe modifications motivated by Loop Quantum Gravity (LQG) effects.

By increasing the effective number of relativistic species (ΔN 
eff
​
 ), the UAT framework reduces the sound horizon (r 
d
​
 ) at the drag epoch, bridging the gap between Planck CMB measurements and local distance ladder observations (SH0ES).

Scientific Key Findings
Hubble Constant: H 
0
​
 ≈72.6 km/s/Mpc (consistent with SH0ES).

Tension Reduction: Reduced from 5.1σ (in ΛCDM) to ≈3.0σ.

Mechanism: ΔN 
eff
​
 ≈1.2−1.7 additional radiation density.

Statistical Evidence: Bayesian Factor lnB=+12.7 (Decisive evidence over ΛCDM).

Repository Structure
UAT_Analysis_Full.py: Main MCMC engine using the emcee sampler. It performs the joint likelihood analysis of CMB, BAO, and H 
0
​
  data.

UAT_Publication_Generator.py: Generates final publication-quality figures, LaTeX tables, and the executive summary.

requirements.txt: List of necessary Python libraries.

Installation
Ensure you have Python 3.8+ installed. You can install the dependencies via pip:

Bash

pip install numpy matplotlib scipy emcee corner
Usage
To run the full cosmological analysis and generate the posterior distributions:

Bash

python UAT_Analysis_Full.py
To generate the specific tables and figures for the manuscript:

Bash

python UAT_Publication_Generator.py
Note on Reproducibility
In accordance with open science principles, this implementation does not use a fixed random seed. This is intentional to demonstrate the stochastic robustness of the UAT solution. While median values for H 
0
​
  and ΔN 
eff
​
  may show minor variations between runs (typically <1σ), the convergence towards the resolution of the Hubble tension remains consistent across different environments.

Citation
If you use this code or the UAT framework in your research, please cite:

Percudani, M. A. (2025). The Unified Applicable Timeframe (UAT): An Autoconsistent Solution to the Hubble Tension via Early Quantum Effects. [https://doi.org/10.5281/zenodo.18125223].

License
This project is licensed under the MIT License - see the LICENSE file for details.
