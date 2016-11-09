# PLVM - Probabilistic Latent Variable Modeling toolbox for multisubject data
The Probabilistic Latent Variable Modeling Toolbox for Multisubject Data 
holds a collection of latent variable algorithms implemented in Matlab™. 
The algorithms support the use of graphical processing units (GPUs) for 
high performance computing. All code can be used freely in research and 
other non-profit applications. If you publish results obtained with this 
toolbox we kindly ask that our and other relevant sources are properly cited. 

This toolbox has been developed at:

The Technical University of Denmark, 
Department for Applied Mathematics and Computer Science,
Section for Cognitive Systems.

The toolbox was developed in connection with the Brain Connectivity project 
at DTU (https://brainconnectivity.compute.dtu.dk/) .

## Algorithms:

* psFA
	- Probabilistic Sparse Factor Analysis (psFA). Models subject specific heteroscedastic feature/voxel noise.
* psPCA
	- Probabilistic Sparse Principal Component Analysis (psPCA). Models subject specific homoscedastic noise.

Common algorithm properties

* No parameters need to be set by default.
* All models support modeling multiple subjects.
* An approximate model solution is found using variational inference.
* The evidence lowerbound is calculated (an approximation to log likelihood).
* Estimating number of components using Automatic Relevance Determination (ARD).
* Ability to individually turn off modeling aspects such as sparsity, ARD and noise modeling.

## Demonstrators:
* demo_psFA
	- Demostration on a toy example showing model settings and their effect
* calcICASSP17
	- Synthetic experiments in the original psFA article (which is currently under review).
