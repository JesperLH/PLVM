From the ICA toolbox by DTU Cognitive Systems (see. http://cogsys.imm.dtu.dk/toolbox/ica/)

	
Maximum likelihood (Infomax) - icaML

The algorithm is equivalent to Infomax by Bell and Sejnowski 1995 [1] using a maximum likelihood formulation. No noise is assumed and the number of observations must equal the number of sources. The BFGS method [2] is used for optimization.

The number of independent components are calculated using Bayes Information Criterion [3] (BIC), with PCA for dimension reduction.

Properties:

Linear and instantaneous mixing.
Square mixing matrix.
No noise.
Update history:

020103 Version 1.4 Included pre-processing with SVD to reduce input dimension. Added optimisation parameter setting and removed log likelihood problem with icaML output
030909 Version 1.5 Fixed help message to inform the user about U. Removed the automatic use of PCA in the quadratic case.
060622 Version 1.6 Errors in icaML_bic corrected.