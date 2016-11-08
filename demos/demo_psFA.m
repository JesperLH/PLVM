%% Demostration script for Probabilistic Sparse Factor Analysis
%
clear; close all; clc
%% Generate some data
rng(64501)
V = 80; T = 6; D = 3; %Voxels, Timesteps, Components
% Number of subjects. 
B = 1; %=> psFA
%B = 5;%=> multi subject (5) psFA 
sparsity_pattern = rand(V,D)*5.*(rand(V,D)>0.5); %
A_true = randn(V,D).*sparsity_pattern;
S_true = randn(D,T,B);
noise = randn(V,T,B)*0.1;

data = my_pagefun(@mtimes,A_true,S_true)+noise;
%% Estimate probability distributions of the psFA model
D_est = 5; %Number of components to look for
[first_moments,other_moments,priors,elbo] = psFA(data,D_est); %#ok<ASGLU>

% Show results
if B == 1
    figure; colormap gray
    subplot(2,3,1:2); imagesc([A_true,zeros(V,D_est-D)]'); title('True A');
    subplot(2,3,4:5); imagesc(first_moments.A'); title('Est. A')
    subplot(2,3,3); imagesc([S_true;zeros(D_est-D,T)]); title('True S')
    subplot(2,3,6); imagesc(first_moments.S); title('Est. S')
end
%% Variable input arguments
scale = sum(data(:).^2)/numel(data);
[V,T,B] = size(data);

% Possible input arguments and their default values.
[first_moments,other_moments,priors,elbo] = psFA(data,D_est,...
    'conv_crit',1e-9,... %Convergence criteria, the algorithm stops if the relative change in lowerbound is below this value
    'maxiter',200,... % Maximum number of iteration, stops here if convergence is not achieved
    'noise_process',true,... % Model subject specific heteroscedastic noise (over voxels)
    'sparse_prior',true,... % Model an elementwise sparsity pattern on A (probabilistic FA is achieved if false)
    'ard_prior',true,... % Model automatic relevance determination (ARD) for the components
    'fixed_sparse',25,... % Number of iterations before modeling sparsity pattern on A
    'fixed_ard',30,... % Number of iterations before modeling component ARD
    'fixed_noise',35,... % Number of iterations before modeling subject specific heteroscedastic noise (over voxels)
    'mean_process',false,... % Model a subject specific mean value (A*S+mean = data)
    'runGPU',false,... %Run the model on graphics processing units (Only a single card is supported)
    'iter_disp',1,... % How many iterations between console output
    'beta',1e-6,... % Hyper-parameter for the mean
    'alpha_a',1e-6,... % Hyper-parameter for the sparsity pattern on A
    'alpha_b',1e-6*scale/V,... 
    'gamma_a',1e-6,... % Hyper-parameter for the ARD process
    'gamma_b',1e-6*scale/(B*V),...
    'tau_a',1e-6,... % Hyper-parameter for the subject specific noise precision
    'tau_b',1e-6*scale/V,... 
    'rngSEED',[],... % Specify a random seed for reproducability ([] means no seed is specified)
    'opts',[]); %#ok<ASGLU>


%% Using a structure for variable input arguments
% NOTE: If an argument is specified both as a variable input argument and
% in the optional structure (see 'maxiter' below), then the value in the 
% optional structure is used.  
opts_stuct.maxiter = 1000;
opts_stuct.runGPU = false;
[first_moments,other_moments,priors,elbo] = ...
    psFA(data,D_est,'maxiter',400,'opts',opts_stuct);

%% For psPCA it is more computationally efficient to use psPCA
% which is called in a similar fashion.
opts_pspca.runGPU = false;
psPCA(data,5,'opts',opts_pspca)