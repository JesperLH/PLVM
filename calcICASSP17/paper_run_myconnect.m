%% Run Myconnectome (Russel Poldrack)
% For a full description see [1]
%
% The resting-state data comes from [1] availabe at
% https://openfmri.org/dataset/ds000031/, then pre-processed using SPM12
% (Statistical Parametric Mapping) availabe at
% http://www.fil.ion.ucl.ac.uk/spm/software/spm12/ .
%
%   1. Image registration (to the first image of session 014)
%   2. Motion correction
%   3. Segmentation of grey matter, white matter and cerebrospinal fluid
%   4. High pass filtering (1/128 Hz)
%   5. Nuisance regressed to erode cerebrospinal fluid and white matter
%   6. Wavelet despiked
%   7. Reslicing
%
%   A grey matter mask was then applied giving 69430 voxels over 518
%   timepoints for each session. Due to memory limitations on the GPU we
%   only considered the 25 first sessions of the data. 
%
%% References
% [1] Jesper L. Hinrich, Søren F.V. Nielsen, Nicolai A. B. Riis, Casper T.
%     Eriksen, Jacob Frøsig, Marco D. F. Kristensen, Mikkel N. Schmidt,
%     Kristoffer Hougaard Madsen, and Morten Mørup, “Scalable group level
%     probabilistic sparse factor analysis,” in 2017 IEEE International
%     Conference on Acoustics, Speech, and Signal Processing, ICASSP’17 (in
%     review). 2017, IEEE.
% [2] Poldrack, R. A. et al. Long-term neural and physiological phenotyping
%     of a single human. Nature communications, 6:8885, 9 December 2015.

%% To produce the same inital solution the random seed is fixed.
% Note exact replication may no be possible due to mul
SEED = 3123123; 
rng(SEED);
close all;
%% Load data
% The data matrix is voxels x (timepoints*sessions)
fprintf('Loading data...'); tic
file= '.\Data\fMRI\myconnect\myconnect_s5f.mat';
load(file); toc

save_folder = './';
save_name = 'icassp_2000';
method = 'psFA';    

mkdir(save_folder);
%% Settings
%Choice of method
method = 'psFA';
%method = 'pFA'; 

opts.maxiter = 2000;
opts.noise_process = true;
opts.sparse_prior = true;
opts.ard_prior = true;
opts.fixed_noise = 50;
opts.fixed_ard = 40;
opts.fixed_sparse = 20;

%Hyper parameters to test
alpha_interval = 1e-6; %[1e-6,1e-3,1]
gamma_interval = 1e-6; %[1e-6,1e-3,1]
prior_labs_gam = {'low'}; %{'low','med','high'};
prior_labs_alpha = {'low'};

opts.alpha_a = 1e-6;
opts.alpha_b = opts.alpha_a;
opts.gamma_a = opts.alpha_a;
opts.gamma_b = opts.alpha_a;
opts.tau_a = opts.alpha_a;
opts.tau_b = opts.alpha_a;

%Number of repeated analysis and their random seed
repeats = 5;
repSEEDs = randi([1 10^7],repeats,1);

%% The 25 sessions is picked out and the data is reshaped into a 3way array
% such that the dimensions are voxels x timestep x sessions
V = size(X,1);
sess = 1:25;
Y = reshape(X(:,1:sum(T(sess))),V,T(1),length(sess));

%% Run psFA and pFA on the myconnect data

% Number of components to look for
for D = [50]%,75] 
    % Loop over different hyper parameter settings for the gamma prior
    for ig = 1:length(gamma_interval) 
        % Loop over different hyper parameter settings for the alpha prior
        for ia = 1:length(alpha_interval)
            for rep = 1:repeats
                %% Run sparse analysis
                opts.gamma_a = gamma_interval(ig);
                opts.alpha_a = alpha_interval(ia);
                rng(repSEEDs(rep))
                gpuDevice(1);
                if strcmpi(method,'psFA')
                    opts.sparse_prior = true;
                elseif strcmpi(method,'pFA')
                    opts.sparse_prior = false;
                end

                 [first_moments,other_moments,priors,elbo] = psFA(Y,D,'opts',opts);

                save(sprintf('%s%s_D%i_rep%i_gamma_%s_alpha_%s',save_folder,save_name...
                ,D,rep,prior_labs_gam{ig},prior_labs_alpha{ia}),...
                'first_moments','other_moments','priors','elbo','opts','sess')
            
            end
        end
    end
end
