clear; close all;

SEED = 165441; %For reproducability
rng(SEED);
save_folder = './calcICASSP17/synthetic/';
mkdir(save_folder)
save_name = 'synthtetic_results';
analysis_methods = {'psFA','pFA','psPCA','pPCA'};
gpuDevice(1);

%% Choose synthetic data parameters
D = 3; % Subspace size
V = 1000; % Voxels
T = 25; % time in each subblock
Nsubs = 3; % number of subjects
noise_lvl = 1e-2;

type_of_data ='random-patterns'; 
%type_of_data ='saw-sin'; D =3;
type_of_noise = 'heteroscedastic'; %'homoscedastic';

%% Choose model parameters
opts.maxiter = 500;%2000;
opts.noise_process = true;
opts.ard_prior = true;
opts.sparse_prior = true;
opts.fixed_noise = 20;
opts.fixed_ard = 10;
opts.fixed_sparse = 5;
opts.runGPU = true;
repeats = 50;

% hyperparameters
opts.alpha_a = 1e-6;
opts.alpha_b = opts.alpha_a;
opts.gamma_a = opts.alpha_a;
opts.gamma_b = opts.alpha_a;
opts.tau_a = opts.alpha_a;
opts.tau_b = opts.alpha_a;


%% Generate Synthetic Data 
%Generate sources
A = orth(randn(V,D));
Asparse = A.*(rand(V,D)>0.5);
assert(all(sum(abs(A))>0),'One of the components has no non-zero values')
A = Asparse;
%Generate activation

if strcmpi(type_of_data,'random-patterns')
    sources = randn(T,D);
elseif strcmpi(type_of_data,'saw-sin')
    sources = nan(T,3);
    l = 2; %repeated pattern
    assert(mod(T,l) == 0,'Not devisible by the times the pattern is repeated')
    sources(:,1)=repmat(linspace(-pi,pi,T/l),1,l); 
    sources(:,2)=cos(4*repmat(linspace(-pi,pi,T/l),1,l)); 
    sources(:,3)=randn(T,1)'.^3; 
    %Same variance
    sources(:,1)=sources(:,1)/sqrt(var(sources(:,1)));
    sources(:,2) = sources(:,2)/sqrt(var(sources(:,2))); 
    sources(:,3) = sources(:,3)/sqrt(var(sources(:,3)));
    
end


%Determine noise variance
if strcmpi(type_of_noise,'homoscedastic')
    subj_noise_var = (0.1+rand(1,Nsubs))*noise_lvl;
elseif strcmpi(type_of_noise,'heteroscedastic')
    subj_noise_var = (0.9+rand(V,Nsubs)*0.2)*noise_lvl;
end

%Construct data
X = nan(V,T,Nsubs);
S = nan(D,T,Nsubs);
for n = 1:Nsubs
    if strcmpi(type_of_data,'saw-sin')
        S(:,:,n) = bsxfun(@times,sources,rand(1,3)*2+0.9)';%sources';
    elseif strcmpi(type_of_data,'random-patterns')
        S(:,:,n) = randn(D,T);
    end
    X(:,:,n) = A*S(:,:,n) + bsxfun(@times,randn(V,T),sqrt(subj_noise_var(:,n)));
end
X = bsxfun(@minus,X,mean(X,2)); % Zero mean

%% Run PCA analysis (Example should not be trivially solved by ordinary PCA)
%Y = reshape(X,size(X,1),size(X,2)*size(X,3));
Y2 = reshape(X,V,T*Nsubs)';
[coeff,score,latent] = pca(Y2);

figure, plot(1-latent./sum(latent)), axis([1,length(latent),0,1])
title('Ordinary PCA'); xlabel('Number of PCs'); ylabel('Variance explained')

opt_noc = find(1-latent./sum(latent)>0.95); opt_noc = opt_noc(1);
fprintf('Optimal number of components is %i . Having an RV coeff. of %1.4f\n',opt_noc,coeffRV(A,coeff(:,1:opt_noc)))
fprintf('If number of components is %i . with an RV coeff. of %1.4f\n',opt_noc-1,coeffRV(A,coeff(:,1:opt_noc-1)))
fprintf('If number of components is %i . with an RV coeff. of %1.4f\n',opt_noc+1,coeffRV(A,coeff(:,1:opt_noc+1)))

%% Run model multiple times with and without sparsity prior on S.
for meth = length(analysis_methods):-1:1
   all_methods(meth).name = analysis_methods{meth};
   all_methods(meth).first_moments = cell(repeats,1);
   all_methods(meth).other_moments = cell(repeats,1);
   all_methods(meth).priors = cell(repeats,1);
   all_methods(meth).elbo = cell(repeats,1);
end

repSEEDs = randi([1 10^7],repeats,1);
t1=tic;
for meth = 1:length(analysis_methods)
    opts_temp = opts;
    for rep = 1:repeats
        rng(repSEEDs(rep))
        if strcmpi(analysis_methods{meth},'psFA')%% Run sparse analysis
            opts_temp.sparse_prior = true;
            opts_temp.noise_process = true;
        elseif strcmpi(analysis_methods{meth},'pFA')%% Run "full" analysis
            opts_temp.sparse_prior = false;
            opts_temp.noise_process = true;
        elseif strcmpi(analysis_methods{meth},'psPCA')%% Run sparse analysis
            opts_temp.sparse_prior = true;
            opts_temp.noise_process = false;
        elseif strcmpi(analysis_methods{meth},'pPCA')%% Run "full" analysis
            opts_temp.sparse_prior = false;
            opts_temp.noise_process = false;
        else
            warning('Unknown method (%s) was not run',analysis_methods{meth})
        end
        
        [first_moments,other_moments,priors,elbo] = psFA(X,D+3,'opts',opts_temp);
        
        all_methods(meth).first_moments{rep} = first_moments;
        all_methods(meth).other_moments{rep} = other_moments;
        all_methods(meth).priors{rep} = priors;
        all_methods(meth).elbo{rep} = elbo;
    end
end
toc(t1)

save([save_folder,save_name],'all_methods','A','S','X','subj_noise_var','opts')