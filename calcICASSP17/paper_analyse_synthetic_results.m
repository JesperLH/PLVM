%% Results Analysis script
clear
load('./calcICASSP17/synthetic/synthtetic_results.mat')
[V,T,B] = size(X);
D = size(A,2);
f_size = 16; %fontsize in figures (titles are f_size+2)
save_folder = './calcICASSP17/synthetic/'; %Saves files at this location
%save_folder = 'no'; %Doesn't save
if ~strcmpi(save_folder,'no')
    mkdir([save_folder 'png'])
    mkdir([save_folder 'eps'])
end

%% PCA analysis
Y = reshape(X,size(X,1),size(X,2)*size(X,3));
[coeff,score,latent] = pca(Y');
%opt_noc = find(1-latent./sum(latent)>0.95); opt_noc = opt_noc(1);
opt_noc = size(A,2);
pca_score = reshape(score(:,1:opt_noc)',opt_noc,size(X,2),size(X,3));

%% ICA analysis
interval = 1:size(X,2)*size(X,3);
Xcat = reshape(X,size(X,1),size(X,2)*size(X,3))';

fprintf('ICA with 3 components....'); tic
[Sica3,Aica3,U] = icaML( Xcat , 3 , [] , 1);
Aica3 = U(:,1:3) * Aica3;
toc;

fprintf('ICA with 6 components....'); tic
[Sica6,Aica6,U] = icaML( Xcat , 6 , [] , 1);
Aica6 = U(:,1:6) * Aica6;
toc

%% Show all elbo
figure; hold all
repeats = length(all_methods(1).first_moments);
nMethods = length(all_methods);
end_elbo = nan(nMethods,repeats);
analysis_methods = {all_methods.name};
for meth = 1:nMethods
    for rep = 1:repeats
        end_elbo(meth,rep) = all_methods(meth).elbo{rep}(end);
    end
end
bar(sign(end_elbo).*log10(abs(end_elbo)))
logy_tick = get(gca,'Ytick');
logy_tick = sign(logy_tick).*(10.^(abs(logy_tick)));
set(gca,'Yticklabel',logy_tick); ylabel('Evidence Lowerbound (ELBO)')
set(gca,'Xtick',1:nMethods,'Xticklabel',analysis_methods); xlabel('Methods')
legend(strcat(strsplit(num2str(1:repeats)),'. run'),'Location','BestOutside')
title('Comparison of achieve evidence lowerbound')
%%
[max_elbo,idx_best] = max(end_elbo,[],2);
Nsubs = size(S,3);
%% Todo, do some matching
f_size = f_size+5;
%% Show histogram + amari and spatial correlation with true A (for GSFA,GFA,PCA,ICAML)
% These are the illustrations shown in the ICASSP17 paper. The
% corresponding amari distance and rv-coefficient can be found in the
% "save_folder" where files are named *METHOD*_distance.txt
bins=25;
noc_plot=3;

for meth = 1:nMethods 
    Aest = all_methods(meth).first_moments{idx_best(meth)}.A;
    
    [amari,spatial_corr,~] = ...
        histAmariCorr(A,Aest(:,1:noc_plot),all_methods(meth).name,...
                      'save',save_folder,'fontsize',f_size,'bins',bins); %#ok<ASGLU>
end
%ICA with 3 components
[amari,spatial_corr,~] = ...
        histAmariCorr(A,Sica3','ica3',...
                      'save',save_folder,'fontsize',f_size,'bins',bins); %#ok<ASGLU>
% ICA with 6 components
[amari,spatial_corr,~] = ...
        histAmariCorr(A,Sica6(1:noc_plot,:)','ica6',...
                      'save',save_folder,'fontsize',f_size,'bins',bins); %#ok<ASGLU>

% PCA
[amari,spatial_corr,~] = ...
        histAmariCorr(A,coeff(:,1:noc_plot),'pca',...
                      'save',save_folder,'fontsize',f_size,'bins',bins); %#ok<ASGLU>

% True components                  
[amari,spatial_corr,~] = ...
        histAmariCorr(A,A,'generated',...
                      'save',save_folder,'fontsize',f_size,'bins',bins);
