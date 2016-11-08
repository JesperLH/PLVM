%% Checks for convergence issues
function [iter_alpha_fixed,iter_gamma_fixed,iter_tau_fixed] = ...
        asses_convergence(lbnorm,conv_crit,iter,...
                        model_alpha,model_gamma,model_tau,...
                        iter_alpha_fixed,iter_gamma_fixed,iter_tau_fixed)
if lbnorm < 0 && abs(lbnorm) < conv_crit
    warn_msg = sprintf('Lowerbound diverged at iteration %i by %6.4e .',iter,gather(lbnorm));
    param = {'alpha','gamma','tau'};
    is_modeled = [model_alpha,model_gamma,model_tau];
    [fixed, idx] = sort([iter_alpha_fixed,iter_gamma_fixed,iter_tau_fixed]);
    for i = 1:length(param)
        % If a parameter is modelled, but fixed until a certian iteration, 
        % then if the method diverges (below convergence criteria) this is 
        % likely the cause. Therefore the parameter is now modelled
        % (i.e. "unfixed")
        if iter < fixed(i) && is_modeled(idx(i))
            warn_msg = sprintf('%s Modeling of %s now commences (was initialy fixed until iteration %i).',warn_msg,param{idx(i)},fixed(i));
            
            if strcmp('alpha',param{idx(i)})
                iter_alpha_fixed = iter;
            elseif strcmp('gamma',param{idx(i)})
                iter_gamma_fixed = iter;
            elseif strcmp('tau',param{idx(i)})
                iter_tau_fixed = iter;
            else
                %Unsupported
            end
            break; %Only one parameter can be unfixed at a time
        end
    end
    warning(warn_msg)
elseif lbnorm < 0
    % If all parameters are modelled and the algorithm diverges, then this
    % can be due to; 1) Requiring too much precision (i.e. very small
    % conv_crit). 2) The specified problem might be unstable, fx. D >> T,V.
    % 3) Unknown issues
    warning(['Lowerbound diverged by %6.4e, this may be ',...
             'due to a numerical error, or input which is ',...
             'not "well behaved".'],lbnorm)
end
end