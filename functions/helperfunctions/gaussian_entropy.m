function entropy = gaussian_entropy(Sigma,d,isdiag,isMultivariate)

if nargin<3
    isdiag=false;
end
if nargin<4
    isMultivariate=true;
end
   if isdiag && isMultivariate
       entropy = 0.5*sum(log(Sigma))+d/2*(1+log(2*pi));
   elseif ~isMultivariate
       entropy = 0.5*d*log(Sigma(1))+d/2*(1+log(2*pi));
   else
       R=chol(Sigma);
       entropy = sum(log(diag(R)))+d/2*(1+log(2*pi));
   end
   %entropy = 1/2*log(det(Sigma));%+d/2*(1+log(2*pi));
end