function ad = amariDist(X,Y,q)

if ~any(size(X)==q)
    error('X must have q components')
elseif ~any(size(Y)==q)
    error('Y must have q components')
end
    
if size(X,1) ~= q
    X = X';
end
if size(Y,1) ~= q
    Y = Y';
end
%max(Q),[],2) % max af i over all j
%max(Q),[],1) % max af j over all i
Q = abs(X*pinv(Y));

%ad = sum(sum(Q,2).*max(Q,[],2)'-1)...
%    +sum(sum(Q,1).*max(Q,[],1)-1);
ad = sum(sum(Q,2)./max(Q,[],2)-1)...
    +sum(sum(Q,1)./max(Q,[],1)-1);
ad = 1/(q)*ad;