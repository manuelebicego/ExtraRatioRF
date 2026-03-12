function DD = getExtraRatioRFDistance(Xdata,randADD,param,CI)
% DD = getExtraRatioRFDistance(Xdata,randADD,param,CI)
% compute ExtraRatioRF distance
%
% Xdata: [nxd]: n objects in a d-dimensional space
% L: number of added tests

nT = length(CI);
n = size(Xdata,1);
DD = zeros(n);
for t = 1:nT
    if (mod(t,10) == 0) || (t == 1) || (t == param.ntrees), if param.verbose, fprintf('Tree %d/%d\n',t,param.ntrees); end; end
    xcx = zeros(n);
    xlambda = zeros(n);
    beta = CI(t).beta;
    b = CI(t).b;
    dep = CI(t).depObj;
    % removing leaves
    d = (sum(b)==0);
    beta(:,d) = [];
    b(:,d) = [];
    dep = dep - 1;
    nb = size(beta,2);
    for i = 1:n
        r = sum(repmat(b(i,beta(i,:)),n,1) == b(:,beta(i,:)),2);
        xcx(i,:) = r';
        xlambda(i,:) = sum((repmat(beta(i,:),n,1) & beta),2);
    end
    den = repmat(dep',n,1) + repmat(dep,1,n) - xlambda;
    li = CI(t).leafid;
    num = (xcx+xcx'-xlambda +(li ~= li'));
    den = den+(den == 0);


    % adding random tests
    ntesttoADD = min(randADD/2,nb);
    % divided by two since we add half of the random tests
    % to the tests of one object, and half to the other
    ranxcx = zeros(n);
    for i = 1:n
        rantest = randperm(nb,ntesttoADD);
        r = sum(repmat(b(i,rantest),n,1) == b(:,rantest),2);
        ranxcx(i,:) = r';
    end
    rannum = ranxcx+ranxcx';
    randen = ones(n)*ntesttoADD*2;
    xDD3 = (num+rannum)./(den+randen);
    DD = DD+xDD3;
end
DD = DD./nT;
DD = 1-DD;


