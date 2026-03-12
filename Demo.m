% demo for computing the ExtraRatioRF distance
% paper "ExtraRatioRF: an extremely randomized Random Forest distance",
% submitted to ECML-PKDD 26

% the code requires the Random Forest Clustering library,
% available from the authors of RatioRF at their web page:
% https://profs.scienze.univr.it/~bicego/code.html
% We provide in the zip the version downloaded at the moment of the
% submission


clear 

% loading 1st dataset used
load('automobile.mat');

h = 6; % maximum depth of the tree


param = RFC_defaultParam;
param.verbose = 0;
param.typeTree = 2; % ERT
param.sampling = 256;
param.ntrees = 300;
param.maxdepth = h;
rfc = RFC_RFtrain(x,param);
CI = RFC_getRFClusInfo(x,rfc,param);

% To compute the original RatioRF
param.dist = 5; % ratioRF
RatioRFDistance = RFC_getRFDist(x,[],param,CI);

% to compute the proposed ExtraRatioRF
L = round(sqrt(2)*h); % Number of random tests to add 
% (from remark 1)

rng('default'); 
ExtraRatioRFDistance = getExtraRatioRFDistance(x,L,param,CI);

