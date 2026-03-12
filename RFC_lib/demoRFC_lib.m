% Demo for Random Forest Clustering library
% It contains the code for the experiments in
% "RatioRF: a novel measure for Random Forest 
% clustering based on the Tversky's Ratio model",
% M. Bicego, F. Cicalese, A. Mensi,
% submitted to IEEE TKDE, 2021
% 
% (c) Manuele Bicego 
% Last update: April 2021
% 

% loading data 
% Example: the iris dataset
rng('default'); 
load('iris.mat');
nclus = 3;

% load default parameters
% help RFC_defaultParam for all options
param = RFC_defaultParam;

% Step1: training RF
rfc = RFC_RFtrain(data,param);
% Step2: getting distance
CI = RFC_getRFClusInfo(data,rfc,param);
DD = RFC_getRFDist(data,rfc,param,CI);
% Step3: clustering
% here we used the Spectral Clustering
% with normalization of Jordan and Weiss
% help RFC_disClus for more options
T = RFC_disClus(DD,nclus,3);

% Check the results
% in labels we have the true labeling
ARI = RandIndex(T,labels);

fprintf('Adjusted Rand Index: %.4f\n',ARI);


