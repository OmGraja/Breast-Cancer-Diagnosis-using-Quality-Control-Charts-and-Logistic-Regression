% demos for ch04

%% Logistic logistic regression for binary classification
% clear; close all;
% k = 2;
% n = 1000;
% [X,t] = kmeansRnd(2,k,n);
Z=Z';
Y=Y';
[model, llh] = logitBin(Z,Y);
plot(llh);
Y = logitBinPred(model,Z)+1;
binPlot(model,Z,Y)
%% Logistic logistic regression for multiclass classification
% clear
% k = 3;
% n = 1000;
% [X,t] = kmeansRnd(2,k,n);
% [model, llh] = logitMn(X,t);
% y = logitMnPred(model,X);
% plotClass(X,y)
