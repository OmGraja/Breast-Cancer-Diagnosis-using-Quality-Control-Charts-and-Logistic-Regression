clear; 
[X, text, alldata] = xlsread('Breast.xlsx');
variables = char(text(1,2:end-1));
observations = char(text(2:end,1));
save mydata.mat X variables observations
Z=X(:,2:8);
Y=X(:,9);
Z=zscore(Z);
Y(find(Y==2))=0;
Y(find(Y==4))=1;
