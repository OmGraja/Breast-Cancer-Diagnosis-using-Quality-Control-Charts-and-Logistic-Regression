clear; close all;

load mydata.mat;

%Plotting of covariance matrix of waferdata using plottable.m (Q4b)
[n,p]=size(X);
%The data for many of the variables are strongly skewed to the right. 
Y=X(:,9);
X=X(:,2:p-1);
%scatterplot matrix of the data
[ha,ax,bigax,P]=plotmatrix(X);
axes(bigax);
delete(P); %delete the histograms
%Put the labels in.



for i=1:length(variables)
    txtax = axes('Position',get(ax(i,i),'Position'),'units','normalized');
    text(.35,.5,variables(i,:))
    set(txtax,'xtick',[],'ytick',[],'xgrid','off','ygrid','off','box','on')
end 



%  X = zscore(X); %standardize the data
figure; 
boxplot(X,variables);
figure;
boxplot(X,variables,'symbol',''); % discarding outliers from the display
 X = zscore(X); %standardize the data

% Center X by subtracting off column means
X0 = bsxfun(@minus,X,mean(X,1));
S = X0'*X0./(n-1); %Covariance matrix
%Display covariance matrix
figure; displaytable(S,variables);

%Applying PCA: 
% [A,Z,variance,Tsquare]=princomp(X) performs PCA on the n-by-p data matrix X, and returns the 
% principal component coefficients, also known as loadings. Rows of X correspond to observations, 
% columns to variables. A is a p-by-p matrix, each column containing coefficients for one principal 
% component. The columns are in order of decreasing component variance.
% Z=the principal component matrix scores; that is, the representation of X in the principal component space. 
% Rows of Z correspond to observations, columns to components.
% Variance= a vector containing the eigenvalues of the covariance matrix of X.
% Tsquare= contains Hotelling's T2 statistic for each data point.
% princomp centers X by subtracting off column means, but does not rescale the columns of X. 
% To perform principal components analysis with standardized variables, that is, based on correlations, 
% use princomp(zscore(X))

[A,Z,variance,Tsquare]=pca(X);

% PC2 coef vs. PC1 coef  
figure;
scatter(A(:,1),A(:,2),15,'ko','MarkerFaceColor',[.49 1 .63],'LineWidth',1);
%title('Scatter plot of 2nd PC vs. 1st PC');
xlabel('PC1 coefficient','fontsize',14,'fontname','times'); 
ylabel('PC2 coefficient','fontsize',14,'fontname','times');
text(A(:,1),A(:,2),variables,'VerticalAlignment','bottom','HorizontalAlignment','left')
%gname(variables); %press Enter or Escape key to stop labeling.

% PC3 coef vs. PC2 coef  
% figure;
% scatter(Ac(:,2),Ac(:,3),15,'ko','MarkerFaceColor',[.49 1 .63],'LineWidth',1);
% %title('Scatter plot of 2nd PC vs. 1st PC');
% xlabel('PC2 coefficient','fontsize',14,'fontname','times'); 
% ylabel('PC3 coefficient','fontsize',14,'fontname','times');

% PC3 coef vs. PC2 coef  
figure;
scatter(A(:,2),A(:,3),15,'ko','MarkerFaceColor',[.49 1 .63],'LineWidth',1);
%title('Scatter plot of 2nd PC vs. 1st PC');
xlabel('PC2 coefficient','fontsize',14,'fontname','times'); 
ylabel('PC3 coefficient','fontsize',14,'fontname','times');
text(A(:,2),A(:,3),variables,'VerticalAlignment','bottom','HorizontalAlignment','left')
%gname(variables); %press Enter or Escape key to stop labeling.

%Plotting Explained variance vs number of Principal Components (Q4d)
%using Plot and Pareto commands
expvar=100*variance/sum(variance);%percent of the total variability explained by each principal component.

figure;
plot(expvar,'ko-','MarkerFaceColor',[.49 1 .63],'LineWidth',1);
xlabel('Number of Principal Components','fontsize',14,'fontname','times');
ylabel('Explained Variance %','fontsize',14,'fontname','times');
%title('Scree Plot: Explained variance vs. Principal Component Number');

figure;
pareto(expvar);
xlabel('Number of Principal Components','fontsize',14,'fontname','times');
ylabel('Explained Variance %','fontsize',14,'fontname','times');
%title('Pareto of Explained variance vs. Principal Component Number');

%The first three principal components explain 87% of the variation. 
% This is an acceptably large percentage.

% PC2 score vs. PC1 score  (Q4f)
figure;
scatter(Z(:,1),Z(:,2),15,'ko','MarkerFaceColor',[.49 1 .63],'LineWidth',1);
%title('Scatter plot of 2nd PC vs. 1st PC');
xlabel('PC1 score','fontsize',14,'fontname','times'); 
ylabel('PC2 score','fontsize',14,'fontname','times');
text(Z(:,1),Z(:,2),observations,'VerticalAlignment','bottom','HorizontalAlignment','left')


%Biploy helps visualize both the principal component coefficients for each variable and the principal 
%component scores for each observation in a single plot.
biplot(A(:,1:2),'Scores',Z(:,1:2),'VarLabels',variables)
%xlabel('$Z_1$','fontsize',14,'fontname','times','Interpreter','LaTex'); 
%ylabel('$Z_2$','fontsize',14,'fontname','times','Interpreter','LaTex');
axis tight;

figure;
biplot(A(:,1:3),'Scores',Z(:,1:3),'VarLabels',variables)
% xlabel('$Z_1$','fontsize',14,'fontname','times','Interpreter','LaTex'); 
% ylabel('$Z_2$','fontsize',14,'fontname','times','Interpreter','LaTex');
% zlabel('$Z_3$','fontsize',14,'fontname','times','Interpreter','LaTex');
axis tight;


%Component correlation matrix
[Ac,Zc,variancec,Tsquarec]=pca(zscore(X));
C = Ac*sqrt(diag(variancec));
figure('Name','Component Correlation Matrix');
plottable(C,'%.2f');
set(gca,'LineWidth',1.2);
set(gca,'FontSize',12);
set(gca,'color',[.95 .95 .95],'XColor','white', 'YColor','white');
set(gcf,'color','white'); %camzoom(1.1); 
set(gcf,'InvertHardCopy','off'); 
set(gcf,'PaperPositionMode','auto');


figure;
scatter(Zc(:,1),Zc(:,2),15,'ko','MarkerFaceColor',[.49 1 .63],'LineWidth',1);
%title('Scatter plot of 2nd PC vs. 1st PC');
xlabel('PC1 score','fontsize',14,'fontname','times'); 
ylabel('PC2 score','fontsize',14,'fontname','times');

figure;
scatter(Zc(:,2),Zc(:,3),15,'ko','MarkerFaceColor',[.49 1 .63],'LineWidth',1);
%title('Scatter plot of 2nd PC vs. 1st PC');
xlabel('PC2 score','fontsize',14,'fontname','times'); 
ylabel('PC3 score','fontsize',14,'fontname','times');

% PC3 coef vs. PC2 coef  
figure;
scatter(Ac(:,2),Ac(:,3),15,'ko','MarkerFaceColor',[.49 1 .63],'LineWidth',1);
%title('Scatter plot of 2nd PC vs. 1st PC');
xlabel('PC2 coefficient','fontsize',14,'fontname','times'); 
ylabel('PC3 coefficient','fontsize',14,'fontname','times');


% text(Ac(:,1),Ac(:,2),variables,'VerticalAlignment','bottom','HorizontalAlignment','left')


%gname(variables); %press Enter or Escape key to stop labeling.
% gname(variables); %press the Enter or Escape key to stop labeling.
% hold on;
% scattercloud(A(:,1),A(:,2));

figure;
alpha = 0.05;
[outliers1, h1] = tsquarechart(X,alpha); %T^2 chart

figure;
k=1;
[outliers2, h2] = pcachart(X,k); %1st PCA control chart
ylabel('$Z_1$','fontsize',14,'fontname','times','Interpreter','LaTex'); 

% figure;
% i=3; j=4;
% plot(X(:,i),X(:,j),'ro','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',4);
% hold on; [h,s]=plotcov2(mean([X(:,i), X(:,j)])', cov(X(:,i),X(:,j)),'num-pts',1000,'plot-axes',0);
% xlabel('$X_1$','fontsize',14,'fontname','times','Interpreter','LaTex'); 
% ylabel('$X_2$','fontsize',14,'fontname','times','Interpreter','LaTex');
% text(X(:,i),X(:,j),countries,'VerticalAlignment','bottom','HorizontalAlignment','left')
% %gname(countries);

% 2D scatter plots of the principal component scores of data
% mapcaplot(X,observations);

