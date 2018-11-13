function  PlotMyData( Z,y )
%PLOTMYDATA Summary of this function goes here
%   Detailed explanation goes here
mal=find(y==4);
ben=find(y==2);

% plot(X(mal,1),X(mal,2),15,'ko','Markersize',5);
% hold on
% plot(X(ben,1),X(ben,2),15,'ko','Markersize',5);
% xlabel('PC1 score','fontsize',14,'fontname','times'); 
% ylabel('PC2 score','fontsize',14,'fontname','times');

scatter(Z(mal,1),Z(mal,2),15,'kx','MarkerFaceColor',[0 .5 .5],'LineWidth',1);
hold on
scatter(Z(ben,1),Z(ben,2),15,'ko','MarkerFaceColor',[.49 1 .63],'LineWidth',2);
%title('Scatter plot of 2nd PC vs. 1st PC');
xlabel('PC1 score','fontsize',14,'fontname','times'); 
ylabel('PC2 score','fontsize',14,'fontname','times');
% text(Z(:,1),Z(:,2),observations,'VerticalAlignment','bottom','HorizontalAlignment','left')

end

