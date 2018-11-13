function [outliers, h] = tsquarechart(X,alpha)
%TSQUAREchart for the Hotelling T^2.
%   TSQUARECHART(X) produces a Hotelling T^2 chart of the n-by-p data matrix X
%
%   alpha (optional) is the significance level. alpha is 0.05 by default.
%
%   OUTLIERS = TSQUARECHART(X,alpha) returns indices of out of control points.
%
%   H = TSQUARECHART(X,alpha) returns a vector of handles, H, to the plotted lines.

if nargin < 2
   alpha = 0.05;
end

[A,Z,lambda,Tsquare]=pca(X);
[n,p] = size(X);

%UCL = p*(n-1)/(n-p)*icdf('f',1-alpha,p,n-p);
%UCL = p*(n+1)*(n-1)/(n^2-n*p)*icdf('f',alpha/2,p,n-p);
%UCL = icdf('chi2',1-alpha,p);
UCL = ((n-1)^2/n)*icdf('beta',1-alpha,p/2,(n-p-1)/2);

incontrol = NaN(1,n);
outcontrol = incontrol;

greenpts = find(Tsquare < UCL);
redpts = find(Tsquare >= UCL);

incontrol(greenpts) = Tsquare(greenpts);
outcontrol(redpts) = Tsquare(redpts);

samples = (1:n)';
hh  = plot(samples,Tsquare,samples,UCL(ones(n,1),:),'r-',...
           samples,incontrol,'ko',samples,outcontrol,'ro');

dx = 0.5 * min(diff(samples));
if any(redpts)
  for i = 1:length(redpts)
     text(samples(redpts(i))+dx, outcontrol(redpts(i)),num2str(redpts(i)));
  end
end

text(samples(n)+dx,UCL,'UCL'); 
hold on; plot(redpts,Tsquare(redpts),'ro');
hold on; hline1 = refline([0 UCL]); set(hline1,'Color','r')

if nargout > 0
  outliers = redpts;
end

if nargout == 2
 h = hh;
end         

% set(hh([3 4]),'MarkerFaceColor',[.49 1 .63],'MarkerSize',4);        
set(hh([3 4]),'MarkerFaceColor',[192 255 62]/255,'MarkerSize',4);  
xlabel('Sample Number','fontsize',14,'fontname','times');
ylabel('Hotelling T^2','fontsize',14,'fontname','times');
% Make sure all points are visible (must be done after setting tick labels
xlim = get(gca,'XLim');
set(gca,'XLim',[min(xlim(1),samples(1)-2*dx), max(xlim(2),samples(end)+2*dx)]);
hold on; hline2 = refline([0 UCL]); set(hline2,'Color','r')

