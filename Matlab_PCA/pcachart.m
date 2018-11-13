function [outliers, h] = pcachart(X,k)
%PCAchart for the k-th PC score.
%   PCACHART(X,k) produces an PCA chart of the n-by-p data matrix X
%
%   k (optional, k<=p) is the k-th PC. k is 1 by default.
%
%   OUTLIERS = PCACHART(X,k) returns indices of out of control points.
%
%   H = PCACHART(X,k) returns a vector of handles, H, to the plotted lines.

if nargin < 2
   k = 1; %first PC
end

[A,Z,lambda,Tsquare]=pca(X);
[n,p] = size(X);

UCL = 3*sqrt(lambda(k));
CL = 0;
LCL = -3*sqrt(lambda(k));

z = Z(:,k); %k-th PC score

incontrol = NaN(1,n);
outcontrol = incontrol;

greenpts = find(z > LCL & z < UCL);
redpts = find(z <= LCL | z >= UCL);

incontrol(greenpts) = z(greenpts);
outcontrol(redpts) = z(redpts);

samples = (1:n)';
hh  = plot(samples,z,samples,UCL(ones(n,1),:),'r-',samples,CL(ones(n,1),:),'g-',...
           samples,LCL(ones(n,1),:),'r-',samples,incontrol,'ko',samples,outcontrol,'ro');

dx = 0.5 * min(diff(samples));
if any(redpts)
  for i = 1:length(redpts)
     text(samples(redpts(i))+dx, outcontrol(redpts(i)),num2str(redpts(i)));
  end
end

text(samples(n)+dx,UCL,'UCL');
text(samples(n)+dx,LCL,'LCL');
text(samples(n)+dx,CL,'CL');  
hold on; plot(redpts,z(redpts),'ro');

if nargout > 0
  outliers = redpts;
end

if nargout == 2
 h = hh;
end         

set(hh([5 6]),'MarkerFaceColor',[192 255 62]/255,'MarkerSize',4);         

% Make sure all points are visible (must be done after setting tick labels
xlim = get(gca,'XLim');
set(gca,'XLim',[min(xlim(1),samples(1)-2*dx), max(xlim(2),samples(end)+2*dx)]);
hold on; hline1 = refline([0 UCL]); set(hline1,'Color','r')
hold on; hline2 = refline([0 LCL]); set(hline2,'Color','r')
hold on; hline3 = refline([0 CL]); set(hline3,'Color','g')
xlabel('Sample Number','fontsize',14,'fontname','times');
ylabel(['PC score #',int2str(k)],'fontsize',14,'fontname','times')


