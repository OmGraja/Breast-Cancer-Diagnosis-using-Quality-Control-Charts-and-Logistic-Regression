function displaytable(M,labels)
%Plot table data with colored cells
%==============
%Example usage:
%==============
% M = rand(10,10);
% labels = cellstr(num2str((1:length(M))','X%d')); %node labels
% displaytable(M,labels)

if nargin<1
    help displaytable
    return
end

if nargin < 2
    labels = num2str((1:length(M))');
end    
[r c] = size(M);
imagesc(M);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)                         
n=length(M);
textStrings = num2str(M(:),'%0.3f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:n);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'FontSize',8,'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(M(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors
       
set(gca, 'Box','on', 'XAxisLocation','top', 'YDir','reverse', ...
    'XLim',[0 c]+0.5, 'YLim',[0 r]+0.5, 'TickLength',[0 0], ...
    'XTick',1:c, 'YTick',1:r, ...
    'XTickLabel',labels, 'YTickLabel',labels, ...
    'LineWidth',1.2, 'Color','none',...
    'FontWeight','bold', 'FontSize',8, 'DataAspectRatio',[1 1 1]);

%# plot grid    
xv1 = repmat((2:c)-0.5, [2 1]); xv1(end+1,:) = NaN;
xv2 = repmat([0.5;c+0.5;NaN], [1 r-1]);
yv1 = repmat([0.5;r+0.5;NaN], [1 c-1]);
yv2 = repmat((2:r)-0.5, [2 1]); yv2(end+1,:) = NaN;
line([xv1(:);xv2(:)], [yv1(:);yv2(:)], 'Color','k', 'HandleVisibility','off')