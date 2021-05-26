function imdb = changedog_get_database_train_val(dogDir)
% 
% SQL 20170703
%

% imageDir
% [label, ~] = textread(fullfile(dogDir, 'data_train.txt'), '%d %s');
% classes = unique(label);
imdb.imageDir = dogDir;

%train--- image names, label, id, url

fnames = dir(fullfile(dogDir,'same364', '*.jpg'));
imageFileList = cell( length(fnames),1);
label = zeros( length(fnames),1);
for f = 1: length(fnames)
    imageFileList{f} = fnames(f).name;
    label(f,1) =str2double(imageFileList{f}(1));
end
classes = unique(label);
% [imageNames, id, url] = textread(fullfile(dogDir, 'train_name_id_url.txt'), '%s %d %s');
% name = strcat(imageNames,'.jpg');
% c= intersect(imageFileList, name); 
% [bool,index]=ismember(imageFileList,name);
imageFileList = imageFileList';
imdb.images.name = imageFileList;
imdb.images.id = 1:numel(imdb.images.name);
imdb.images.labels= label;

% imdb.images.set = ones(1, numel(imdb.images.name)) ;

% set, 1 for train, 3 for val
total = 0;
imdb.images.set = zeros(1,numel(imdb.images.name));
for i = 1 : 2
    image_perclass = numel(find(imdb.images.labels==i));
    train_image_perclass = ceil(image_perclass *9/10);  %2:1% train : val = 9 : 1
    
    randomized_idx = randperm(image_perclass);    
    train_image_perclass_idx = total + randomized_idx(1:train_image_perclass); % randomly split the train and val sets 
    val_image_perclass_idx = total + randomized_idx(train_image_perclass+1:end);
    
    imdb.images.set(train_image_perclass_idx) = 1;
    imdb.images.set(val_image_perclass_idx) = 2;
    total = total + image_perclass;
end

% meta
imdb.meta.sets = {'train', 'val','test'};
imdb.meta.classes = classes;


