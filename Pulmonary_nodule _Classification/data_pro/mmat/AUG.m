



A=dir('H:\con\*.jpg');    %Õº∆¨∏Ò Ω

% load('name_total.mat');

subDir='H:\con\';

%  name=name_total;  

for i=1:numel(A)
    
    
    name{i,1}=A(i).name;
    
   
end

% save name_example name;

for v=1:numel(name)
    
    num_cut=strfind(name{v,1},'_');
    
    
    num_point=strfind(name{v,1},'.');
    
    name_reall{v,1}=name{v,1}(1:num_cut(end)-1);
    
    label_reall{v,1}=name{v,1}(num_cut+1:num_point-1);
    
    
    
end


squence=cell(4,1);
for j=1:4
    
    
    squence{j,1}=strcat('0',num2str(j));
    
    
end



count=1;

for k=1:numel(name)

    im=imresize(imread(strcat(subDir,name{k,1})),[224,224]);
    
    
%     im=imresize(im,[224,224]);
    
    imwrite(im,strcat('H:\im_aug\',[name_reall{k,1},'_',squence{1,1},'.jpg']));
    
%     im_1=imrotate(im,90);
    
    
    imwrite(imrotate(im,90),strcat('H:\im_aug\',[name_reall{k,1},'_',squence{2,1},'.jpg']));
    
    imwrite(imrotate(im,180),strcat('H:\im_aug\',[name_reall{k,1},'_',squence{3,1},'.jpg']));
    
    imwrite(imrotate(im,270),strcat('H:\im_aug\',[name_reall{k,1},'_',squence{4,1},'.jpg']));



end




  








