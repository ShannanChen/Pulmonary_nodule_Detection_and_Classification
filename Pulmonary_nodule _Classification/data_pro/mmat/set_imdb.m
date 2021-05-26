

load('imdb_40.mat');

clear images;
load('name_reall.mat');
load('label_reall.mat');



squence=cell(4,1);

for j=1:4
 
    squence{j,1}=strcat('0',num2str(j));
     
end

count=1; 
for i=1:numel(name_reall)
    
    for k=1:numel(squence)
        
        name{1,count}=strcat(name_reall{i,1},'_',squence{k,1});
        
        count=count+1;
        
    end
end
    


images.name=name;

for v=1:numel(label_reall)
    
  if   strcmp(label_reall{v,1},'0')==1
      
       label(v,1)=str2num(label_reall{v,1});
  else
       label(v,1)=1;
  
  end
    
end

for j=1:numel(label)
    
    images.label(1,4*j-3)=label(j,1) ;   
    images.label(1,4*j-2)=label(j,1) ;
    images.label(1,4*j-1)=label(j,1) ;
    images.label(1,4*j)=label(j,1) ;
    
end



images.labels=images.label;


images.id=[1:numel(images.label)];



for  k=1:numel(label)
% images.set=

if   mod(k,10)==1    %index ³ý 10 Óà 1  £¬test     
    
     set(1,k)=2;
else
    set(1,k)=1;      %train 
end

end

for j=1:numel(set)
    
    images.set(1,4*j-3)=set(1,j) ;   
    images.set(1,4*j-2)=set(1,j) ;
    images.set(1,4*j-1)=set(1,j) ;
    images.set(1,4*j)=set(1,j) ;
    
end


save imdb classes images meta;














