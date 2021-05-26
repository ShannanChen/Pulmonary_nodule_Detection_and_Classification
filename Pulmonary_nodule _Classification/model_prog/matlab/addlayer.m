function n=addlayer(res)
a=2;
b=2;
%%%%%%%%note:matlab 中第三维为通道数
res = gather(res);
m=single(zeros(size(res,1),size(res,2),size(res,3)));  %receive feature map
n= zeros(size(res,1),size(res,2),size(res,3),size(res,4)); %receive res
A=single(zeros(size(res,1),size(res,3)));  %每一个通道对应一维
for bs=1:size(res,4)      %共bs个数据
    img=single(zeros(size(res,1),size(res,2),size(res,3))); %一张图像的大小
    img=res(:,:,:,bs);  
    PP=single(spacial(img));      %图片大小的二维矩阵
    CC=single(channel(img));    %一个通道长度的一维向量
    for j=1:size(res,3)
        m(:,:,j)=img(:,:,j).*PP.*CC(1,j);     
    end
    %n(:,:,:,bs)=gpuArray(single(m));
    n(:,:,:,bs)= m;
end
 n =gpuArray(single(n));  %在测试是删掉gpuarray  #gpuArray(
end
    
%     %%%%%%计算spacial 权重
%     %所有通道相加，S
%     %所有元素做a次方，求矩阵元素之和，得到一个数并做根号a次方   z
%     %得到一个矩阵：（S/z）××（1/b）
%     %%%%%%%%%%%%%%%%%%首先针对一张图片的所有特征
%     S=sum(img,3);
%     SS=(S^a);
%     z=sum(SS(:))^(1./a);
%     matrix(:,:,bs)=(S./z)^(1./b);
%     %%%%%%%%%%%%%%%%%%%comple spacial 权重
    %%%%%%%%%%%%%%%%计算channel 权重   一个向量
%     BB=ones(1,size(res,3));
%     CC=ones(size(res,4),size(res,3));
% area=size(res,1)*size(res,2);
% for bs=1:size(res,4) 
%     imgc=res(:,:,:,bs);
%     for c=1:size(res,3)      %%%%%%%%%将第bs个共c通道的2维图像
% %             img=img+res(:,:,c,bs);    %求c个通道的所有图像对应元素相加得到一个二维的矩阵
%             B(1,c)=length(find(res(:,:,c,bs)))./area; %将c个通道的所有图像中每个图像求大于0的比例
%             img=img+res(:,:,c,bs);
%     end
%     Bsum=sum(B);
%     for j=1:length(B)
%         BB(1,j)=log(Bsum./B(1,j));
%     end
%     CC(size(res,4),:)=BB;
% crow=
    
%       img=single(img);
%     B=single(zeros(1,size(res,3)));                    
%     B=single(B);
%     area=size(res,1)*size(res,2);
%         for c=1:size(res,3)      %%%%%%%%%将第bs个共c通道的2维图像
%             img=img+res(:,:,c,bs);    %求c个通道的所有图像对应元素相加得到一个二维的矩阵
%             
%             B(1,c)=1./log(length(find(res(:,:,c,bs)))./area); %将c个通道的所有图像中每个图像求大于0的比例
%             img=img+res(:,:,c,bs);
%         end
%       
%       A=img./sum(img(:));
%          
%         for  c=1:size(res,3) 
%            m(:,:,c)=B(1,c)*A*res(:,:,c,bs);    %求一个bs的处理后的c为图像B(1,c).*
%         end
%      n(:,:,:,bs)=m;    %将共size(res,4)的bs存储到n中
% end
% n = gpuArray(n);
% end
%            
%         
% 
% 
%    
