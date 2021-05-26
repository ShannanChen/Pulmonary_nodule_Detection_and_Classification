function  BB=channel(img)
     BB=single(ones(1,size(img,3)));
    area=single(size(img,1)*size(img,2));
    for c=1:size(img,3)      %%%%%%%%%将第bs个共c通道的2维图像
    %             img=img+res(:,:,c,bs);    %求c个通道的所有图像对应元素相加得到一个二维的矩阵
           B(1,c)=single(length(find(img(:,:,c)))./area); %将c个通道的所有图像中每个图像求大于0的比例
    end
     Bsum=single(sum(B));
     for j=1:length(B)
         if B(1,j)==0
             BB(1,j)=0;
         else
            BB(1,j)=single(log(Bsum./B(1,j)));
        end
     end
