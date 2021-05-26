function  matrix=spacial(img)
    %%%%%%计算spacial 权重
    %所有通道相加，S
    %所有元素做a次方，求矩阵元素之和，得到一个数并做根号a次方   z
    %得到一个矩阵：（S/z）××（1/b）
    %%%%%%%%%%%%%%%%%%首先针对一张图片的所有特征
    S=single(sum(img,3));
    SS=single((S.^2));
    z=single(sum(SS(:)).^(1./2));
    matrix=single((S./z).^(1./2));
    %%%%%%%%%%%%%%%%%%%comple spacial 权重
