net = load('/media/cad/c5290228-7b88-43ba-809c-870ee6577d2c/data/zyc/chengxu/matconvnet-1.0-beta22bcn/examples/imagenet/vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;

root='';
imgpath=dir(root,'train','*.png');

vec=zeros(length(imgpath),4096);

for i=1:imgpath
    % Obtain and preprocess an image.
    im = imread(strcat(root,'train',imgpath(i).name)) ;
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;

    % Run the CNN.
    res = vl_simplenn(net, im_) ;

    train = res(19).x;
    vec(i,4096)=train;

end
   save vec.mat