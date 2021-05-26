function [net, info] = cnn_imagenet_nodule(varargin)
%CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%  This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%  VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.

% run(fullfile(fileparts(mfilename('D:\matconvnet-1.0-beta24')), ...
%   '..', '..', 'matlab', 'vl_setupnn.m')) ;
 
%%%%%%%%%%%%%%%%%%%%#################################################

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%1212进行筛选四倍交叉验证%%%%%%%%%%%
  
  %%%%%%%%%%%%#########################################################
   % run(fullfile('D:\matconvnet-1.0-beta24', ...
%         'matlab', 'vl_setupnn.m')) ;
path = '/home/cad/zyc/chengxu/matconvnet-1.0-beta22bcn/examples/imagenet/';
%opts.dataDir = fullfile(path, 'data', 'cpmpete32*32train') ; 
opts.dataDir ='/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/交叉验证/3/all224/' ;     %'/home/cad/zyc/shuju/64/secall/sec1207/';
% opts.modelType = 'imagenet-vgg-f-fortwo' ;                           %%%---%%%
opts.expDir='/home/cad/zyc/chengxu/matconvnet-1.0-beta22bcn/examples/imagenet/data/';
% opts.modelType = 'vgg-f-fortwo' ;     
opts.modelType = 'vgg-f';%'imagenet-vgg-verydeep-16' ; %%%===
opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;


[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
opts.expDir = fullfile(path, 'data', ['im_aug-' sfx],'reset','1173-42rgb') ;     %%%---
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
%opts.imdbPath = fullfile(opts.expDir, 'complete32*32train.mat');
%opts.imdbPath = fullfile(opts.expDir, 'allsametrain224three91.mat');
%opts.imdbPath =fullfile('/home/cad/zyc/shuju/1201/precdata/aug1202','imdbaug1202.mat');%';
%opts.imdbPath = fullfile('/home/cad/zyc/shuju/1204/1204/','1204aug.mat')
opts.imdbPath ='/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/交叉验证/3/rgb42imdb3.mat'    %opts.imdbPath = fullfile('/home/cad/zyc/shuju/64/secall','sec1207.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [2]; end;

                                           %%%%%%%%%%%%%%%%%%%%%%%%%%

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
%   imdb = load(opts.imdbPath) ;
  load(opts.imdbPath) ;
  imdb.imageDir = fullfile(opts.dataDir, '');                          %%%---
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

%Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  train = find(imdb.images.set == 1) ;
  images = fullfile(imdb.imageDir, imdb.images.name(train(1:100:end))) ;
  [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
                                                    'imageSize', [224 224], ...
                                                    'numThreads', opts.numFetchThreads, ...
                                                    'gpus', opts.train.gpus) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end
[v,d] = eig(rgbCovariance) ;
rgbDeviation = v*sqrt(d) ;
clear v d ;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

if isempty(opts.network)
  switch opts.modelType
    case 'resnet-50'
    net = cnn_imagenet_init_resnet('averageImage', rgbMean, ...
                                     'colorDeviation', rgbDeviation, ...
                                     'classNames', imdb.classes.name, ...
                                     'classDescriptions', imdb.classes.description) ;
      opts.networkType = 'dagnn' ;
      
    case 'vgg-f'                                      %%%---               
         net=load('vgg-f.mat');                %%%=====
         
% % % %%%%%%%create_layer%%%%%%%%
%         net.layer=cell(1,22);
%         net.layer=net.layers(1:8);
%         net.layer{9}.name='create_layer';
%         net.layer{9}.type = 'addlayer';
%         for i=1:11
%             net.layer{i+9}=net.layers{i+8};
%         end
%         net.layer{21}.stride = [1,1];
%         net.layer{21}.type = 'conv';
%         net.layer{21}.name = 'fc8';
%         net.layer{21}.weights{1} = 0.01*randn(1,1,4096,2,'single');
%         net.layer{21}.weights{2} = ones(2,1,'single');
%         net.layer{21}.opts = {};
%         net.layer{21}.dilate = 1;
%         net.layer{21}.size=[1,1,4096,2];              %%%---%%%
%         net.layer{21}.precious=false;
%         net.layer{22}.precious=false;
%         net.layer{22} = struct('type', 'softmaxloss') ;
%         net.layer{22}.name = 'prob';
%         clear net.layers
%         net.layers=cell(1,22);
%         net.layers=net.layer;
%         clear net.layer
%         net = vl_simplenn_tidy(net) ;imageStatsPath
% 
% %         
% % %%%%%%%%%%%%%%%%%%%%%      
            net.layers=net.layers(1:19);
            net.layers{20}.pad = [0,0,0,0];
            net.layers{20}.stride = [1,1];
            net.layers{20}.type = 'conv';
            net.layers{20}.name = 'fc8';
            net.layers{20}.weights{1} = 0.001*randn(1,1,4096,2,'single');    % 0.01改为0.005
            net.layers{20}.weights{2} = ones(2,1,'single');
            net.layers{20}.opts = {};
            net.layers{20}.dilate = 1;
            net.layers{20}.size=[1,1,4096,2];              %%%---%%%
            net.layers{21}.precious=false;   %vgg-f上保留
            net.layers{21} = struct('type', 'softmaxloss') ;
            net.layers{21}.name = 'prob';
            %%%%%%%%%原本的向上%%%%%%%%%%%
            %%%%%%%%%%%%%公共的与加层的同样使用
            net = vl_simplenn_tidy(net) ;
            net.meta.inputSize = [224,224,3, 32] ;
            % net.meta.normalization.cropSize = 1;
            net.meta.normalization.cropSize = 1;
            
            load(fullfile(opts.expDir, 'imageStats.mat'));%%%-======--%%%
            net.meta.normalization.averageImage =averageImage ;
            net.meta.classes.name ={'benign','malignant'} ;
            net.meta.classes.description = {'benign';'malignant'};
            net.meta.augmentation.jitterLocation = true ;                    %%%--opt--%%%
            net.meta.augmentation.jitterFlip = false ;
            % net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
            net.meta.augmentation.jitterBrightness = double(0.1 * rgbCovariance) ; 
            net.meta.augmentation.jitterAspect = [2/3, 3/2] ;  
                                    %%%%%%%%        
            net.meta.trainOpts.batchSize=64;%100%256;                     %%%-------learningRate-----
            net.meta.trainOpts.learningRate =logspace(-4,-8.5,150);%[0.00005*ones(1,20), 0.00005*ones(1,150)]%logspace(-3,-6,100);% [logspace(-4,-6,30),0.00005*ones(1,10)];%[0.001 * ones(1,30), 0.001*ones(1,20), 0.0005*ones(1,50)];%logspace(-3,-6,150) ;%[0.001 * ones(1,20), 0.0001*ones(1,5), 0.00001*ones(1,10)]; %%[0.0001 * ones(1,20), 0.0001*ones(1,5), 0.000001*ones(1,10)] %[0.0001 * ones(1,20), 0.0001*ones(1,5), 0.00005*ones(1,10)] ;% [0.001 * ones(1,20), 0.001*ones(1,5), 0.0005*ones(1,10)] ;
            net.meta.trainOpts.weightDecay =0.0005 ; %0.0003 ;
            net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
                          %%%%%%%%

      case  'imagenet-vgg-verydeep-16'
            net=load('imagenet-vgg-verydeep-16.mat');
            net.layers=net.layers(1:35);

            %add layer
            net.layers{36}.pad = [0,0,0,0];
            net.layers{36}.stride = [1,1];
            net.layers{36}.type = 'conv';
            net.layers{36}.name = 'fc8';
            net.layers{36}.weights{1} = 0.005*randn(1,1,4096,2,'single');
            net.layers{36}.weights{2} = ones(2,1,'single');
            net.layers{36}.opts = {};
            net.layers{36}.dilate = 1;
            net.layers{36}.size=[1,1,4096,2];              %%%---%%%
            net.layers{36}.precious=false;
            net.layers{37}.precious=false;   %vgg-f上保留
            net.layers{37} = struct('type', 'softmaxloss') ;
            net.layers{37}.name = 'prob';
            net.layers{37}.weights={};
            net = vl_simplenn_tidy(net) ;
            net.meta.inputSize = [224,224,3, 32] ;
            % net.meta.normalization.cropSize = 1;
            net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ;
            load(fullfile(path, 'data/im_aug-vgg-f-bnorm-simplenn/reset/1173-42rgb','imageStats.mat'));%%%-======--%%%
            net.meta.normalization.averageImage =averageImage ;
            net.meta.classes.name ={'benign','malignant'} ;
            net.meta.classes.description = {'benign';'malignant'};
            net.meta.augmentation.jitterLocation = true; %false ;                    %%%--opt--%%%
            net.meta.augmentation.jitterFlip = true ;
            % net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
            net.meta.augmentation.jitterBrightness = double(0.1 * rgbCovariance) ;
            net.meta.augmentation.jitterAspect = [2/3, 3/2] ;
            net.meta.trainOpts.batchSize=256;%100%256;   
            net.meta.trainOpts.numSubBatches = 1; 
            %%%-------learningRate-----
            net.meta.trainOpts.learningRate =[0.0001*ones(1,40),0.00005*ones(1,15),0.00001*ones(1,15)];% [logspace(-4,-6,30),0.00005*ones(1,10)];%[0.001 * ones(1,30), 0.001*ones(1,20), 0.0005*ones(1,50)];%logspace(-3,-6,150) ;%[0.001 * ones(1,20), 0.0001*ones(1,5), 0.00001*ones(1,10)]; %%[0.0001 * ones(1,20), 0.0001*ones(1,5), 0.000001*ones(1,10)] %[0.0001 * ones(1,20), 0.0001*ones(1,5), 0.00005*ones(1,10)] ;% [0.001 * ones(1,20), 0.001*ones(1,5), 0.0005*ones(1,10)] ;
            net.meta.trainOpts.weightDecay =0.0005 ; %0.0003 ;
            net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ; 
    otherwise
      net = cnn_imagenet_init('model', opts.modelType, ...
                              'batchNormalization', opts.batchNormalization, ...
                              'weightInitMethod', opts.weightInitMethod, ...
                              'networkType', opts.networkType, ...
                              'averageImage', rgbMean, ...
                              'colorDeviation', rgbDeviation, ...
                              'classNames', imdb.classes.name, ...
                              'classDescriptions', imdb.classes.description) ;
  end
else
  net = opts.network ;
  opts.network = [] ;
end

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainFn = @cnn_train%_cancer ;                          %%%%%%%%%%%%%%%%%
  case 'dagnn', trainFn = @cnn_train_dag ;
end
imdb.images.label=imdb.images.labels;
[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat')

switch opts.networkType
  case 'simplenn'
    save(modelPath, '-struct', 'net') ;
  case 'dagnn'
    net_ = net.saveobj() ;
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

if numel(meta.normalization.averageImage) == 3
  mu = double(meta.normalization.averageImage(:)) ;
else
  mu = imresize(single(meta.normalization.averageImage), ...
                meta.normalization.imageSize(1:2)) ;
end

useGpu = numel(opts.train.gpus) > 0 ;

bopts.test = struct(...
  'useGpu', useGpu, ...
  'numThreads', opts.numFetchThreads, ...
  'imageSize',  meta.normalization.imageSize(1:2), ...
  'cropSize', meta.normalization.cropSize, ...
  'subtractAverage', mu) ;

% Copy the parameters for data augmentation
bopts.train = bopts.test ;
for f = fieldnames(meta.augmentation)'
  f = char(f) ;
  bopts.train.(f) = meta.augmentation.(f) ;
end

fn = @(x,y) getBatch(bopts,useGpu,lower(opts.networkType),x,y) ;

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;
else
  phase = 'test' ;
end
data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
if nargout > 0
  labels = imdb.images.label(batch) ;
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end

