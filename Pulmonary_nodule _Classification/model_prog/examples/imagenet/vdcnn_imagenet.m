function [net, info] = vdcnn_imagenet(varargin)
opts.dataDir ='/home/cad/zyc/shuju/所有处理的数据形式1208/roundlinshi/augxyzroundmin';%'/home/cad/zyc/shuju/所有处理的数据形式1208/linshi/all/';% fullfile('/home/cad/zyc/20171018handledata/last1116/1125/all','/');
% opts.modelType = 'imagenet-vgg-f-fortwo' ;                           %%%---%%%
% opts.modelType = 'vgg-f-fortwo' ;     
opts.modelType = 'imagenet-vgg-verydeep-16' ; %%%===
opts.network = [] 
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;
sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
     %%%---
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
%opts.imdbPath = fullfile(opts.expDir, 'same364.mat');
%opts.imdbPath = fullfile('D:\zyc\last1117\caise', 'mulwuchufu1119.mat');
%opts.imdbPath =fullfile('/home/cad/zyc/shuju/1201/12345/','1202.mat'); %fullfile('/home/cad/zyc/20171018handledata/last1116/1125/','1125pepole123.mat')
%opts.imdbPath =fullfile('/home/cad/zyc/shuju/1201/precdata/aug1202','imdbaug1202.mat');%';
%opts.imdbPath = fullfile('/home/cad/zyc/shuju/1205/zxyz','1205.mat');
%opts.imdbPath ='xyzminroud1209aug'%'/home/cad/zyc/shuju/所有处理的数据形式1208/linshi/newhandquanwu.mat'
opts.imdbPath =fullfile('/home/cad/zyc/shuju/所有处理的数据形式1208/roundlinshi/xyzminroud1209aug.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [2]; end;

                                           %%%%%%%%%%%%%%%%%%%%%%%%%%

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
  imdb=imdb.imdb;
  imdb.images.label=imdb.images.labels;
  imdb.imageDir = fullfile(opts.dataDir, '');                          %%%---
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
 
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Compute image statistics (mean, RGB covariances, etc.)
opts.expDir='/home/cad/zyc/chengxu/matconvnet-1.0-beta22bcn/examples/imagenet/data/im_aug-imagenet-vgg-verydeep-16-bnorm-simplenn';
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
    case 'imagenet-vgg-verydeep-16'
        net=load('imagenet-vgg-verydeep-16.mat');  
        net.layers=net.layers(1:35);
        net.layers{36}.pad = [0,0,0,0];
        net.layers{36}.stride = [1,1];
        net.layers{36}.type = 'conv';
        net.layers{36}.name = 'fc8';
        net.layers{36}.weights{1} = 0.001*randn(1,1,4096,2,'single');
        net.layers{36}.weights{2} = 0.001*randn(2,1,'single');      %1119��weight��0.001���ﵽ0.2����Ϊ0.0005
        net.layers{36}.opts = {};
        net.layers{36}.dilate = 1;
        net.layers{36}.size=[1,1,4096,2];              %%%---%%%
        net.layers{36}.precious=false;
        net.layers{37}.precious=false;
        net.layers{37} = struct('type', 'softmaxloss') ;
        net.layers{37}.name = 'prob';
        net.layers{37}.weights={};
        net = vl_simplenn_tidy(net) ;
        
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%addlayer
%         % % %%%%%%%create_layer%%%%%%%%
%         net=load('imagenet-vgg-verydeep-16.mat');  
%         net.layer=cell(1,38);
%         net.layer=net.layers(1:24);%net.layer=net.layers(1:8);
%         net.layer{25}.name='create_layer'; %9
%         net.layer{25}.type = 'addlayer';         %9
%         for i=1:(35-24)%1:11   %20-4=16%20-11=9
%             net.layer{i+18}=net.layers{i+17};
%         end
%         
%         net.layer{37}.pad = [0,0,0,0];
%         net.layer{37}.stride = [1,1];
%         net.layer{37}.type = 'conv';
%         net.layer{37}.name = 'fc8';
%         net.layer{37}.weights{1} = 0.005*randn(1,1,4096,2,'single');
%         net.layer{37}.weights{2} = 0.005*randn(2,1,'single');      %1119��weight��0.001���ﵽ0.2����Ϊ0.0005
%         net.layer{37}.opts = {};
%         net.layer{37}.dilate = 1;
%         net.layer{37}.size=[1,1,4096,2];              %%%---%%%
%         net.layer{37}.precious=false;
%         net.layer{38}.precious=false;
%         net.layer{38} = struct('type', 'softmaxloss') ;
%         net.layer{38}.name = 'prob';
%         net.layer{38}.weights={};
%         
%        clear net.layers
%         net.layers=cell(1,38);
%         net.layers=net.layer;
%          clear net.layer
%         clear net.layer
%         net = vl_simplenn_tidy(net) ;
   
        
        net.meta.inputSize = [224,224,3,32] ;
            % net.meta.normalization.cropSize = 1;
            net.meta.normalization.cropSize = 1;
             net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ;
            % load(fullfile('D:\W_cancer\cancer\data\BreakHis_17_v2-vgg-f-fortwo-bnorm-simplenn','imageStats.mat'));  
            load(fullfile('/home/cad/zyc/chengxu/matconvnet-1.0-beta22bcn/examples/imagenet/data/im_aug-imagenet-vgg-verydeep-16-bnorm-simplenn','imageStats.mat'));%%%-======--%%%
            net.meta.normalization.averageImage =averageImage ;
            net.meta.classes.name ={'benign','malignant'} ;
            net.meta.classes.description = {'benign';'malignant'};
            % net.meta.augmentation.jitterLocation = true ;
            % net.meta.augmentation.jitterFlip = true ;
            net.meta.augmentation.jitterLocation = true ;                    %%%--opt--%%%
            net.meta.augmentation.jitterFlip = true ;
            % net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
            net.meta.augmentation.jitterBrightness = double(0.1 * rgbCovariance) ;
            net.meta.augmentation.jitterAspect = [2/3, 3/2] ;

                                    %%%%%%%%       

            net.meta.trainOpts.batchSize=2;                     %%%-------learningRate-----
            %net.meta.trainOpts.numSubBatches =2;
            net.meta.trainOpts.learningRate =[0.00007*ones(1,15),0.00006*ones(1,15),0.00005*ones(1,15),0.00004*ones(1,15),0.00001*ones(1,10),logspace(-7,-9,20)];%[0.00005*ones(1,20),0.00005*ones(1,10),0.000005*ones(1,10)];%logspace(-3,-6,100); %logspace(-4,-6,80)0.8;%[0.001 * ones(1,20), 0.001*ones(1,5), 0.0005*ones(1,10)] ;

            net.meta.trainOpts.weightDecay = 0.0003 ;
            net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;


    case 'resnet-50'
        net = cnn_imagenet_init_resnet('averageImage', rgbMean, ...
            'colorDeviation', rgbDeviation, ...
            'classNames', imdb.classes.name, ...
            'classDescriptions', imdb.classes.description) ;
        opts.networkType = 'dagnn' ;
        
    case 'vgg-f-self'                                      %%%---
    
        %     net =load('imagenet-vgg-f-fortwo.mat');        
                         %%%%%%%%
             net=load('vgg-f-self.mat');                %%%=====
             net.layers=net.layers(1:19);
             % add layer
             net.layers{20}.pad = [0,0,0,0];
             net.layers{20}.stride = [1,1];
             net.layers{20}.type = 'conv';
             net.layers{20}.name = 'fc8';
             net.layers{20}.weights{1} = 0.01*randn(1,1,4096,2,'single');
             net.layers{20}.weights{2} = 0.01*randn(2,1,'single');
             net.layers{20}.opts = {};
             net.layers{20}.dilate = 1;

             net.layers{20}.size=[1,1,4096,2];              %%%---%%%
             net.layers{20}.precious=false;

             net.layers{21}.precious=false;
             net.layers{21} = struct('type', 'softmaxloss') ;
             net.layers{21}.name = 'prob';
             net = vl_simplenn_tidy(net) ;

            net.meta.inputSize = [224,224,3, 32] ;
            % net.meta.normalization.cropSize = 1;
            net.meta.normalization.cropSize = 1;

            % load(fullfile('D:\W_cancer\cancer\data\BreakHis_17_v2-vgg-f-fortwo-bnorm-simplenn','imageStats.mat'));  
            load(fullfile('','imageStats.mat'));%%%-======--%%%
            net.meta.normalization.averageImage =averageImage ;
            net.meta.classes.name ={'benign','malignant'} ;
            net.meta.classes.description = {'benign';'malignant'};



            % net.meta.augmentation.jitterLocation = true ;
            % net.meta.augmentation.jitterFlip = true ;
            net.meta.augmentation.jitterLocation = false ;                    %%%--opt--%%%
            net.meta.augmentation.jitterFlip = false ;
            % net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;

            net.meta.augmentation.jitterBrightness = double(0.1 * rgbCovariance) ;
            net.meta.augmentation.jitterAspect = [2/3, 3/2] ;

                                    %%%%%%%%       

            net.meta.trainOpts.batchSize=128;                     %%%-------learningRate-----
            net.meta.trainOpts.learningRate = [0.0005 * ones(1,20), 0.001*ones(1,5), 0.0005*ones(1,10)] ;

            net.meta.trainOpts.weightDecay = 0.0003 ;
            net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
                          %%%%%%%%
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

