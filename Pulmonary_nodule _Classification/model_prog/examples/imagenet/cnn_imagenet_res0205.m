function [net, info] = cnn_imagenet_res0205(varargin)
opts.dataDir = '/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/交叉验证/2/all224//';
opts.modelType = 'resnet-50' ; %%%===
opts.network = [] ;
opts.networkType = 'dagnn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
opts.expDir = fullfile('/home/cad/zyc/chengxu/matconvnet-1.0-beta22bcn/examples/imagenet/data/bp0131im_aug-resnet-50-bnorm-dagnn/2/res3d_branch2a') ;     %%%---
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath ='/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/交叉验证/2/rgb42imdb2.mat';%'/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/handle_data/handled/fusion_44_1231.mat' %fullfile('/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/grayhandle42/gray42-105.mat') %fullfile('/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/56/handle_data/handled/fusion_56_1231.mat');%fullfile('/home/cad/zyc/shuju/1201/precdata/aug1202/','imdbaug1202.mat')
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [2]; end;

                                           %%%%%%%%%%%%%%%%%%%%%%%%%%

% -------------------------------------------------------------------------
%                                                              Prepare data
% ------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
  imdb=imdb.imdb;
  imdb.images.label=  imdb.images.labels;
  imdb.imageDir = fullfile(opts.dataDir, '');                          %%%---
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Compute image statistics (mean, RGB covariances, etc.)
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
%     net = cnn_imagenet_init_resnet('averageImage', rgbMean, ...
%                                      'colorDeviation', rgbDeviation, ...
%                                      'classNames', imdb.classes.name, ...
%                                      'classDescriptions', imdb.classes.description) ;
%       aa=load('/home/cad/zyc/chengxu/matconvnet-1.0-beta22bcn/examples/imagenet/data/bp0131im_aug-resnet-50-bnorm-dagnn/1/1-rgb-quan/0.0000160net.mat');
%       aa=aa.net;
%       net=dagnn.DagNN.loadobj( aa);
      net=dagnn.DagNN.loadobj('imagenet-resnet-50-dag.mat');
%       opts.networkType = 'dagnn' ;   
      net.layers= net.layers(1:173);   %%% ---delete  softmax---
      net.addLayer('fc1000',...
          dagnn.Conv('size',[1,1,2048,2]),...
          'pool5',...
          'fc1000',...
          {'fc1000_filter','fc1000_bias'});
      net.params(214).value = 0.001*randn(1,1,2048,2,'single');     %%%%%%改动1119   源0.001改为0.005效果平稳
      net.params(215).value = 0.001*randn(2,1,'single');
      net.addLayer('loss', ...
             dagnn.Loss('loss', 'softmaxlog') ,...
             {'fc1000', 'label'}, ...
             'objective') ;
      net.addLayer('top1error', ...
             dagnn.Loss('loss', 'classerror'), ...
             {'fc1000', 'label'}, ...
             'top1error') ;   
         
      load(fullfile('/home/cad/zyc/chengxu/matconvnet-1.0-beta22bcn/examples/imagenet/data/bp0131im_aug-resnet-50-bnorm-dagnn/2/res3d_branch2a','imageStats.mat'));%%%-======--%%%
      %net.meta.normalization.cropSize = 1;   %20180205zyc    以前是这个
      net.meta.augmentation.jitterLocation = true ;                    %%%--opt--%%%
      net.meta.augmentation.jitterFlip = false ;
      net.meta.augmentation.jitterAspect = [2/3, 3/2] ;
      net.meta.augmentation.jitterBrightness = double(0.1 * rgbCovariance) ;  
      net.meta.normalization.averageImage =averageImage ;
      %%%%%%%20180205zyc    新加的%%%%%%%%%%
      net.meta.normalization.imageSize = [224 224 3] ;
       net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
       net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ;
       %net.meta.augmentation.jitterScale  = [0.4, 1.1] ;
       %%%%%%%%%%%%%向上%%%%%%%%%%%%%%%
       %%%-------learningRate-----%%%
      net.meta.trainOpts.batchSize=16;                    
      net.meta.trainOpts.learningRate =0.000005*ones(1,185);%[logspace(-4.5,-7,60)];%logspace(-4.5,-6,60);%0.000005*ones(1,100);%[0.000005*ones(1,40),0.000001*ones(1,10),]%,0.000001*ones(1,20),0.0000005*ones(1,50)];%0.000005*ones(1,220);%logspace(-5,-7,200);%[0.000009 * ones(1,25),0.000007 * ones(1,45),0.000006 * ones(1,30),0.000005 * ones(1,20)];%在全部中的权重%[logspace(-4,-7,50)];%[0.01 * ones(1,10), 0.005*ones(1,20), 0.0005*ones(1,10),0.0005*ones(1,10),0.0001*ones(1,10)] ;%[logspace(-3,-6,100)]; 
      net.meta.trainOpts.weightDecay =0.0005;%0.0005 ;
      net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
      net.meta.classes.name ={'benign','malignant'} ;
      net.meta.classes.description = {'benign','malignant'};
     
      net.meta.normalization.averageImage =averageImage ;
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
  case 'simplenn', trainFn = @cnn_train_cancer ;                          %%%%%%%%%%%%%%%%%
  case 'dagnn', trainFn = @cnn_train_dag ;
end

[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

% net = cnn_imagenet_deploy(net) ;
% modelPath = fullfile(opts.expDir, 'net-deployed.mat');
% 
% switch opts.networkType
%   case 'simplenn'
%     save(modelPath, '-struct', 'net') ;
%   case 'dagnn'
%     net_ = net.saveobj() ;
%     save(modelPath, '-struct', 'net_') ;
%     clear net_ ;
% end

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
      varargout{1} = {'data', data, 'label', labels} ;
%         varargout{1} = {'inputs', data, 'label', labels} ;
  end
end

