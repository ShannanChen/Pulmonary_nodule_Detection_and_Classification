function [net, info] = cnn_imagenet_res001(varargin)
%CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%  This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%  VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.

% run(fullfile(fileparts(mfilename('D:\matconvnet-1.0-beta24')), ...
%   '..', '..', 'matlab', 'vl_setupnn.m')) ;
 


   % run(fullfile('D:\matconvnet-1.0-beta24', ...
%         'matlab', 'vl_setupnn.m')) ;

opts.dataDir ='/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/交叉验证/4/all224/';%'/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/28/handle_data/handled/all/';%'/home/cad/zyc/shuju/1219/output/zrgb/2倍交叉验证/1/alltrainjpg/all/';(64)1231%'/home/cad/zyc/shuju/所有处理的数据形式1208/roundlinshi/augxyzroundmin';%'/home/cad/zyc/shuju/1201/precdata/aug1202/all/';%'/home/cad/zyc/20171018handledata/last1116/1125/all/';
% opts.modelType = 'imagenet-vgg-f-fortwo' ;                           %%%---%%%

% opts.modelType = 'vgg-f-fortwo' ;     
opts.modelType = 'resnet-50' ; %%%===
opts.network = [] ;
opts.networkType = 'dagnn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;


[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
opts.expDir = fullfile('/home/cad/zyc/chengxu/matconvnet-1.0-beta22bcn/examples/imagenet/', 'data', ['weightdecayim_aug-' sfx],'0204rgb42-40.00005/') ;     %%%---
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
%opts.imdbPath =fullfile('/home/cad/zyc/20171018handledata/last1116/all1126/','1126123duo.mat');% '/home/cad/zyc/20171018handledata/last1116/1125/1125pepole123.mat';%fullfile(opts.expDir, 'wuchongfu.mat');
opts.imdbPath = fullfile('/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/交叉验证/4/rgb42imdb4.mat');%fullfile('/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/28/handle_data/handled/fusion_28_1231.mat')% fullfile('/home/cad/zyc/shuju/1219/output/zrgb/2倍交叉验证/1/alltrainjpg/1228imdb.mat');(64)1231%fullfile('/home/cad/zyc/shuju/1201/precdata/aug1202/','imdbaug1202.mat')
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

      net=dagnn.DagNN.loadobj( load('imagenet-resnet-50-dag.mat'));
      
      opts.networkType = 'dagnn' ;
      
      
      net.layers= net.layers(1:173);   %%% ---delete  softmax---
      
      
      net.meta.augmentation.jitterLocation =true ;                    %%%--opt--%%%
      net.meta.augmentation.jitterFlip = false ;
% net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;

      net.meta.augmentation.jitterAspect = [2/3, 3/2] ;
      net.meta.augmentation.jitterBrightness = double(0.1 * rgbCovariance) ;
      
      
      %%%===   average ===%%%
%       load(fullfile('D:\W_cancer\cancer\data\im_cut-vgg-f-bnorm-simplenn','imageStats.mat'));
      
      net.meta.normalization.averageImage =averageImage ;
      
      
       %%%-------learningRate-----%%%
      net.meta.trainOpts.batchSize=16;                    
      net.meta.trainOpts.learningRate =[logspace(-4.5,-5,20),logspace(-5,-7,40)]%20;%[0.000005 * ones(1,50),0.000002 * ones(1,50)]%[0.000007 * ones(1,45),0.000006 * ones(1,30),0.000005 * ones(1,20),0.000002 * ones(1,10),0.000001 * ones(1,10)];%在全部中的权重%[logspace(-4,-7,50)];%[0.01 * ones(1,10), 0.005*ones(1,20), 0.0005*ones(1,10),0.0005*ones(1,10),0.0001*ones(1,10)] ;%[logspace(-3,-6,100)]; 

      net.meta.trainOpts.weightDecay = 0.00005;%0.0005 ;
      net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
      
      
      %%%---net  params---%%%
      net.addLayer('fc1000',...
          dagnn.Conv('size',[1,1,2048,2]),...
          'pool5',...
          'fc1000',...
          {'fc1000_filter','fc1000_bias'});
      
      
      net.params(214).value = 0.001*randn(1,1,2048,2,'single');     %%%%%%改动1119   源0.001改为0.005效果平稳
      net.params(215).value = 0.001*randn(2,1,'single');
      
%       net.layers(1).inputs='input';
      
      net.meta.normalization.cropSize = 1;
      
      net.addLayer('loss', ...
             dagnn.Loss('loss', 'softmaxlog') ,...
             {'fc1000', 'label'}, ...
             'objective') ;
      net.addLayer('top1error', ...
             dagnn.Loss('loss', 'classerror'), ...
             {'fc1000', 'label'}, ...
             'top1error') ;   
         

%       net.layers(1).inputs='input';
     
      net.meta.classes.name ={'benign','malignant'} ;
      net.meta.classes.description = {'benign','malignant'};
     
      
      
      
    case 'vgg-m'                                      %%%---
    
%     net =load('imagenet-vgg-f-fortwo.mat');        
                         %%%%%%%%
                         
                         
 net=load('vgg-m.mat');                %%%=====
net.layers=net.layers(1:19);

% add layer
net.layers{20}.pad = [0,0,0,0];
net.layers{20}.stride = [1,1];
net.layers{20}.type = 'conv';
net.layers{20}.name = 'fc8';
net.layers{20}.weights{1} = 0.001*randn(1,1,4096,2,'single');
net.layers{20}.weights{2} = 0.001*randn(2,1,'single');
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
load(fullfile('/home/cad/zyc/chengxu/matconvnet-1.0-beta22bcn/examples/imagenet/data/weightdecayim_aug-resnet-50-bnorm-dagnn/0204rgb42-40.00005/','imageStats.mat'));%%%-======--%%%
net.meta.normalization.averageImage =averageImage ;
net.meta.classes.name ={'benign','malignant'} ;
net.meta.classes.description = {'benign';'malignant'};



% net.meta.augmentation.jitterLocation = true ;
% net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterLocation = false ;   %true                 %%%--opt--%%%
net.meta.augmentation.jitterFlip = false ; %true
% net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;

net.meta.augmentation.jitterBrightness = double(0.1 * rgbCovariance) ;
net.meta.augmentation.jitterAspect = [2/3, 3/2] ;
    
                        %%%%%%%%       
   
net.meta.trainOpts.batchSize=160;                     %%%-------learningRate-----
net.meta.trainOpts.learningRate = [0.001 * ones(1,20), 0.001*ones(1,5), 0.0005*ones(1,10)] ;

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

