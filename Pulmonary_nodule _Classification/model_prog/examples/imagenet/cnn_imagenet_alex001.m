

function [net, info] = cnn_imagenet_alex001(varargin)
%CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%  This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%  VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.

% run(fullfile(fileparts(mfilename('D:\matconvnet-1.0-beta24')), ...
%   '..', '..', 'matlab', 'vl_setupnn.m')) ;
 


   % run(fullfile('D:\matconvnet-1.0-beta24', ...
%         'matlab', 'vl_setupnn.m')) ;

opts.dataDir ='/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/交叉验证/4/all227/';% fullfile('D:\W_cancer\cancer', '','BreakHis_17_v3') ; 
% opts.modelType = 'imagenet-vgg-f-fortwo' ;                           %%%---%%%

% opts.modelType = 'vgg-f-fortwo' ;     
opts.modelType = 'imagenet-matconvnet-alex' ; %%%===
opts.network = [] ;
opts.networkType = 'dagnn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;                    


[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
opts.expDir =fullfile('/home/cad/zyc/chengxu/matconvnet-1.0-beta22bcn/examples/imagenet/', 'data', ['alexim_aug-' sfx],'4-rgb42') ;     %%%---
% fullfile('D:\W_cancer\cancer\imagenet_res', 'data', ['im_cut-' sfx]) ;     %%%---
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile('/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/交叉验证/4/rgb42imdb4.mat') %fullfile('/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/56/handle_data/handled/fusion_56_1231.mat');%fullfile('/home/cad/zyc/shuju/1201/precdata/aug1202/','imdbaug1202.mat')
%fullfile(opts.expDir, 'imdb.mat');              %opt 
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1]; end;

                                           %%%%%%%%%%%%%%%%%%%%%%%%%%

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
  if isfield(imdb,'imdb')
      imdb=imdb.imdb;
  end
  imdb.imageDir = fullfile(opts.dataDir, '');                          %%%---
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
if isfield(imdb.images,'labels')
    imdb.images.label= imdb.images.labels;
end
% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
 
  train = find(imdb.images.set == 1) ;
  images = fullfile(imdb.imageDir, imdb.images.name(train(1:100:end))) ;
  [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
                                                    'imageSize', [227 227], ...
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
    case 'imagenet-matconvnet-alex'
%     net = cnn_imagenet_init_resnet('averageImage', rgbMean, ...
%                                      'colorDeviation', rgbDeviation, ...
%                                      'classNames', imdb.classes.name, ...
%                                      'classDescriptions', imdb.classes.description) ;

      net=dagnn.DagNN.loadobj( load('imagenet-matconvnet-alex.mat'));
      opts.networkType = 'dagnn' ;
      net.layers= net.layers(1:17);   %%% ---delete  fc8---
%       net.layers(1).inputs='data';    %%%---input---
%       net.meta.augmentation.rgbVariance=[];
      net.meta.augmentation.jitterLocation = true ;                    %%%--opt--%%%
      net.meta.augmentation.jitterFlip =  true  ;
      net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ;
% net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
      net.meta.augmentation.jitterAspect = [2/3, 3/2] ;
      net.meta.augmentation.jitterBrightness = double(0.1 * rgbCovariance) ;
%       clear net.meta.augmentation.rgbVariance;
      %%%===   average ===%%%
%       load(fullfile('D:\W_cancer\cancer\data\im_cut-vgg-f-bnorm-simplenn','imageStats.mat'));      
      net.meta.normalization.averageImage =averageImage ;
       %%%-------learningRate-----%%%
      net.meta.trainOpts.batchSize=64;                    
      net.meta.trainOpts.learningRate =[logspace(-4.5,-7,200),logspace(-7,-8,50)];% [0.0001 * ones(1,20), 0.0001*ones(1,20), 0.0001*ones(1,10)] ;
      net.meta.trainOpts.weightDecay = 0.0005 ;
      net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
      %%%---net  params---%%%
      net.addLayer('fc8',...
          dagnn.Conv('size',[1,1,4096,2]),...
          'x24',...
          'prediction',...
          {'fc8f','fc8b'});   
      net.params(15).value = 0.001*randn(1,1,4096,2,'single');         % rand  init
      net.params(16).value = 0.001*randn(2,1,'single');
%       net.layers(1).inputs='input'; 
      net.meta.normalization.cropSize = 1;
      net.addLayer('loss', ...
             dagnn.Loss('loss', 'softmaxlog') ,...
             {'prediction', 'label'}, ...
             'objective') ;
      net.addLayer('top1error', ...
             dagnn.Loss('loss', 'classerror'), ...
             {'prediction', 'label'}, ...
             'top1error') ;        
%       net.layers(1).inputs='input';

        v=net.getVarIndex('data');
        if isnan(v)
            net.renameVar('input','data');

        end

    
     
      net.meta.classes.name ={'benign','malignant'} ;
      net.meta.classes.description = {'benign','malignant'};
     
% load(fullfile('D:\W_cancer\cancer\data\BreakHis_17_v2-vgg-f-fortwo-bnorm-simplenn','imageStats.mat'));  
%load(fullfile('D:\W_cancer\cancer\imagenet_res\data\im_cut-imagenet-matconvnet-alex-bnorm-simplenn','imageStats.mat'));%%%-======--%%%
    load(fullfile('/home/cad/zyc/chengxu/matconvnet-1.0-beta22bcn/examples/imagenet/data/alexim_aug-imagenet-matconvnet-alex-bnorm-dagnn/4-rgb42','imageStats.mat'));%%%-======--%%%

    net.meta.normalization.averageImage =averageImage ;
    net.meta.classes.name ={'benign','malignant'} ;
    net.meta.classes.description = {'benign';'malignant'};
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
data = getImageBatch1(images, opts.(phase), 'prefetch', nargout == 0) ;
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

