% run matlab/vl_setupnn
useGpu = 1;
dataDir ='/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/交叉验证/2/all224/';%'/home/cad/zyc/shuju/所有处理的数据形式1208/roundlinshi/五倍交叉验证/sx/1/all/';%'/home/cad/zyc/shuju/所有处理的数据形式1208/linshi/all/'; %'/home/cad/zyc/shuju/1201/precdata/aug1202/all/';
% ����imdb
%imdb =  load(fullfile(dataDir, 'data', 'im_aug-vgg-f-bnorm-simplenn', 'traincomz22441.mat'));
%imdb =load('/home/cad/zyc/shuju/所有处理的数据形式1208/roundlinshi/五倍交叉验证/sx/1/sxxyzminroud1212aug1.mat');% 
imdb =load(fullfile('/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/交叉验证/2/rgb42imdb2.mat'));%('/home/cad/zyc/shuju/所有处理的数据形式1208/linshi/newhandquanwu.mat');
imdb = imdb.imdb;
imdb.images.label=imdb.images.labels;

%����modelType
%net = load(fullfile('/home/cad/zyc/matconvnet-1.0-beta22/examples/nodule_classfication/data/addlayer-vgg-f-bnorm-simplenn', 'net-epoch-50.mat'));
net = load(fullfile('/media/cad/18742566539/1120cifar/receivevgg/2-42rgb/', 'net-epoch-50.mat'));  %vd
%net = load('/home/cad/zyc/matconvnet-1.0-beta22/examples/nodule_classfication/data/41三张灰度尺度相等的/net-epoch-150.mat')
net = net.net;
%net=load('vgg-f')

net = vl_simplenn_tidy(net);
% ����ʱ������Ҫ��softmanloss(����BP) ֻ����һ�����򴫲��Ϳ�����
net.layers{1, end}.type = 'softmax';                             
if useGpu 
    net = vl_simplenn_move(net, 'gpu');
end
% ��ȡ���Լ���ͼƬ����
test_index = find(imdb.images.set == 2);  
% test_index = test_index(1 : 10);
test_label = imdb.images.label(test_index);
% ��Ӧ�Ĳ��Լ������·��
test_img_path = imdb.images.name(test_index);

 pre = zeros(1, numel(test_label));
% ��ȡ���Ե�ͼƬ��ݵ�test_data

fid = fopen('/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/txt/vggtxt/2-rgb42.txt','wb');
for i = 1 : numel(test_label)
    fprintf('%d/%d\n', i, numel(test_label));
   % impath = fullfile(dataDir, 'data', 'traincomz224', test_img_path(i));
   impath=fullfile('/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/交叉验证/2/all224/',test_img_path(i));
    im = imread(impath{1});
    % ��im��ת��Ϊ����������
    im_ = single(im) ; % note: 255 range
    
    % ��һ����С ��ͼƬ���ŵ�224 * 224 �Ĵ�С
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
     im_ =single(im_ - net.meta.normalization.averageImage) ;
    [~,~,d]=size(im_);
    if d ~= 3
        img(: ,:, 1)=im_;
        img(:, :, 2) = im_;
        img(:, :, 3) = im_;
        im_ = img;
    end
    if useGpu
        im_ = gpuArray(im_);
    end
  
    % res���˼������Լ��м�����������һ������������࣬��һ������  sofamaxloss��
    res=vl_simplenn(net, im_);

    % res(i +1 ).x ��i������ res(end).x ���һ������
    scores = squeeze(gather(res(end).x)) ;%  删除单一维
    % best �����ֵ������bestScore���Ӧ��ֵ
    [~, best] = max(scores) ;
    pre(i) = best;
%      fprintf(fid,'%s\t%d\t%d\n',test_img_path{i},test_label(i),best);
 fprintf(fid,'%s\t%d\t%d\t%s\t%s\t\n',test_img_path{i},test_label(i),pre(i),num2str(scores(1)),num2str(scores(2)));
end
   fclose(fid);
% accurcy = length(find(pre == test_label')) / length(test_label);
pre_true_index = find(pre == 1);
pre_false_index = find(pre == 2);

test_label_true_index = find(test_label == 1);
test_label_false_index = find(test_label == 2);
tp = 0;
tn = 0;
fp = 0;
fn = 0;
for i = 1 : numel(pre_true_index)
    if ismember(pre_true_index(i), test_label_true_index)
        tp = tp + 1;
    else
        tn = tn + 1;
    end
end

for i = 1 : numel(pre_false_index)
    if ismember(pre_false_index(i) , test_label_false_index)
        fp = fp + 1;
    else
        fn = fn + 1;
    end
end

fprintf('%d\n', tp);    %tp 良性结节中的计算正确的（TP）
fprintf('%d\n', fp);   %fp 恶性结节中计算正确的     （TN）
fprintf('%d\n', tn);   %tn 恶性结节中计算错误的     （FN）
fprintf('%d\n', fn);    %fn为良性结节中计算错误的  （FP）
fprintf('acc:%d%\n', ((tp+fp)./(tp+tn+fp+fn))*100);
fprintf('teyidu:%d%\n', ((fp./(fp+tn))*100));
fprintf('senstive:%d\n',( tp./(tp+fn))*100);




% c=zeros(size(B));
% for i=1:length(B)
% c(i)=sum(find(A==B(i)));
% end



