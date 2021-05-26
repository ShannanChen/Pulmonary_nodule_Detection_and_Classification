function cnn_plate_test_many_dag()
   useGpu = 1;
    
%     dataDir ='/home/cad/zyc/shuju/所有处理的数据形式1208/roundlinshi/augxyzroundmin/' ;   

    % load imdb
      %  imdb =  load('/home/cad/zyc/shuju/所有处理的数据形式1208/roundlinshi/五倍交叉验证/sx/1/sxxyzminroud1212aug1.mat');
    imdb =  load(fullfile('/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/交叉验证/1/rgb42imdb1.mat'));  
    if isfield(imdb, 'imdb')
        imdb = imdb.imdb;
    end
  
    % load the net
    %modelPath = fullfile(vl_rootnn, 'examples', 'nodule_classfication', 'data', '1126im_aug-resnet-50-bnorm-dagnn', 'net-epoch-108.mat');
    modelPath = fullfile('/media/cad/18742566539/1120cifar/jieguoresnet0122/1-42/xin/','net-epoch-40.mat');
    tic;
    fprintf('loading the net...\n');
    net = load(modelPath);
    fprintf('%f\n', toc);
    if isfield(net, 'net')
        net = net.net;
    end
    net.mode = 'test';
    
    
    isDag = true;
    if isDag
        opts.networkType = 'dagnn';
        net = dagnn.DagNN.loadobj(net);
        if useGpu
            net.move('gpu');
        end
        % Drop existing loss layers
        drop = arrayfun(@(x) isa(x.block, 'dagnn.Loss'), net.layers);
        for n = {net.layers(drop).name}
            net.removeLayer(n);
        end
        % Extract raw predictions from softmax
        sftmx = arrayfun(@(x) isa(x.block, 'dagnn.SoftMax'), net.layers);
        predVar = 'score';
        
        for n = {net.layers(sftmx).name}
            l = net.getLayerIndex(n);
            v = net.getVarIndex(net.layers(l).outputs{1});
            if net.vars(v).fanout == 0
                predVar = net.layers(l).inputs{1};
                net.removeLayer(n);
            end
        end
%         net.removeLayer('fc63');
        % add softmax
        net.addLayer('score', dagnn.SoftMax(), 'fc1000', 'score', {});
        % 
        v = net.getVarIndex('data');
        if ~isnan(v)
            net.renameVar('data', 'input');
        end
    else
        net = vl_simplenn_tidy(net);
        net = vl_simplenn_move(net, 'gpu');
    end
    
    % prepare the test data
    test_index = find(imdb.images.set ==2);
    test_label = imdb.images.labels(test_index);
    test_img_path = imdb.images.name(test_index);
    
    
    % shuffle the data
%     test_index = test_index(randperm(length(test_index)));
%     test_label = imdb.images.labels(test_index);
%     test_img_path = imdb.images.name(test_index);
    
    test_num = length(test_index);
    
%     scores = zeros(1, 1, 63, test_num);
    
    pre = gpuArray(zeros(1, numel(test_label)));%%%%%%%%%%%%%%%%%%%%用GPU时删掉
    %pre = zeros(1, numel(test_label));
    
    
    
 fid = fopen('/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/txt/resnettxt/X1-rgb42.txt','wb');%%%存储图片名，真实标签、预测标签
    for j = 1 : test_num
        fprintf('%d/%d\n', j, test_num);
        impath=fullfile('/home/cad/zyc/shuju/1219/output/fusion_net_data_1230/44/rgb/42/交叉验证/1/all224/',test_img_path(j));%'/home/cad/zyc/shuju/所有处理的数据形式1208/roundlinshi/augxyzroundmin/'
        im = imread(impath{1}); 
        im = single(imresize(im , net.meta.normalization.imageSize(1:2)));
        im = bsxfun(@minus, im, net.meta.normalization.averageImage) ;
%         im = repmat(im, 1, 1, 3);
        if isDag
            inputs ={'input',gpuArray(im)};%%%%%%%%%%%%%%%%%%%%%%%%%%%直接计算结果可以删掉%号
            %inputs = {'input', im};
            net.eval(inputs);
            %scores(:, :, :, j) = gather(net.vars(net.getVarIndex('fc1000')).value);
            %%%%%%%%%%%%%%%
             %[~, best]= max(net.vars(175).value);         %%%%%%得到分类结果
%              pre(j) = best;
            % fprintf(fid,'%s\t%d\t%d\n',test_img_path{j},test_label(j),best);
            %%%%%%%%%%%%%%%%
%             l = net.getLayerIndex('score');
            scores = gather(net.vars(176).value);
            
            [~, best]= max(scores);         %%%%%%得到分类结果
              pre(j) = best;
            %  fprintf('%d\t%s\t\n'  ,pre(j),num2str(scores(1)) );
              %fprintf(fid,'%s\t%d\t%d\n',test_img_path{j},test_label(j),pre(j) );
              
%             fprintf('%d\t%d\t',scores(1),scores(2));
          fprintf(fid,'%s\t%d\t%d\t%s\t%s\t\n',test_img_path{j},test_label(j),pre(j),num2str(scores(1)),num2str(scores(2)));
              
        end
    end 
  fclose(fid);
%     scores = reshape(scores, [], test_num);
%     pre = zeros(1, test_num);
%     for k = 1 : test_num
%         [~, pre(k)] = max(scores(:,k));  % obtain the index of max
%     end
    
%     true_num = length(find(result == test_label));
%     accuracy = true_num / test_num;
%     disp(accuracy);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
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
    
    for k = 1 : numel(pre_false_index)
        if ismember(pre_false_index(k) , test_label_false_index)
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
    fprintf('teyidu:%d%\n', (fp./(length(test_label_false_index)))*100);
    fprintf('senstive:%d\n',( tp./(length(test_label_true_index)))*100);

    
end

