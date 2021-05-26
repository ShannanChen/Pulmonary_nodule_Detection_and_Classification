# 基于残差网络的肺结节良恶性分类

### 1、下载**MatConvNet**（它是一个实现卷积神经网络的MATLAB工具箱）. 
下载链接：https://github.com/vlfeat/matconvnet，环境配置请参考官方文档

### 2、数据来源：

https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

（2）数据预处理
./data_pro

（3） 网络结构改变
方法：根据数据的类别修改全连接层。
例如：resnet50

    %%%---net  params---%%%
      net.addLayer('fc1000',...
          dagnn.Conv('size',[1,1,2048,2]),...
          'pool5',...
          'fc1000',...
          {'fc1000_filter','fc1000_bias'});
      net.params(214).value = 0.001*randn(1,1,2048,2,'single');     %
      net.params(215).value = 0.001*randn(2,1,'single');
      
具体代码：
resnet50 和vgg-f的代码为：  ./model_prog/examples/imagenet/cnn_imagenet_res001.m   
alexnet：./model_prog/examples/imagenet/cnn_imagenet_alex001.m
cifarnet： ./model_prog/examples/cifar/cnn_cifar.m 

（4）测试代码：
./model_prog/examples/imagenet/cnn_cub200_test.m




