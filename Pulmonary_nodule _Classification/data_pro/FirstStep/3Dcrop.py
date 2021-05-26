#encoding:utf-8
#from step1 import step1_python
#from full_prep import *
from PIL import Image
import warnings
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology
from glob import *
import cv2
import math
from scipy.ndimage.interpolation import zoom
def load_scan(path):  #加载切片
    dcm=glob(path+"*dcm")
    slices = [dicom.read_file(s) for s in dcm]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))   #从小到大排序  从肺的底部到头部
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2;
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num + 1;
        slice_num = int(len(slices) / sec_num)
        slices.sort(key=lambda x: float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:  # 处理数据中的丢失值
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices
def get_pixels_hu(slices):  # 将像素值转化为HU值 返回图像和原始像素间的距离
    image = np.stack([s.pixel_array for s in slices])      #np.stack   合并变成了立体图# s.pixel_array 图片上的得到信息，表示像素将的物理距离
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)    image = image.astype(np.int16)
    image = image.astype(np.int16)
    plt.figure()
    plt.imshow(slices[2].pixel_array)
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16),np.array(slices[0].ImagePositionPatient[2],dtype=np.float32),np.array([slices[0].SliceThickness]+slices[0].PixelSpacing,
                                                     dtype=np.float32)  #将slices[0].SliceThickness与slices[0].PixelSpacing连起来eg【1.25,0.68,0.68】
##########################read new.txt#######################################
def xyz(labelfile):
    new_txt = open(labelfile, 'r')  # , encoding='utf-8')
    x=[]
    y=[]
    count=0
    count1=0
    count2=0
    count3=0
    one=0
    for line_data0 in new_txt.readlines():
        # 按空格将数据转化为list
        list_line_data = line_data0.split(' ')
        x.append(list_line_data[2])
        y.append(list_line_data[3])
    new_txt.close()

    ###########################选取切片的条件坐标的提取
    for i in range(len(x) - 1):
        if (x[i] == x[i + 1] and y[i] == y[i + 1]) :# or (float(math.fabs(float(x[i])-float(x[i+1])))<32.0 and float(math.fabs(float(y[i])-float(y[i+1])))<32.0):
            if i==count:
                count = count+1
            elif x[i]==x[i+1] and y[i+1]==y[i]:
                if i==count+count1+1:
                    count1 = count1 + 1
                elif x[i+1] == x[i] and y[i+1] == y[i ]:
                    if i==count+count1+count2+2:
                        count2 = count2 + 1
                    elif x[i] == x[i+1] and y[i] == y[i+1]:
                        count3 = count3 + 1
    xyzl=[]
    new_txt1 = open(labelfile, 'r')  #第二次打开
    for line_data1 in new_txt1.readlines():
        list_line_data = line_data1.split(' ')
        one=one+1
        if one==math.ceil((count)/2 +1)or one==math.ceil((count1)/2+count+2 ) or one==math.ceil((count2)/2+count1+count+3 ) or one==math.floor((count3)/2+count1+count+count2+4):
            xyzl.append([list_line_data[1],list_line_data[2],list_line_data[3],list_line_data[4],list_line_data[0]])
    new_txt1.close()
    return xyzl
###############################################################################
def resample(imgs, spacing, new_spacing,xyzl,origin,order = 2):     #重采样
    if len(imgs.shape)==3:
        newxyzl=[]
        print(int(math.fabs(((float(xyzl[0][0])-origin)))))
        print (spacing[0])
        new_shape = np.round(imgs.shape *spacing/new_spacing)  #new_shape为实际上取整后转化后图像的大小
        for i in range(len(xyzl)):
            print (float(xyzl[i][1]))
            x=np.round(float(xyzl[i][1])*spacing[1]/new_spacing[1])
            y=np.round(float(xyzl[i][2])*spacing[2]/new_spacing[2])
            z=math.fabs(((float(xyzl[i][0])-origin))*new_spacing[0])
            l=float(xyzl[i][3])
            uid=xyzl[i][4]
            newxyzl.append([z,x,y,l,uid])
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape                  #实际的像素间距离
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)   #线性插值eg 从原来的261,512，512变为了326,350,350
            print(int(newxyzl[i][0]),int(newxyzl[i][1]),int(newxyzl[i][2]))
            s = np.array(imgs[int(newxyzl[i][0]), :, :])
            plt.figure()
            plt.imshow((imgs[int(newxyzl[i][0]), :, :]))
            y=np.array(imgs[:,int(newxyzl[i][2]),:])
            plt.figure()
            plt.imshow(y)
            x=np.array(imgs[:,:,int(newxyzl[i][1])])  #int(new_shape[2])
            print(s.shape,x.shape,y.shape)
            plt.figure()
            plt.imshow(x)
        return imgs, true_spacing,newxyzl
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
############################################################################
#######################################crop
def extract_real_cubic_from_dcm(imgs,newxyzl,normalization_output_path,LIDC):


    #img_arrays = imgs.transpose(0, 1, 2)
    img_arrays=imgs    #imgs是zyx
    a=img_arrays.shape
    print(img_arrays.shape,a[0],a[1])
    for i in range(len(newxyzl)):
        label=int(newxyzl[i][3])
        uid=newxyzl[i][4]
        v_center = np.array([int(newxyzl[i][0]), int(newxyzl[i][1]), int(newxyzl[i][2])])  # nodule center


        ##################################切3D
        window_size=22
        zyx_1 = v_center - window_size  # 注意是: Z, Y, X
        zyx_2 = v_center + window_size + 1
        nodule_box = np.zeros([45, 45, 45], np.int16)  # ---nodule_box_size = 45
        img_crop = imgs [zyx_1[0]:zyx_2[0], zyx_1[2]:zyx_2[2], zyx_1[1]:zyx_2[1]]  # ---截取立方体
        img_crop = set_window_width(img_crop)  # ---设置窗宽，小于-1000的体素值设置为-1000
        print (img_crop.shape)
        zeros_fill = math.floor((45 - (2 * window_size+1 )) / 2) #window_size+1
        try:
            nodule_box[zeros_fill:45 - zeros_fill, zeros_fill:45 - zeros_fill,
            zeros_fill:45 - zeros_fill] = img_crop  # ---将截取的立方体置于nodule_box
        except:
            continue
        nodule_box[nodule_box == 0] = -1000  # ---将填充的0设置为-1000，可能有极少数的体素由0=>-1000，不过可以忽略不计
        img3=normalazation(nodule_box)
        if label <3:
            labels=1
        elif label>3:
            labels=2
        else:
            continue

        # plot_nodule(img3)
    ########################################################aug

        save_3d_nodule(normalization_output_path, img3, uid, labels, i,LIDC)
        angle_transpose(img3, 90, "_leftright",labels,normalization_output_path,uid,i,LIDC)
        angle_transpose(img3, 180, "_updown",labels,normalization_output_path,uid,i,LIDC)
        angle_transpose(img3, 270, "_diagonal",labels,normalization_output_path,uid,i,LIDC)
        print("Final step done...")



            ########################################


        # save_3d_nodulel(normalization_output_path, imgl, uid, labels, i)
        # save_3d_noduleu(normalization_output_path, imgu, uid, labels, i)
        # save_3d_noduled(normalization_output_path, imgd, uid, labels, i)




######################################################################################
##############################aug


def angle_transpose(file,degree,flag_string,labels,normalization_output_path,uid,i,LIDC):
    '''
     @param file : a npy file which store all information of one cubic
     @param degree: how many degree will the image be transposed,90,180,270 are OK
     @flag_string:  which tag will be added to the filename after transposed
    '''
    array = file
    #array = array.transpose(2, 1, 0)  # from x,y,z to z,y,x
    newarr = np.zeros(array.shape,dtype=np.float32)
    for depth in range(array.shape[0]):
        jpg = array[depth]
        jpg.reshape((jpg.shape[0],jpg.shape[1],1))
        img = Image.fromarray(jpg)
        #img.show()
        out = img.rotate(degree)  # 逆时针旋转90度
        newarr[depth,:,:] = np.array(out).reshape(array.shape[1],-1)[:,:]
    #newarr = newarr.transpose(2,1,0)
    #np.save(file.replace(".npy",flag_string+".npy"),newarr)
    print(labels,str(LIDC), str(uid), i,str(degree))
    #plot_nodule(newarr)
    normalization_output_paths = os.path.join(normalization_output_path, str(LIDC))
    if os.path.isdir(normalization_output_paths):
        print('have')
    else:
        os.mkdir(normalization_output_paths)
    np.save(os.path.join(normalization_output_paths, "%s_%s_%s_%d-%s.npy" % (labels,str(LIDC), str(uid), i,str(degree))), newarr)


############################################################



def plot_nodule(nodule_crop):
    # Learned from ArnavJain
    # https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    f, plots = plt.subplots(int(nodule_crop.shape[0] / 4) + 1, 4, figsize=(10, 10))

    for z_ in range(nodule_crop.shape[0]):
        plots[int(z_ / 4), z_ % 4].imshow(nodule_crop[z_, :, :])

    # The last subplot has no image because there are only 19 images.
    plt.show()





# ################################
def set_window_width(image, MIN_BOUND=-1000.0):
    image[image < MIN_BOUND] = MIN_BOUND
    return image
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array =(image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return np.array(image_array ,dtype='float32')
def save_3d_nodule(normalization_output_path,nodule_box, uid,labels,i,LIDC):
    normalization_output_paths=os.path.join(normalization_output_path,str(LIDC))
    if os.path.isdir(normalization_output_paths):
        print('have')
    else:
        os.mkdir(normalization_output_paths)
    np.save(os.path.join(normalization_output_paths, "%s_%s_%s_%d-annotations.npy" % (labels,LIDC,uid,i)), nodule_box)


# def save_3d_nodulel(normalization_output_path, nodule_box, uid, labels, i):
#     np.save(os.path.join(normalization_output_path, "%s_%s_%d-leftright.npy" % (labels, uid, i)), nodule_box)
#
# def save_3d_noduleu(normalization_output_path,nodule_box, uid,labels,i):
#     np.save(os.path.join(normalization_output_path, "%s_%s_%d-updown.npy" % (labels,uid,i)), nodule_box)
#
# def save_3d_noduled(normalization_output_path,nodule_box, uid,labels,i):
#     np.save(os.path.join(normalization_output_path, "%s_%s_%d-adiagonal.npy" % (labels,uid,i)), nodule_box)

###############################################################

if __name__ == '__main__':
    new_spacing=np.array([1,1,1])
    output='/home/cad/zyc/shuju/hand1208onlyminround3d/augrot'
    if os.path.isdir(output):
        pass
    else:
        os.mkdir(output)
    #name=['00','02','03','04']
    #for nna in name:
    INPUT_FOLDER =  "/media/cad/18742566539/LIDC lung cancer database/"   #os.path.join('/home/cad/zyc/shuju/1201/',nna)   #'/media/cad/18742566539/1234/fenkai/02/';
        # INPUT_FOLDER='/home/zyc/chengxu/LIDCToolbox-master/try1/try/'
        # output = '/home/zyc/chengxu/LIDCToolbox-master/try1/receive/'
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    for LIDC in patients:
            siroot2 = os.path.join(INPUT_FOLDER,LIDC)
            print(siroot2)
            fname2 = os.listdir(siroot2)
            for fn2 in fname2:
                stroot3 = siroot2 + "/" + fn2
                fname3 = os.listdir(stroot3)
                for fn3 in fname3:
                    root4 = stroot3 + "/" + fn3 + "/"
                    print(root4)
                    #path = os.path.join(root4, 'txt')
                    path_dcm = glob(root4 + "*.dcm")
                    print(path_dcm)
                    if len(path_dcm) >= 10: #and os.path.isdir(path) and len(os.listdir(path)) > 0:#len(xmlpath) == 0:
                        case = load_scan(root4)
                        case_pixels, origin,spacing = get_pixels_hu(case)

                        txt_paths=os.path.join(root4,'txt')
                        txt_names = os.listdir(txt_paths)
                        if 'new.txt' in txt_names:

                            labelfile=os.path.join(root4,'txt','new.txt')

                            print (len(os.listdir(os.path.join(root4,'txt'))))
                            if len(os.listdir(os.path.join(root4,'txt')))>2:
                                xyzl=xyz(labelfile)
                            else:
                                break
                            imgs, true_spacing, newxyzl=resample(case_pixels, spacing, new_spacing,xyzl,origin, order=2)
                            extract_real_cubic_from_dcm(imgs,newxyzl,output,LIDC)

                    else:
                        break
