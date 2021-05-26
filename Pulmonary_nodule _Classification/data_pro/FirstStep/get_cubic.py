# -*- coding:utf-8 -*-
'''
 LUNA2016 data prepare
'''
#import SimpleITK as sitk
import dicom
import numpy as np
import csv
from glob import glob
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import traceback
import random
from PIL import Image
import math
import cv2
'''
########################重采样###################
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices
##############################################
'''
def extract_real_cubic_from_mhd(dcim_path,annatation_file,plot_output_path,normalization_output_path):
    print (dcim_path)
    file_list=glob(dcim_path+"*.dcm")

    df_node = pd.read_csv(annatation_file)
    #file_listg = dicom.read_file(file_list)
    print  (df_node)
    for i, file in df_node.iterrows():
        for img_file in file_list:
        #mini_df = df_node(df_node["third"]==img_file) #get all nodules associate with file
            file_name = str(img_file).split("/")[-1]
        #for i ,file  in df_node.iterrows():
            if  (str(file['third'])+str('.dcm'))==(file_name):  #mini_df.shape[0]>0: # some files may not have a nodule--skipping those
                dcm = dicom.read_file(img_file)
                img_array=dcm.pixel_array
                plt.figure()
                plt.imshow(img_array,cmap='gray')
            else :
                continue
            print("begin to process real nodules...")
            imgs1 = np.zeros([45, 45])
            print(img_array.shape)
            v_center=[file['x'],file['y']]
            mag=file['label']
            print (mag)
            print (v_center)
            window_size=22

            imgs = img_array[int(v_center[0] - window_size):int(v_center[0] + window_size + 1),
                       int(v_center[1] - window_size):int(v_center[1] + window_size + 1)]
            zeros_fill = int(math.floor((45 - (2 * window_size + 1)) / 2))
            try:
                        imgs1[zeros_fill:45 - zeros_fill,zeros_fill:45 - zeros_fill] = imgs  # ---将截取的立方体置于nodule_box
            except:
                    continue
            imgs1[imgs1 == 0] = -1000
            imgs1 = 255*normalazation(imgs1)
            cv2.imwrite(os.path.join(normalization_output_path, "%d_%s_real_size45.jpg" %(i,mag)),imgs1)


def check_nan(path):
    '''
     a function to check if there is nan value in current npy file path
    :param path:
    :return:
    '''
    for file in os.listdir(path):
        array = np.load(os.path.join(path,file))
        a = array[np.isnan(array)]
        if len(a)>0:
            print("file is nan :  ",file )
            print(a)

def plot_cubic(npy_file):
    '''
       plot the cubic slice by slice

    :param npy_file:
    :return:
    '''
    cubic_array = np.load(npy_file)
    f, plots = plt.subplots(int(cubic_array.shape[2]/3), 3, figsize=(50, 50))
    for i in range(1, cubic_array.shape[2]+1):
        plots[int(i / 3), int((i % 3) )].axis('off')
        plots[int(i / 3), int((i % 3) )].imshow(cubic_array[:,:,i], cmap=plt.cm.bone)

def plot_3d_cubic(image):
    '''
        plot the 3D cubic
    :param image:   image saved as npy file path
    :return:
    '''
    from skimage import measure, morphology
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    image = np.load(image)
    verts, faces = measure.marching_cubes(image,0)
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    plt.show()

# LUNA2016 data prepare ,first step: truncate HU to -1000 to 400
def truncate_hu(image_array):
    image_array[image_array > 400] = 0
    image_array[image_array <-1000] = 0

# LUNA2016 data prepare ,second step: normalzation the HU
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work

def search(path, word):
    '''
       find filename match keyword from path
    :param path:  path search from
    :param word:  keyword should be matched
    :return:
    '''
    filelist = []
    for filename in os.listdir(path):
        fp = os.path.join(path, filename)
        if os.path.isfile(fp) and word in filename:
            filelist.append(fp)
        elif os.path.isdir(fp):
            search(fp, word)
    return filelist


def get_all_filename(path,size):
    list_real = search(path, 'real_size' + str(size) + "x" + str(size))
    list_fake = search(path, 'fake_size' + str(size) + "x" + str(size))
    return list_real+list_fake


def angle_transpose(file,degree,flag_string):
    '''
     @param file : a npy file which store all information of one cubic
     @param degree: how many degree will the image be transposed,90,180,270 are OK
     @flag_string:  which tag will be added to the filename after transposed
    '''
    array = np.load(file)
    array = array.transpose(2, 1, 0)  # from x,y,z to z,y,x
    newarr = np.zeros(array.shape,dtype=np.float32)
    for depth in range(array.shape[0]):
        jpg = array[depth]
        jpg.reshape((jpg.shape[0],jpg.shape[1],1))
        img = Image.fromarray(jpg)
        #img.show()
        out = img.rotate(degree)  # 逆时针旋转90度
        newarr[depth,:,:] = np.array(out).reshape(array.shape[1],-1)[:,:]
    newarr = newarr.transpose(2,1,0)
    np.save(file.replace(".npy",flag_string+".npy"),newarr)


if __name__ =='__main__':



    annatation_file = '/home/cad/zyc/chengxu/shujuchuli/docpack/nodule1.csv'
    #candidate_file = '/home/cad/zyc/shuju/lunamhd/csv/candidates.csv'
    plot_output_path = '/home/cad/zyc/chengxu/shujuchuli/docpack/plot/'
    normalazation_output_path = '/home/cad/zyc/chengxu/shujuchuli/docpack/n/'
    #test_path = '/home/cad/zyc/shuju/lunamhd/cubic_normalization_test/'
    root = "/home/cad/zyc/chengxu/shujuchuli/LIDC lung cancer database/"
    fname = os.listdir(root) #加载LIDC lung cancer database中所有文件

    for fn in fname:
        root2 = root + fn     #第一层（LIDC-idrc为第一层）
        print(root2)
        fname2 = os.listdir(root2) #加载LIDC lung cancer database/LIDC/

        for fn2 in fname2:
            root3 = root2 + "/" + fn2    #第二层路径
            fname3 = os.listdir(root3) #j下载LIDC中所有文件
            for fn3 in fname3:
                root4 = root3 + "/" + fn3 + "/"
                if len(os.listdir(root4))<10:
                    continue
                else:
                    print(root4)
                    extract_real_cubic_from_mhd(root4, annatation_file, plot_output_path, normalazation_output_path)
                #extract_fake_cubic_from_mhd(root4, candidate_file, plot_output_path,normalazation_output_path)

    files = [os.path.join(normalazation_output_path, x) for x in os.listdir(normalazation_output_path)]
    print("how many files in normalzation path:  ", len(files))
    real_files = [m for m in files if "real" in m]
    print("how many files in real path:  ", len(real_files))
    #fake_files = [m for m in files if "fake" in m]
    #print("how many files in fake path:  ", len(fake_files))
    for file in real_files:
        angle_transpose(file, 90, "_leftright")
        angle_transpose(file, 180, "_updown")
        angle_transpose(file, 270, "_diagonal")
    print("Final step done...")

    working_path = "/home/cad/zyc/shuju/lunamhd/cubic/"
    file_list = glob(working_path + "*.npy")


