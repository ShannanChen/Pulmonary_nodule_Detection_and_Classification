from xml.dom import minidom
import os
import dicom
import glob
import csv
import math
import pandas as pd

imguid1=[]
imglabel1=[]
imguid2=[]
imglabel2=[]
imguid3=[]
imglabel3=[]
imguid4=[]
imglabel4=[]

n=0
def xmlRead(path, root2):
    doc = minidom.parse(path)
    try:
        uid = str(doc.getElementsByTagName("SeriesInstanceUID")[0].firstChild.data)
    except (IndexError):
        uid = (doc.getElementsByTagName("SeriesInstanceUid")[0].firstChild.data)
    studyUid = doc.getElementsByTagName("StudyInstanceUID")[0].firstChild.data
    filepath = root2 + '/' + studyUid + '/' + uid + '/'
    dcmpath = glob.glob(filepath + "*.dcm")
    txtroot=root2 + '/' + studyUid + '/' + uid + '/' + 'txt' + '/'
    if os.path.exists(txtroot):
        message = 'OK, the "%s" file exists.'
    else:
        os.mkdir(txtroot)


    session = doc.getElementsByTagName("readingSession")
    for t,sess in enumerate(session):


        if t == 0:
            unbNodule = sess.getElementsByTagName("unblindedReadNodule")
            for unbn in unbNodule:
                print(unbn.toxml())
                noduleId = unbn.getElementsByTagName("noduleID")
                margin = unbn.getElementsByTagName("malignancy")
                sopUid = unbn.getElementsByTagName("imageSOP_UID")
                z = unbn.getElementsByTagName("imageZposition")

                x = unbn.getElementsByTagName("xCoord")
                y = unbn.getElementsByTagName("yCoord")

                print("---------1-----------")
                print(unbn.childNodes)
                for g in unbn.childNodes:
                    print(g.nodeName)

                    if g.nodeName == 'characteristics':
                        if g.nodeName == 'characteristics':
                            for (sopuid, zz) in zip(sopUid, z):
                                imguid1.append(sopuid.firstChild.data)
                                imguid1.append(zz.firstChild.data)
                                imguid1.append(margin[0].firstChild.data)

                                print(imguid1)

                        # for sopuid in sopUid, z:
                        #     for n in range(1,len(sopuid)):
                        #         # print(sopuid[n].firstChild.data)
                        #         imguid1.append(sopuid[n].firstChild.data)
                        # imguid1.append(margin[0].firstChild.data)
                        #
                        # print(imguid1)
                        # nnn = int(len(imguid1)/2)
                        # print(nnn)
                        # for ln in range(1,nnn):
                        #    print(imguid1[ln]+"  "+imguid1[ln+nnn]+"   "+imguid1[int(len(imguid1))-1])
                    else:
                        continue

                fl = open(os.path.join(filepath,'txt/','1label1118.txt'), 'w')
                for n, i in enumerate(imguid1):
                    # for n, i in imguid:
                    print(n, i)
                    fl.write((str(i)))
                    fl.write((str(' ')))

                    if (n + 1) % 3 == 0:
                        fl.write((str('\n')))
                fl.close()
        if t == 1:
            unbNodule = sess.getElementsByTagName("unblindedReadNodule")
            for unbn in unbNodule:
                print(unbn.toxml())
                noduleId = unbn.getElementsByTagName("noduleID")
                margin = unbn.getElementsByTagName("malignancy")
                sopUid = unbn.getElementsByTagName("imageSOP_UID")
                z = unbn.getElementsByTagName("imageZposition")

                x = unbn.getElementsByTagName("xCoord")
                y = unbn.getElementsByTagName("yCoord")

                print("---------1-----------")
                print(unbn.childNodes)
                for g in unbn.childNodes:
                    print(g.nodeName)

                    if g.nodeName == 'characteristics':
                        if g.nodeName == 'characteristics':
                            for (sopuid, zz) in zip(sopUid, z):
                                imguid2.append(sopuid.firstChild.data)
                                imguid2.append(zz.firstChild.data)
                                imguid2.append(margin[0].firstChild.data)

                                print(imguid2)

                                # for sopuid in sopUid, z:
                                #     for n in range(1,len(sopuid)):
                                #         # print(sopuid[n].firstChild.data)
                                #         imguid1.append(sopuid[n].firstChild.data)
                                # imguid1.append(margin[0].firstChild.data)
                                #
                                # print(imguid1)
                                # nnn = int(len(imguid1)/2)
                                # print(nnn)
                                # for ln in range(1,nnn):
                                #    print(imguid1[ln]+"  "+imguid1[ln+nnn]+"   "+imguid1[int(len(imguid1))-1])
                    else:
                        continue
                fl = open(os.path.join(filepath, 'txt/', '2label1118.txt'), 'w')
                for n, i in enumerate(imguid2):
                    # for n, i in imguid:
                    print(n, i)
                    fl.write((str(i)))
                    fl.write((str(' ')))

                    if (n + 1) % 3 == 0:
                        fl.write((str('\n')))
                fl.close()
        if t == 2:
            unbNodule = sess.getElementsByTagName("unblindedReadNodule")
            for unbn in unbNodule:
                print(unbn.toxml())
                noduleId = unbn.getElementsByTagName("noduleID")
                margin = unbn.getElementsByTagName("malignancy")
                sopUid = unbn.getElementsByTagName("imageSOP_UID")
                z = unbn.getElementsByTagName("imageZposition")

                x = unbn.getElementsByTagName("xCoord")
                y = unbn.getElementsByTagName("yCoord")

                print("---------1-----------")
                print(unbn.childNodes)
                for g in unbn.childNodes:
                    print(g.nodeName)

                    if g.nodeName == 'characteristics':
                        if g.nodeName == 'characteristics':
                            for (sopuid, zz) in zip(sopUid, z):
                                imguid3.append(sopuid.firstChild.data)
                                imguid3.append(zz.firstChild.data)
                                imguid3.append(margin[0].firstChild.data)

                                print(imguid3)

                                # for sopuid in sopUid, z:
                                #     for n in range(1,len(sopuid)):
                                #         # print(sopuid[n].firstChild.data)
                                #         imguid1.append(sopuid[n].firstChild.data)
                                # imguid1.append(margin[0].firstChild.data)
                                #
                                # print(imguid1)
                                # nnn = int(len(imguid1)/2)
                                # print(nnn)
                                # for ln in range(1,nnn):
                                #    print(imguid1[ln]+"  "+imguid1[ln+nnn]+"   "+imguid1[int(len(imguid1))-1])
                    else:
                        continue

                fl = open(os.path.join(filepath, 'txt/', 'label.txt'), 'w')
                for n, i in enumerate(imguid3):
                    # for n, i in imguid:
                    print(n, i)
                    fl.write((str(i)))
                    fl.write((str(' ')))

                    if (n + 1) % 3 == 0:
                        fl.write((str('\n')))
                fl.close()
        if t==3:
            unbNodule = sess.getElementsByTagName("unblindedReadNodule")
            for unbn in unbNodule:
                print(unbn.toxml())
                noduleId = unbn.getElementsByTagName("noduleID")
                margin = unbn.getElementsByTagName("malignancy")
                sopUid = unbn.getElementsByTagName("imageSOP_UID")
                z = unbn.getElementsByTagName("imageZposition")

                x = unbn.getElementsByTagName("xCoord")
                y = unbn.getElementsByTagName("yCoord")

                print("---------1-----------")
                print(unbn.childNodes)
                for g in unbn.childNodes:
                    print(g.nodeName)

                    if g.nodeName == 'characteristics':
                        if g.nodeName == 'characteristics':
                            for (sopuid, zz) in zip(sopUid, z):
                                imguid4.append(sopuid.firstChild.data)
                                imguid4.append(zz.firstChild.data)
                                imguid4.append(margin[0].firstChild.data)

                                print(imguid4)

                                # for sopuid in sopUid, z:
                                #     for n in range(1,len(sopuid)):
                                #         # print(sopuid[n].firstChild.data)
                                #         imguid1.append(sopuid[n].firstChild.data)
                                # imguid1.append(margin[0].firstChild.data)
                                #
                                # print(imguid1)
                                # nnn = int(len(imguid1)/2)
                                # print(nnn)
                                # for ln in range(1,nnn):
                                #    print(imguid1[ln]+"  "+imguid1[ln+nnn]+"   "+imguid1[int(len(imguid1))-1])
                    else:
                        continue

                fl = open(os.path.join(filepath, 'txt/', '4label1118.txt'), 'w')
                for n, i in enumerate(imguid4):
                    # for n, i in imguid:
                    print(n, i)
                    fl.write((str(i)))
                    fl.write((str(' ')))

                    if (n + 1) % 3 == 0:
                        fl.write((str('\n')))
                fl.close()

    # txtfile = root2 + '/' + studyUid + '/' + uid + '/'+'txt'+'/'
    #
    # file_list = glob.glob(txtfile + "*.txt")
    # for x, txt in enumerate(file_list):
    #     print(txt)
    #     print(file_list(2))
    #






