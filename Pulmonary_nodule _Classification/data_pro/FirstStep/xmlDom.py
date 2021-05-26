from xml.dom import minidom
import os
import dicom
import glob
import csv
import math


outpath = "/home/cad/zyc/chengxu/shujuchuli/docpack/nodule1.csv"
def xmlRead(path, root2):
    doc = minidom.parse(path)
    try:
        uid = str(doc.getElementsByTagName("SeriesInstanceUID")[0].firstChild.data)
    except (IndexError):
        uid = (doc.getElementsByTagName("SeriesInstanceUid")[0].firstChild.data)
    studyUid = doc.getElementsByTagName("StudyInstanceUID")[0].firstChild.data
    filepath = root2 + '/' + studyUid + '/' + uid + '/'
    dcmpath = glob.glob(filepath + "*.dcm")

    session = doc.getElementsByTagName("readingSession")
    for sess in session:
        unbNodule = sess.getElementsByTagName("unblindedReadNodule")
        for unbn in unbNodule:
            print  (unbn.toxml())
            noduleId = unbn.getElementsByTagName("noduleID")
            margin = unbn.getElementsByTagName("malignancy")
            sopUid = unbn.getElementsByTagName("imageSOP_UID")
            z = unbn.getElementsByTagName("imageZposition")

            x = unbn.getElementsByTagName("xCoord")
            y = unbn.getElementsByTagName("yCoord")

            print("---------1-----------")
            print (unbn.childNodes)
            for g in unbn.childNodes:
                print (g.nodeName)

                if g.nodeName=='characteristics':
            #if len(x) == 1:

                    xmax = xmin = x[0].firstChild.data
                    ymax = ymin = y[0].firstChild.data
                    for xx in x:
                        if xmax < xx.firstChild.data:
                            xmax = xx.firstChild.data
                        elif xmin > xx.firstChild.data:
                            xmin = xx.firstChild.data
                    for yy in y:
                        if ymax < yy.firstChild.data:
                            ymax = yy.firstChild.data
                        elif ymin > yy.firstChild.data:
                            ymin = yy.firstChild.data

                else:
                    continue
                yabs = int(ymax) - int(ymin)
                xabs = int(xmax) - int(xmin)
                if yabs > xabs:
                    r = yabs / 2
                else:
                    r = xabs / 2

                i = 0
                zdot = 0
                for zz in z:
                    i += 1
                    if i == math.ceil(len(z) / 2.0):
                        zdot = zz.firstChild.data

                j = 0
                sliceId = 0
                for sopuid in sopUid:
                    j += 1
                    if j == math.ceil(len(sopUid) / 2.0):
                        sliceId = sopuid.firstChild.data
                        print(sliceId)

                xdot = (int(xmax) + int(xmin)) / 2
                ydot = (int(ymin) + int(ymax)) / 2

                print(xdot, ydot, zdot)

                label = 0
                if len(margin) == 0:
                    print("no margin")
                else:
                    if int(margin[0].firstChild.data) > 2:
                        label = margin[0].firstChild.data
                    else:
                        label = 0
                data = [(studyUid, uid, sliceId, xdot, ydot, zdot, r, label)]
                with open(outpath, 'a+') as csvfile:
                    writer = csv.writer(csvfile, dialect='excel')
                    writer.writerow(data)
                csvfile.close()

            print("---------2-----------")

        nonNodule = sess.getElementsByTagName("nonNodule")
        for nonn in nonNodule:
            noduleId = nonn.getElementsByTagName("noduleID")
            margin = nonn.getElementsByTagName("margin")
            sopUid = nonn.getElementsByTagName("imageSOP_UID")
            z = nonn.getElementsByTagName("imageZposition")
            x = nonn.getElementsByTagName("xCoord")
            y = nonn.getElementsByTagName("yCoord")

            print("---------3-----------")
            if len(x) == 1:
                break
            else:
                xmax = xmin = x[0].firstChild.data
                ymax = ymin = y[0].firstChild.data
                for xx in x:
                    if xmax < xx.firstChild.data:
                        xmax = xx.firstChild.data
                    elif xmin > xx.firstChild.data:
                        xmin = xx.firstChild.data
                for yy in y:
                    if ymax < yy.firstChild.data:
                        ymax = yy.firstChild.data
                    elif ymin > yy.firstChild.data:
                        ymin = yy.firstChild.data

            yabs = int(ymax) - int(ymin)
            xabs = int(xmax) - int(xmin)
            if yabs > xabs:
                r = yabs / 2
            else:
                r = xabs / 2

            i = 0
            zdot = 0
            for zz in z:
                i += 1
                if i == int(len(z) / 2):
                    zdot = zz.firstChild.data

            j = 0
            sliceId = 0
            for sopuid in sopUid:
                j += 1
                if j == int(len(sopUid) / 2):
                    sliceId = sopuid.firstChild.data

            xdot = (int(xmax) + int(xmin)) / 2
            ydot = (int(ymin) + int(ymax)) / 2

            print(xdot, ydot, zdot)

            label = 0
            if len(margin) == 0:
                print("no margin")
            else:
                if int(margin[0].firstChild.data) > 2:
                    label = 1
                else:
                    label = 0

            data = [(studyUid, uid, sliceId, xdot, ydot, zdot, r, label)]
            with open(outpath, 'a+') as csvfile:
                writer = csv.writer(csvfile, dialect='excel')
                writer.writerow(data)
            csvfile.close()
            print("---------4-----------")

