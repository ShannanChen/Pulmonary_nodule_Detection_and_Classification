import dcmChange
import xmlDom
import os
import glob
import findlabel

#    dcmChange.nameChange(path)
root = "/home/cad/zyc/1118python/"
fname = os.listdir(root)
for fn in fname:
    root2 = root + fn
    print(root2)
    fname2 = os.listdir(root2)

    for fn2 in fname2:
        root3 = root2 + "/" + fn2
        fname3 = os.listdir(root3)
        for fn3 in fname3:
            root4 = root3 + "/" + fn3 +"/"
            print(root4)
#            fname4 = os.listdir(root4)
#            for fn4 in fname4:
#               print()
            xmlpath = glob.glob(root4 + "*.xml")
            if len(xmlpath) == 0:
                break
            else:
                print(xmlpath[0])
                #print("---------namechage begin----------")
                #dcmChange.nameChange(xmlpath[0], root2)
                #print("----------namechage done-----------")
                print("-----------xmlDom begin------------")
                #xmlDom.xmlRead(xmlpath[0], root2)
                findlabel.xmlRead(xmlpath[0], root2)
                print("-----------xmlDom done------------")