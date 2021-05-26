from xml.dom import minidom
import os
import dicom
import glob



def nameChange(xmlpath, root2):

    doc = minidom.parse(xmlpath)
    try:
        uid = str(doc.getElementsByTagName("SeriesInstanceUID")[0].firstChild.data)
    except (IndexError):
        uid = str(doc.getElementsByTagName("SeriesInstanceUid")[0].firstChild.data)


    studyUid = str(doc.getElementsByTagName("StudyInstanceUID")[0].firstChild.data)
    filepath = root2 + '/' + studyUid + '/' + uid + '/'
    dcmpath = glob.glob(filepath + "*.dcm")

    for dp in dcmpath:
        sopuid = dicom.read_file(dp).SOPInstanceUID
        os.rename(dp, filepath + sopuid +".dcm")
