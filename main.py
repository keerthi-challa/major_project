import os

import cvzone
from cvzone.ClassificationModule import Classifier
import cv2

cap = cv2.VideoCapture(0)

classifier = Classifier(r'C:\Users\DELL\PycharmProjects\pythonProject1\Model\keras_model.h5', r'C:\Users\DELL\PycharmProjects\pythonProject1\Model\labels.txt')
imgArrow = cv2.imread(r'C:\Users\DELL\PycharmProjects\pythonProject1\arrow.png', cv2.IMREAD_UNCHANGED)
classIDBin = 0
# Import all the waste images
imgWasteList = []
pathFolderWaste = r"C:\Users\DELL\PycharmProjects\pythonProject1\Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

# Import all the waste images
imgBinsList = []
pathFolderBins = r"C:\Users\DELL\PycharmProjects\pythonProject1\Bins"
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

classDic = {1: 1,
            2: 2}

while True:
    _, img = cap.read()
    imgResize = cv2.resize(img, (454, 340))

    imgBackground = cv2.imread(r'C:\Users\DELL\PycharmProjects\pythonProject1\background.png')

    predection = classifier.getPrediction(img)

    classID = predection[1]
    print(classID)
    if classID != 0:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))

        classIDBin = classDic[classID]

    imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))

    imgBackground[148:148 + 340, 159:159 + 454] = imgResize
    # Displays
    cv2.imshow("Image", img)
   # cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)
