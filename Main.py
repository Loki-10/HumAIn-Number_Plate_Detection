import cv2
import numpy as np
import os
import DetectChars
import DetectPlates
import PossiblePlate

Black_colour = (0.0, 0.0, 0.0)
White_colour = (255.0, 255.0, 255.0)
Yellow_colour = (0.0, 255.0, 255.0)
Green_colour = (0.0, 255.0, 0.0)
Red_colour = (0.0, 0.0, 255.0)

def main():
    img_path="25.jpeg"
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

    imgOriginalScene  = cv2.imread(img_path)

    if imgOriginalScene is None:
        print("\nInvalid Image Name : Image is not present in that path !!!\n\n")
        os.system("pause")
        return


    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

    if len(listOfPossiblePlates) == 0:
        print("\nNo Number plates were detected\n")
    else:

        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

        licPlate = listOfPossiblePlates[0]

        if len(licPlate.strChars) == 0:
            print("\nNo characters were detected\n\n")
            return

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

        print("\nFounded Characters and Numbers in Number plate:  = " + licPlate.strChars +  "\n")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)

        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)

    cv2.waitKey(0)

    return

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Red_colour, 2)        
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Red_colour, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Red_colour, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Red_colour, 2)

def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(plateHeight) / 30.0
    intFontThickness = int(round(fltFontScale * 1.5))

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)


    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)

    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))

    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

if __name__ == "__main__":
    main()
