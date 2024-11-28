import cv2
import numpy as np


def StackImages(imgArray,scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        
        
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank]*rows
        hor_con = [imgBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
        
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver, (c*eachImgWidth, eachImgHeight*d), (c*eachImgWidth + len(lables[d][c])*13+27, 30+eachImgHeight*d), (255,255,255), cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth*c+10, eachImgHeight*d+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,255), 2) 

    
    return ver












def rectContour(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        # print(area)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon




def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx





def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    # print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)] #[0,0]
    myPointsNew[3] = myPoints[np.argmax(add)] #[h,w]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)] #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]
    return myPointsNew




def splitBoxes(img,questions,choices):
    vsplit_num = questions
    hsplit_num = choices
    
    boxes = []
    rows = np.vsplit(img,vsplit_num)
    for r in rows:
        cols = np.hsplit(r,hsplit_num)
        for box in cols:
            boxes.append(box)
            # cv2.imshow("Split",box)
        # cv2.imshow("Split",r)  
    
    return boxes





def showAnswers(img, myIndex, grading , ans, questions, choices):
    secH = int(img.shape[1] / questions)
    secW = int(img.shape[0]/choices)
    
    cords = []
    cords.append([secW,secH])
    
    for x in range(0,questions):
        myAns = myIndex[x]
        cX = (myAns*secW) + secW//2
        cY = (x*secH) + secH//2
        
        cords.append([cX,cY])
        
        # COLOR BASED ON RIGHT OR WRONG MARKING
        if grading[x]==1:
            myColor = (0,255,0)
        else:
            myColor = (0,0,255)
            correctAns = ans[x]
            
            cv2.circle(img,((correctAns*secW)+secW//2,(x*secH)+secH//2),20,(0,100,0),cv2.FILLED)
        
        cv2.circle(img,(cX,cY),20,myColor,cv2.FILLED)
    
    return cords,img