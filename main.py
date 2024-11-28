import cv2
import numpy as np
import utlis


# print("OpenCV version:", cv2.__version__)


##########################################

path = "photos/13.jpg"
widthImg = 500
heightImg = 500

# questions = 24
# choices = 8
# ans = [5,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]

# questions = 5
# choices = 4


# ans = [1,3,0,1,0]



questions = 10
choices = 4
ans = [0,2,1,3,1,2,1,3,3,2]
# ans = [0,2,3,1,0,2,3,1,0,2,3,1,0,2,3,1,0,2,3,1,3]

# ans = [1,2,0,1,4,1,2,0,1,4,1,2,0,1,4,1,2,0,1,4,1,2,0,1,4]
webcamFeed = True
cameraNo = 0
###########################################




        
# WITHOUT CAMERA
img = cv2.imread(path)


# PREPROCESSING
img = cv2.resize(img, (widthImg, heightImg))
imgContour = img.copy()
imgbigContour = img.copy()
imgFinal = img.copy()
checkImg = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)
    
#FINDING ALL CONTOURS
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContour, contours, -1, (0, 255, 0), 10)
#FINDING RECTANGLE
rectCon = utlis.rectContour(contours)
if len(rectCon) < 2:
    print("Not enough contours found!")
    print(len(rectCon))
    
biggestContour = utlis.getCornerPoints(rectCon[0])
gradePoints = utlis.getCornerPoints(rectCon[1])
# print(gradePoints)
if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgbigContour, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(imgbigContour, gradePoints, -1, (0, 0, 255), 20)
    
    biggestContour = utlis.reorder(biggestContour)   
    gradePoints = utlis.reorder(gradePoints)
    # # print(biggestContour)
    # # print(gradePoints)
    
    
    # cv2.polylines(imgbigContour, [biggestContour], True, (0, 255, 0), 20)
    # cv2.polylines(imgbigContour, [gradePoints], True, (255, 0, 0), 20)
    
    #OMR BOX
    pts1 = np.float32(biggestContour)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    
    
    #GRADING BOX
    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0, 0], [325, 0], [0,150], [325,150]])
    matrixGrade = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixGrade, (325, 150))
    # cv2.imshow("Grade", imgGradeDisplay)
    
    
    #APPLY THRESHOLD (change this 170 value to change threshold)
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1] 
    boxes = utlis.splitBoxes(imgThresh,questions,choices)
    # cv2.imshow("Test", boxes[2])
    # print(cv2.countNonZero(boxes[1]), cv2.countNonZero(boxes[2]))
    
    
    
    #Getting NonZero pixel value of each box
    myPixelVal = np.zeros((questions,choices))
    countC = 0
    countR = 0
    
    for image in boxes:
        # cv2.imshow("new_split",image)
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC+=1
        if(countC == choices): countR+=1;countC = 0


    # print(myPixelVal)
    
    
    # FINDING MARKED INDEX VALUES
    myIndex = []
    for x in range(0,questions):
        arr = myPixelVal[x]
        # print("arr: ",arr)
        myIndexVal= np.where(arr == np.amax(arr))
        # print("yes",np.argmax(arr))
        # print(myIndexVal)
        # myIndex.append(int(np.argmax(arr)))    
        myIndex.append(myIndexVal[0][0])
        
    print(myIndex)
    
    
    
    # GRADING
    grading=[]
    for x in range(0,questions):
        if ans[x]==myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)    
    # print(grading)
    score = (sum(grading) / questions)*100
    print("Final Score: ", score)
    
    
    #DISPLAYING ANSWERS (from utlis)
    imgResult = imgWarpColored.copy()
    cords,imgResult = utlis.showAnswers(imgResult, myIndex, grading , ans, questions, choices)
    checkImg = imgResult.copy()
    imgRawDrawing = np.zeros_like(imgWarpColored)
    _,imgRawDrawing = utlis.showAnswers(imgRawDrawing, myIndex, grading , ans, questions, choices)
    invMatrix = cv2.getPerspectiveTransform(pts2, pts1)
    imgInvWarp = cv2.warpPerspective(imgRawDrawing , invMatrix, (widthImg, heightImg))
    
    imgRawGrade = np.zeros_like(imgGradeDisplay)
    cv2.putText(imgRawGrade, str(int(score))+ "%" , (60,100), cv2.FONT_HERSHEY_COMPLEX,3,(150,150,150),3) 
    cv2.imshow("Grade", imgRawGrade)
    invMatrixGrade = cv2.getPerspectiveTransform(ptG2, ptG1)
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixGrade, (widthImg, heightImg))
    
    
    
    
    imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
    imgFinal = cv2.addWeighted(imgFinal,1,imgInvGradeDisplay,1,0)
    
    
# SHOW IMAGES
imgBlank = np.zeros_like(img)
imgArray = ([img, imgGray, imgBlur, imgCanny],
            [imgContour, imgbigContour, imgWarpColored, imgThresh],
            [imgResult,imgRawDrawing, imgInvWarp, imgFinal])



lables = [["Original", "Gray", "Blur","Canny"],
        ["Contours","Biggest Cont", "Warp", "Threshold"],
        ["Result", "Raw Drawing", "Inverse Warp", "Final"]]
imgStack = utlis.StackImages(imgArray,0.5, lables)



cv2.imshow("Final Result", imgFinal)
cv2.imshow("Stacked Images", imgStack)



# After calculating myIndex, grading, and score
output_file = "resultasd.txt"

# Create or overwrite the text file
with open(output_file, 'w') as f:
    f.write("My Index:\n")
    f.write(', '.join(map(str, myIndex)) + '\n\n')  
    
    f.write("PixelVal shape:\n")
    f.write(', '.join(map(str, myPixelVal.shape)) + '\n\n')  
    
    f.write("Grading:\n")
    f.write(', '.join(map(str, grading)) + '\n\n')  
    
    f.write("ImageResultShape:\n")
    f.write(', '.join(map(str, imgResult.shape)) + '\n\n') 
    
    f.write("Cords:\n")
    f.write(', '.join(map(str, cords)) + '\n\n') 
    
    f.write(f"Final Score: {score:.2f}%\n") 

print(f"Results saved to {output_file}")


cv2.waitKey(0)




