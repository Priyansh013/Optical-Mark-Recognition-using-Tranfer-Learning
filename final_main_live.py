import cv2
import numpy as np
import utlis5 as utlis  
import time

# Constants
path = "1.jpg"  
webCamFeed = True  
widthImg = 500 
heightImg = 500  
questions = 5  
choices = 5  
ans = [1, 2, 2, 3, 1]  
cameraNo = 0  

# Initialize Camera
cap = cv2.VideoCapture(cameraNo)
cap.set(10, 150)  

# Variables to track contour stability
previous_contours = None
contour_stable_time = 0.01  # Time in seconds to consider contours as stable
last_contour_change_time = time.time()  

while True:
    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    # Preprocessing
    img = cv2.resize(img, (widthImg, heightImg))  
    imgContours = img.copy()
    imgFinal = img.copy()
    imgBiggestContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Apply Gaussian blur
    imgCanny = cv2.Canny(imgBlur, 10, 70)  # Detect edges

    imgStacked = np.zeros_like(img)  

    try:
        # Find All Contours
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) 

        # Find Rectangle Contours 
        rectCon = utlis.rectContour(contours)

        if len(rectCon) < 2:
            print("Not enough contours found!")
            continue
        # getting corner points
        biggestContour = utlis.getCornerPoints(rectCon[0])  
        gradePoints = utlis.getCornerPoints(rectCon[1])  

        if biggestContour.size != 0 and gradePoints.size != 0:
            biggestContour = utlis.reorder(biggestContour)  
            gradePoints = utlis.reorder(gradePoints)

            # Perspective Transformation for the Biggest Contour (Main Document)
            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # Perspective Transformation for Grade Contour (Grade Box)
            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

            # Threshold Image (for Detecting Boxes)
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.adaptiveThreshold(imgWarpGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, 11, 2)

            # Split Boxes (divide the threshold image into individual boxes for answers)
            boxes = utlis.splitBoxes(imgThresh)

            # Initialize Pixel Values
            myPixelVal = np.zeros((questions, choices))
            countC = 0
            countR = 0

            for image in boxes:
                totalPixels = cv2.countNonZero(image)  # Count non-zero pixels 
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if countC == choices:
                    countR += 1
                    countC = 0

            # Find the Index Values of the Marked Answers
            myIndex = []
            for x in range(0, questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))  # Find the highest value in each row
                myIndex.append(myIndexVal[0][0])

            # Grading Logic
            grading = []
            for x in range(0, questions):
                if ans[x] == myIndex[x]:  
                    grading.append(1)
                else:
                    grading.append(0)

            score = (sum(grading) / questions) * 100 
            print(f"Score: {score}%")

            # Prepare Final Result Image (Without Green Contours)
            imgResult = imgWarpColored.copy()
            imgResult = utlis.showAnswers(imgResult, myIndex, grading, ans, questions, choices)  

            # Draw Grade on the Grade Box
            imgRawGrade = np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade, str(int(score)) + '%', (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)

            # Inverse Warp for Final Output
            invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvWarp = cv2.warpPerspective(imgResult, invMatrix, (widthImg, heightImg))
            invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

            # Combine Final Images
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

            # Check if contours are stable 
            if previous_contours is not None and np.array_equal(biggestContour, previous_contours):
                # If contours haven't changed for 0.05 second, capture and display the image
                if time.time() - last_contour_change_time >= contour_stable_time:
                    print("Contours stable for 1 second, capturing image...")
                    cv2.imwrite('captured_image.jpg', imgFinal)  
                    cv2.imshow("Captured Image", imgFinal)  # captured image display
            else:
                last_contour_change_time = time.time()  

            # Update previous contours for next comparison
            previous_contours = biggestContour

        # Stack Images for Display
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                      [imgContours, imgBiggestContours, imgWarpColored, imgThresh],
                      [imgResult, imgRawGrade, imgInvWarp, imgFinal])

        lables = [["Original", "Gray", "Blur", "Canny"],
                  ["Contours", "Biggest Contour", "Warp", "Threshold"],
                  ["Result", "Grade", "Inv Warp", "Final"]]

        imgStacked = utlis.stackImages(imageArray, 0.5, lables)

    except Exception as e:
        print(f"Error: {e}")
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                      [imgContours, imgBiggestContours, imgWarpColored, imgThresh],
                      [imgResult, imgGradeDisplay, imgInvWarp, imgFinal])
        imageArrayResized = [[cv2.resize(img, (widthImg, heightImg)) for img in row] for row in imageArray]

        imgStacked = utlis.stackImages(imageArrayResized, 0.5, lables)

    # Final Display
    cv2.imshow('Stacked Images', imgStacked)

    # Save the final image when the user presses 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('final.jpg', imgFinal)
        cv2.waitKey(300)
