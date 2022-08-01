import cv2
import numpy as np
import random
import os
import time

print("Packages Imported")


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def importingPictures():
    # importing picture

    img = cv2.imread("Resources/man.png")
    cv2.imshow("output", img)
    cv2.waitKey(0)

#importingPictures()


def importingVideo():
    # importing video

    cap = cv2.VideoCapture("Resources/video/test_video.mp4")

    while True:
        success, img = cap.read()
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#importingVideo()

def laptopCamera():
    # IMPORTING MY LAPTOP CAMERA

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(10, 100)

    while True:
        success, img = cap.read()
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


#laptopCamera()

def changesToPictures():
    # making some manipulations on the picture
    img = cv2.imread("Resources/man.png")
    kernel = np.ones((5, 5), np.uint8)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgCanny = cv2.Canny(img, 150, 200)
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
    imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

    # cv2.imshow("Blur Image", imgBlur)
    # cv2.imshow("Gray Image", imgGray)
    # cv2.imshow("Canny Image", imgCanny)
    # cv2.imshow("Dialation Image", imgDialation)
    # cv2.imshow("Eroded Image", imgEroded)

    imgStack = stackImages(0.6, ([img, imgGray, imgBlur], [imgCanny, imgEroded, imgDialation]))
    cv2.imshow("stackedImages", imgStack)

    cv2.waitKey(0)


#changesToPictures()

def resizeAndCrop():
    # resizing and cropping image
    img = cv2.imread("Resources/man.png")
    print(img.shape)
    cv2.imshow("Image", img)

    imgResize = cv2.resize(img, (300, 300))
    cv2.imshow("Image Resize", imgResize)
    print(imgResize.shape)

    imgCropped = img[0:250, 100:300]
    cv2.imshow("Image Cropped", imgCropped)

    cv2.waitKey(0)


def shapesAndText():
    # shapes and text

    img = np.zeros((512, 512, 3), np.uint8)
    cv2.line(img, (0, 0), (512, 512), (0, 255, 0), 3)
    cv2.rectangle(img, (0, 0), (350, 350), (0, 0, 250), 3)
    cv2.rectangle(img, (0, 0), (250, 250), (255, 0, 0), cv2.FILLED)
    cv2.circle(img, (420, 50), 30, (255, 255, 0), 3)
    cv2.putText(img, "OPENCV", (360, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 1)
    cv2.imshow("Image", img)

    cv2.waitKey(0)

#shapesAndText()


def warpPerspective():
    # image warpPerspective
    img = cv2.imread("Resources/cards.jpg")

    width, height = 250, 350
    pts1 = np.float32([[576, 134], [717, 140], [562, 262], [730, 270]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))

    cv2.imshow("Image", img)
    cv2.imshow("Image Output", imgOutput)

    cv2.waitKey(0)


def joiningImages():
    # joining images

    img = cv2.imread("Resources/man.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # imgHor = np.hstack((img, img))
    # imgVer = np.vstack((img, img))
    imgStack = stackImages(0.5, ([img, imgGray, img], [img, img, img]))

    # cv2.imshow("Horizontal", imgHor)
    # cv2.imshow("Vertical", imgVer)
    cv2.imshow("ImageStack", imgStack)

    cv2.waitKey(0)


def colorDetection():
    # color detection
    def empty(a):
        pass

    cv2.namedWindow("TrackBar")
    cv2.resizeWindow("TrackBar", 640, 240)
    cv2.createTrackbar("Hue min", "TrackBar", 0, 179, empty)
    cv2.createTrackbar("Hue max", "TrackBar", 179, 179, empty)
    cv2.createTrackbar("Sat min", "TrackBar", 44, 255, empty)
    cv2.createTrackbar("Sat max", "TrackBar", 255, 255, empty)
    cv2.createTrackbar("Val min", "TrackBar", 172, 255, empty)
    cv2.createTrackbar("Val max", "TrackBar", 255, 255, empty)

    while True:
        img = cv2.imread("Resources/car.jpg")
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("Hue min", "TrackBar")
        h_max = cv2.getTrackbarPos("Hue max", "TrackBar")
        s_min = cv2.getTrackbarPos("Sat min", "TrackBar")
        s_max = cv2.getTrackbarPos("Sat max", "TrackBar")
        v_min = cv2.getTrackbarPos("Val min", "TrackBar")
        v_max = cv2.getTrackbarPos("Val max", "TrackBar")
        print(h_min, h_max, s_min, s_max, v_min, v_max)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)

        # cv2.imshow("Original", img)
        # cv2.imshow("HSV", imgHSV)
        # cv2.imshow("Mask", mask)
        # cv2.imshow("image Result", imgResult)

        imgStack = stackImages(0.6, ([img, imgHSV], [mask, imgResult]))
        cv2.imshow("stacked images", imgStack)

        if cv2.waitKey(1) == ord('q'):
            break


def shapesDetection():
    # contours/ shapes detection

    def getContours(img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            print(area)
            cv2.drawContours(imgContour, cnt, -1, (0, 0, 0), 2)
            peri = cv2.arcLength(cnt, True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor == 3:
                objectType = "Tri"
            elif objCor == 4:
                aspRatio = w / float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = "Squre"
                else:
                    objectType = "Rec"
            elif objCor > 4:
                objectType = "circle"
            else:
                objectType = "None"

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(imgContour, objectType, (x + (w // 2) - 10, y + (w // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 0, 0), 1)

    img = cv2.imread("Resources/shapes2.png")
    imgContour = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    getContours(imgCanny)
    imgBlank = np.zeros_like(img)
    imgStack = stackImages(1, ([img, imgGray, imgBlur], [imgCanny, imgContour, imgBlank]))

    cv2.imshow("imgStack", imgStack)

    cv2.waitKey(0)


#shapesDetection()

def pictureFaceDetection():
    # face detection

    faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")
    img = cv2.imread("Resources/people.webp")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Result", img)

    cv2.waitKey(0)




def negative():
    # negative image

    img = cv2.imread("Resources/man2.png")
    imgNegative = 255 - img
    imgStack = stackImages(0.4, [img, imgNegative])
    cv2.imshow("Stacked Images1", imgStack)

    cv2.waitKey(0)


def saveImage():
    # how to save image in a directory that you want

    img = cv2.imread("Resources/man.png")
    imgCopy = img.copy()
    imgCopy = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('Resources/manGray.png', imgCopy)
    cv2.imshow("output", imgCopy)
    cv2.waitKey(0)


def randomImage():
    # displaying random image
    path = r"C:\Users\אביתר שלומי\PycharmProjects\opencvProject\Resources"
    random_filename = random.choice([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
    ])
    print(random_filename)

    img = cv2.imread(f"Resources/{random_filename}")
    cv2.imshow("img", img)

    cv2.waitKey(0)



def imageOnVideo():
    # load the overlay image. size should be smaller than video frame size
    img = cv2.imread('Resources/man3.png')

    # Get Image dimensions
    img_height, img_width, _ = img.shape

    # Start Capture
    cap = cv2.VideoCapture(0)

    # Get frame dimensions
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Print dimensions
    print('image dimensions (HxW):', img_height, "x", img_width)
    print('frame dimensions (HxW):', int(frame_height), "x", int(frame_width))

    # Decide X,Y location of overlay image inside video frame.
    # following should be valid:
    #   * image dimensions must be smaller than frame dimensions
    #   * x+img_width <= frame_width
    #   * y+img_height <= frame_height
    # otherwise you can resize image as part of your code if required

    x = 300
    y = 100

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # add image to frame
        frame[y:y + img_height, x:x + img_width] = img

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Exit if ESC key is pressed
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



def videoFaceDetection():
    # video face detection
    # name  = input("enter your name\n")
    faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "face", (x + (w // 2) - 100, y + (h - 180)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
            # cv2.putText(img, name + "'s face", (x + (w // 2) - 100, y + (h - 180)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


#videoFaceDetection()

def changingImages():
    # loading random image and taking the size
   while True:
       path = r"C:\Users\אביתר שלומי\PycharmProjects\opencvProject\Resources"
       random_filename = random.choice([
           x for x in os.listdir(path)
           if os.path.isfile(os.path.join(path, x))
       ])
       print(random_filename)

       img = cv2.imread(f"Resources/{random_filename}")
       new_img = cv2.resize(img, (200, 200))


       cv2.imshow("Video", new_img)
       if cv2.waitKey(250) & 0xFF == ord('q'):
           break



def facedetection_filter():
    # video face detection with picture above your head

    # face classifier and  frame size
    faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    numberOfSeconds = 0
    startingTime = int(time.time())
    frameNumber = 0

    while True:
        success, frame = cap.read()
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.3, minNeighbors=5)

        # loading random image every fourth frame
        if frameNumber % 4 == 0 and numberOfSeconds < 7:
            path = r"C:\Users\אביתר שלומי\PycharmProjects\opencvProject\Resources\tanks"
            random_filename = random.choice([
                x for x in os.listdir(path)
                if os.path.isfile(os.path.join(path, x))
            ])


            img = cv2.imread(f"Resources/tanks/{random_filename}")

        for (x, y, w, h) in faces:
            # changing the size of the image based on the size of the face
            new_img = cv2.resize(img, (w // 2, h // 2))
            new_img_height, new_img_width, _ = new_img.shape

            miny = min(y, h // 2 + 20)
            minx = min(x, w // 4)

            # if the picture getting out of the frame move back
            if (frame_width - 1) <= x + new_img_width:
                x = x - (frame_width + 1 - new_img_width)

            # the location of the image in the frame
            frame[y - miny:y - miny + new_img_height, x + minx:x + minx + new_img_width] = new_img

            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.putText(frame, "face", (x + (w // 2) - 100, y + (h - 180)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

            if random_filename == "namer.jpg":
                cv2.putText(frame, "namer", (x + minx, y - miny), cv2.FONT_HERSHEY_COMPLEX, new_img_height/70, (0, 0, 0), 2)
            if random_filename == "merkava4.jpg":
                cv2.putText(frame, "merkava4", (x + minx, y - miny), cv2.FONT_HERSHEY_COMPLEX, new_img_height/70, (0, 0, 0), 2)
            if random_filename == "merkava3.jpg":
                cv2.putText(frame, "merkava3", (x + minx, y - miny), cv2.FONT_HERSHEY_COMPLEX, new_img_height/70, (0, 0, 0), 2)
            if random_filename == "merkava1.jpg":
                cv2.putText(frame, "merkava1", (x + minx, y - miny), cv2.FONT_HERSHEY_COMPLEX, new_img_height/70, (0, 0, 0), 2)
            if random_filename == "eytan.jpg":
                cv2.putText(frame, "eytan", (x + minx, y - miny), cv2.FONT_HERSHEY_COMPLEX, new_img_height/70, (0, 0, 0), 2)

        frameNumber += 1
        currentTime = int(time.time())
        numberOfSeconds = currentTime - startingTime


        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break





#facedetection_filter()




