import cv2 as cv

video = cv.VideoCapture(0)

while True:
    # Захват видео с вебки; ret -> Bool
    ret, frame = video.read()
    frameCopy = frame
    if not ret:
        print('Something wrong with your cam')

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv = cv.blur(hsv, (5,5))
    mask = cv.inRange(hsv, (89, 124, 73), (255, 255, 255))
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)
    cv.imshow('Result', mask)

    contour = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contour = contour[1]
    if contour:
        contour = sorted(contour, key=cv.contourArea, reverse=True)
        cv.drawContours(frame, contour, 0, (255, 0, 255), 3)

        (x, y, w, h) = cv.boundingRect(contour[0])
        cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv.imshow('Rectangle', frame)

        roImg = frameCopy[y:y+h, x:x+w]
        cv.imshow('Object', roImg)

    if cv.waitKey(1) == ord('q'):
        break

video.release()
video.destroyAllWindows()
