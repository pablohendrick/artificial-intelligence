import numpy as np
import cv2
import sys
from time import sleep

VIDEO = 'D:/MEI/Portfólio/movement-detection/Dados/Ponte.mp4'

algorithm_types = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']

def Kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3,3), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)
    return kernel

def Filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, Kernel('dilation'), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, Kernel('dilation'), iterations=2)
        return dilation
    
def Subtractor(algorithm_type):
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames = 120, 
                                                        decisionThreshold=0.8)
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG(history = 100, nmixtures = 5,
                                                        backgroundRatio = 0.7, 
                                                        noiseSigma = 0)
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2(history = 500, detectShadows=True,
                                                varThreshold=100)
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, 
                                                 detectShadows=True)
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, 
                                                        useHistory =True,
                                                        maxPixelStability=15*60,
                                                        isParallel=True)
    print('Invalidate Detect')
    sys.exit(1)

w_min = 50  
h_min = 50  
offset = 2  
line_ROI = 620  
cars = 0

def centroid(x, y, w, h):
    x1 = w // 2
    y1 = h // 2
    cx = x + x1
    cy = y + y1
    return cx, cy
  
detec = []
def set_info(detec):
    global cars
    for (x, y) in detec:
        if (line_ROI + offset) > y > (line_ROI - offset):
            cars += 1
            cv2.line(frame, (25, line_ROI), (1200, line_ROI), (0, 127, 255), 3)
            detec.remove((x, y))
            print("Cars detected so far: " + str(cars))

def show_info(frame, mask):
    text = f'Cars: {cars}'
    cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Original Video", frame)
    cv2.imshow("To detect", mask)

cap = cv2.VideoCapture(VIDEO)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

algorithm_type = algorithm_types[1]
background_subtractor = Subtractor(algorithm_type)

while True:
    
    hasFrame, frame = cap.read()
    
    if not hasFrame:
        break
    
    mask = background_subtractor.apply(frame)
    mask = Filter(mask, 'combine')
    
    contour, img = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (25, line_ROI), (1200, line_ROI), (255, 127, 0), 3)
    for (i, c) in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contour = (w >= w_min) and (h >= h_min)
        if not validate_contour:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = centroid(x, y, w, h)
        detec.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

    set_info(detec)
    show_info(frame, mask)
    
    if cv2.waitKey(1) == 27: #ESC
        break

cv2.destroyAllWindows()
cap.release()
