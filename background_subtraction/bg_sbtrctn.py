import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fgbgAdaptiveGaussian = cv2.createBackgroundSubtractorMOG2()
#fgbgAdaptiveGaussian = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    fgbgAdaptiveGaussianmask = fgbgAdaptiveGaussian.apply(frame)
    cv2.namedWindow("BG Adaptive Gaussian",0)
    cv2.imshow("BG Adaptive Gaussian", fgbgAdaptiveGaussianmask)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
print("rogram closed")