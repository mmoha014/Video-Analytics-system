import cv2
from mainHaar import VIdeo_Unit, Preprocessing
import matplotlib.pyplot as plt
import setting

plt.ion()

face_cascade = cv2.CascadeClassifier('../haar_cascade_pretrianed_classifiers/visionary/chaarcascade_fullbody.xml')
vc = VIdeo_Unit("/media/morteza/+989127464877/DataSets/VOT Dataset/VOT2013/iceskater/color",True,"hacar_cascade_output.avi")#cv2.VideoCapture('video.avi')

# if vc.isOpened():
#     rval , frame = vc.read()
# else:
#     rval = False

i = 0
rval = True
while rval:    
    rval, frame = vc.get_frame()
    print(i)
    i += 1
    # car detection.
    
    cars = face_cascade.detectMultiScale(frame, 1.1, 2)
    
    ncars = 0
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        ncars = ncars + 1
    vc.write_output_video(frame)
    # show result
    # plt.imshow(frame)
    # plt.pause(0.001)
    # plt.show()
    cv2.imshow("Result",frame)
    cv2.waitKey(1);
    vc.release()