import cv2
import setting
from utils import C_VIDEO_UNIT, C_PREPROCESSING


face_cascade = cv2.CascadeClassifier()#'/home/morteza/Desktop/Dr. Jain/project implementation/week2/Purified codes/haar_cascade_pretrianed_classifiers/visonary/haarcascade_fullbody.xml')
ped=face_cascade.load('visionary.net_pedestrian_cascade_web_LBP.xml')
vc = C_VIDEO_UNIT("/media/morteza/+989127464877/DataSets/VOT_Dataset/VOT2013/iceskater/color/",True,"hacar_cascade_output.avi")#cv2.VideoCapture('video.avi')

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
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
        
vc.release()