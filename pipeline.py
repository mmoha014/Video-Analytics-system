from video_capture import C_VIDEO_UNIT
from preprocessing import C_PREPROCESSING
from Object_detection import C_DETECTION
from tracking import C_TRACKER
from setting import *
from utils import *
import cv2
from Detection_Models.YOLO import C_DETECTION_YOLO as YOLO
import time
import resource
import cProfile, pstats, io

def profile(fnc):
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.print_stats()

        print(s.getvalue())
        # print("morteza")
        return retval,s.getvalue()
    return inner

def test():
    a=0
    for i in range(1000):
        a=i**2+600/300
    return a
# @track
# @profile        
def main():
    # groundtruth_box  = C_PREPROCESSING.VOT2013_read_groundtruth_file(FILE_ADDRESS_DEEP_GROUNDTHRUTH)
    def body(frame, detection):
        predicted_box,frame = detection.Detection_BoundingBox(frame)
        # if frame_number % 20 == 0:
        #     print("change detection method to Deep_YOLO")
        #     detection.Switch_detection("Deep_Yolo")
        # if frame_number % 40 == 0:
        #     print("change detection method to HOG_Pedestrian")
        #     detection.Switch_detection("HOG_Pedestrian")
            # DETECTION_METHOD = "Deep_Yolo"

        # t2 = time.time()
        # time_sum += t2-t1

        # gt = correct_position(groundtruth_box[frame_number-1])
        # tmpgt = correct_position(groundtruth_box[frame_number-1])
        # cv2.rectangle(frame, (gt[0], gt[1]), (gt[2],gt[3]), (0,0,0),0)
        # max_idx, max_acc = 0, 0
        # for i, bb in enumerate(predicted_box):
        #     accuracy = box_iou2(correct_position(bb), correct_position(gt))
        #     if max_acc<accuracy:
        #         max_acc=accuracy
        #         max_idx =  i    
        # acc_per_frame.append(max_acc)
        
        # if len(predicted_box)>0:
        #     pb = correct_position(predicted_box[max_idx])
        #     cv2.rectangle(frame, (pb[0], pb[1]), (pb[2],pb[3]), (155,255,12),1)
        
        
        # Post Processing
        # frame = C_PREPROCESSING.Color_Conversion(frame,"GRAY")
        
        # Output    
        cv2.imshow("output", frame)
        

    detection = C_DETECTION(DETECTION_METHOD)

    video = C_VIDEO_UNIT(INPUT_VIDEO_SNOURCE)
    out1 = cv2.VideoWriter('/home/mgharasu/Documents/Purified codes/outputs/', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (200,200))

    acc_per_frame = []
    ret, frame = video.get_frame()
    image_size = frame.shape    
    frame_number = 0
    time_sum = 0

    while True:        
        ret, frame = video.get_frame()
        t1 = time.time()
        frame_number += 1
        if not ret:
            break
        
        print(frame_number)
        # Preprocessing    
        # frame  = C_PREPROCESSING.resize(frame, min(400, frame.shape[1]))
        # frame = C_PREPROCESSING.Color_Conversion(frame,"GRAY") //deep network needs the color image. If we feed grayscale image, there will be error in code.

        # %time body(frame, detection)
        _,frame = detection.Detection_BoundingBox(frame)
        cv2.imshow("output", frame)
        # t1 = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break #esc to quit
        # Detection
       
        t2=time.time()
        # break
        # print("time elapsed:", t2-t1)

    # print("average acc:",np.average(acc_per_frame),", max acc: ",max(acc_per_frame),", min acc: ",min(acc_per_frame), ", number of frames with zero accuracy: ",len(acc_per_frame)-np.count_nonzero(acc_per_frame))
    # print("The average processing time for every frame is",np.divide(time_sum, frame_number, dtype=np.float), 'and the total number of frames is ', frame_number)
    # print('Also, the total frames (size=',image_size,') are processin in ',time_sum,'seconds')
    cv2.destroyAllWindows() 
    return 2
    
if __name__ == "__main__":  
    # t1=time.time() 
    a=main()

#    print(test())
    
    # t2=time.time()
    # print('pipeline time:', t2-t1)
    

    
