from video_capture import C_VIDEO_UNIT
from preprocessing import C_PREPROCESSING
from Object_detection import C_DETECTION
from tracking import C_TRACKER
from setting import *
from utils import *
import cv2
from Detection_Models.YOLO import C_DETECTION_YOLO as YOLO
from MOT_File_Generator import C_MOT_OUTPUT_GENERATER

# 
# @track
def main():
    # groundtruth_box  = C_PREPROCESSING.VOT2013_read_groundtruth_file(FILE_ADDRESS_DEEP_GROUNDTHRUTH)
    
    #['faster_rcnn_X_101_32x8d_FPN_3x','faster_rcnn_R_101_DC5_3x','retinanet_R_101_FPN_3x','faster_rcnn_R_50_FPN_3x']
    detection = C_DETECTION(DETECTION_METHOD)
    # detection2 = C_DETECTION('faster_rcnn_R_101_DC5_3x')

    video = C_VIDEO_UNIT(INPUT_VIDEO_SOURCE)#"/home/mgharasu/Videos/traffic camera/1/")#"/media/mgharasu/+989127464877/DataSets/Virginia Traffic Videos/VB/video1")
    MOT =  C_MOT_OUTPUT_GENERATER(FILE_ADDRESS_DEEP_GROUNDTHRUTH)#"/home/mgharasu/Videos/traffic camera/output/gt1.txt")

    acc_per_frame = []
    # ret, frame = video.get_frame()
    # image_size = frame.shape    
    frame_number = 0
    time_sum = 0
    start_frame = 0
    shared = None
    while True:
        ret, frame = video.get_frame_position(start_frame,shared)

        
        if not ret:
            break
        frame = cv2.resize(frame,(960,960))
        print(start_frame)
        
        start_frame += 1
        # Detection
        predicted_box,frame = detection.Detection_BoundingBox(frame)
        boxes = []
        for i, b in enumerate(predicted_box):
                boxes.append([i, b[0],b[1],b[2],b[3]])
        if len(boxes)>0:
            MOT.write(start_frame,boxes)
        else:
            MOT.write(start_frame,[[0,0,0,0,0]])
        # print(boxes)
        frame = draw_box(predicted_box, frame)
        # cv2.imshow("output", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break #esc to quit
    
    cv2.destroyAllWindows()
    MOT.close()
    
    
if __name__ == "__main__":  
    # t1=time.time() 
    main()
    # t2=time.time()
    # print('pipeline time:', t2-t1)
    

    
