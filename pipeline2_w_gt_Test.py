from video_capture import C_VIDEO_UNIT
# from multitracking import box_iou2, correct_position
from preprocessing import C_PREPROCESSING
from Object_detection import C_DETECTION
from tracking import C_TRACKER
from utils import *

from setting import *
import cv2

# from mem_track import track


# @track
def main():
    #>>>>>>>>>>>>>>>>>>>>>>>>>> begin <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<    
    detection = C_DETECTION(DETECTION_METHOD)
    video = C_VIDEO_UNIT(INPUT_VIDEO_SNOURCE,True, OUTPUT_VIDEO_WRITE)

    tracker = []

    acc_per_frame = []
    ret, frame = video.get_frame()
    image_size = frame.shape
    frame_number = 0
    time_sum = 0

    search_for_frame_to_detect_object = True

    while True:
        ret, frame = video.get_frame()
        if not ret:
            break

        frame_number += 1

        if frame_number % UPDATE_TRACKER == 0:
            search_for_frame_to_detect_object = True        
        # Preprocessing    
        # frame  = C_PREPROCESSING.resize(frame, min(400, frame.shape[1]))
        # frame = C_PREPROCESSING.Color_Conversion(frame,"GRAY") //deep network needs the color image. If we feed grayscale image, there will be error in code.
        # search for frame to detect object- processing the first frames of input source 
        if search_for_frame_to_detect_object:
            detected_boxes, frame = detection.Detection_BoundingBox(frame)
            
            if len(detected_boxes)>0:            
                search_for_frame_to_detect_object = False
                #multi tracker
                if tracker == []:
                    tracker = C_TRACKER(TRACKER_TYPE)
                    tracker.Add_Tracker(frame, detected_boxes)
                    # frame_number = 0
                    continue
                else:
                    tracker.update_pipeline(frame, detected_boxes)
            
            # Updating tracker when object detectin looking for objects in frame if object detection method fails to detect. In this situation, tracker keeps on his task.
            # We must not leave a frame without tracking
            if tracker != []:
                frame = tracker.update(frame)                

            cv2.imshow("output",frame)
            video.write_output_video(frame)
            cv2.waitKey(1)
            continue
        else:

            # t1 = time.time()
            # Detection
            # detected_boxes,frame = detection.Detection_BoundingBox(frame)
            frame = tracker.update(frame)          
           
            # Output
            cv2.imshow("output", frame)
            video.write_output_video(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break #esc to quit
            

    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    # t1=time.time()
    main()
    # t2=time.time()
    # print('pipeline2 time:', t2-t1)