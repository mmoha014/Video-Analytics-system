from video_capture import C_VIDEO_UNIT
# from multitracking import box_iou2, correct_position
from preprocessing import C_PREPROCESSING
from Object_detection import C_DETECTION
from tracking import C_TRACKER
from utils import *
from MOT_File_Generator import C_MOT_OUTPUT_GENERATER # version 2
from setting import *
import cv2
from sys import stdout
import time
# from cpu_memory_track import monitor            
# def write_MOT_OUTPUT(boxes_ids):
    


# from mem_track import track


# pipeline_start_time = time.time()

def max_accuracy(predicted_box, gt):
            max_idx, max_acc = 0, 0
            for i, bb in enumerate(predicted_box):
                accuracy = box_iou2(correct_position(bb), correct_position(gt))
                if max_acc<accuracy:
                    max_acc=accuracy
                    max_idx =  i
            return max_acc

# @track
def main():
    #>>>>>>>>>>>>>>>>>>>>>>>>>> begin <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #>>>>>>>>>>>>>>>>>>>>>>>>>> version 2 - Adapting the source code to compare with DevKit of MOTChallenge  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    groundtruth_box  = C_PREPROCESSING.MOT2015_read_groundtruth_file(FILE_ADDRESS_DEEP_GROUNDTHRUTH)
    detection = C_DETECTION(DETECTION_METHOD)
    video = C_VIDEO_UNIT(INPUT_VIDEO_SNOURCE)    
    MOT = C_MOT_OUTPUT_GENERATER('./logs/'+FOLDER+'.txt')# version 2 

    tracker = None

    acc_per_frame = []
    # ret, frame = video.get_frame()
    frame_size = None#frame.shape
    frame_number = -1
    time_sum = 0
    detection_time = []
    tracker_time = []
    
    search_for_frame_to_detect_object = True

    while True:        
        ret, frame = video.get_frame()
        if not ret or frame_number>13:
            break

        frame_number += 1
        stdout.write('\r%d'% frame_number)
        if frame_size == None:
            frame_size = frame.shape

        if frame_number % UPDATE_TRACKER == 0 or TRACKER_TYPE is "Kalman_Filter":
            search_for_frame_to_detect_object = True            

        # if frame_number % 35 == 0 and frame_number>0:            
        #     print("change tracker to MEDIANFLOW")
        #     tracker.switch_Tracker(frame, "MEDIANFLOW")
        
        # if frame_number % 80 == 0 and frame_number>0:            
        #     print("change tracker to MIL")
        #     tracker.switch_Tracker(frame, "MIL")
        
        # if frame_number % 120 == 0 and frame_number>0:    
        #     print("change tracker to KCF")
        #     tracker.switch_Tracker(frame, "KCF")
        
            
        # gt = groundtruth_box[frame_number-1]
        # Preprocessing    
        # frame  = C_PREPROCESSING.resize(frame, min(400, frame.shape[1]))
        # frame = C_PREPROCESSING.Color_Conversion(frame,"GRAY") //deep network needs the color image. If we feed grayscale image, there will be error in code.
        # search for frame to detect object- processing the first frames of input source 
        if search_for_frame_to_detect_object:
            t1 = time.time()
            detected_boxes, frame = detection.Detection_BoundingBox(frame)
            t2 = time.time()
            detection_time.append(t2-t1)

            if TRACKER_TYPE is "Kalman_Filter":
                if tracker is None:
                    tracker = C_TRACKER(TRACKER_TYPE)
                trackers, colors = tracker.update_pipeline(frame, detected_boxes)
                frame = KalmanFilter_draw_box(frame, trackers, colors)
                cv2.imshow("output", frame)
                cv2.waitKey(1)
                MOT.write(frame_number, tracker.Get_MOTChallenge_Format())
                continue
            
            if len(detected_boxes)>0:            
                search_for_frame_to_detect_object = False
                #multi tracker
                if tracker is None:
                    tracker = C_TRACKER(TRACKER_TYPE)
                    tracker.Add_Tracker(frame, detected_boxes)
                    MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2
                    # frame_number = 0
                    continue
                else:
                    t1= time.time()
                    tracker.update_pipeline(frame, detected_boxes)
                    t2 = time.time()
                    tracker_time.append(t2-t1)
                    # version 1
                    MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2
                    continue

            
            if tracker is not None:
                # search_for_frame_to_detect_object, frame = tracker.update2(frame, frame_size)
                t1 = time.time()
                frame = tracker.update(frame)#, frame_size)
                t2 = time.time()
                tracker_time.append(t2-t1)
                MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2

            cv2.imshow("output",frame)
            cv2.waitKey(1)
            continue
        else:
            t1 = time.time()
            frame = tracker.update(frame)
            t2 = time.time()
            tracker_time.append(t2-t1)
            MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2
            # print(tracker.Get_MOTChallenge_Format())
            # print(groundtruth_box)
            # Output            
            try:
                cv2.imshow("output", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break #esc to quit
            except:
                print("error in reading image and showing it")

    MOT.close() # version 2
    cv2.destroyAllWindows()
    # print("detection time: ", np.average(detection_time))
    # print("tracking time: ", np.average(tracker_time))
        
if __name__ == "__main__":
    # main()
    from cpu_memory_track import monitor
    cpu,mem = monitor(main)
