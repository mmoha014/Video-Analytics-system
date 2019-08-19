from video_capture import C_VIDEO_UNIT
# from multitracking import box_iou2, correct_position
from preprocessing import C_PREPROCESSING
from Object_detection import C_DETECTION
from tracking import C_TRACKER
from utils import *
from MOT_File_Generator import C_MOT_OUTPUT_GENERATER # version 2
from setting import *
import cv2


            
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

    groundtruth_box  = C_PREPROCESSING.MOT_read_groundtruth_file(FILE_ADDRESS_DEEP_GROUNDTHRUTH)
    detection = C_DETECTION(DETECTION_METHOD)
    video = C_VIDEO_UNIT(INPUT_VIDEO_SNOURCE)    
    MOT = C_MOT_OUTPUT_GENERATER(FOLDER+'.txt')# version 2 

    tracker = []

    acc_per_frame = []
    # ret, frame = video.get_frame()
    frame_size = None#frame.shape
    frame_number = -1
    time_sum = 0

    search_for_frame_to_detect_object = True

    while True:
        ret, frame = video.get_frame()
        if not ret:
            break

        frame_number += 1
    
        if frame_size == None:
            frame_size = frame.shape

        if frame_number % UPDATE_TRACKER == 0:
            search_for_frame_to_detect_object = True            
       
        if search_for_frame_to_detect_object:
            detected_boxes, frame = detection.Detection_BoundingBox(frame)
            
            if len(detected_boxes)>0:            
                search_for_frame_to_detect_object = False
                #multi tracker
                if tracker == []:
                    tracker = C_TRACKER(TRACKER_TYPE)
                    tracker.Add_Tracker(frame, detected_boxes)
                    # MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2
                    frame = C_PREPROCESSING.MOT_gt_show(frame, groundtruth_box[frame_number])

                    # frame_number = 0
                    continue
                else:
                    tracker.update_pipeline(frame, detected_boxes)
                    # version 1
                    # MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2
                    frame = C_PREPROCESSING.MOT_gt_show(frame, groundtruth_box[frame_number])
                    continue

            
            if tracker != []:
                # search_for_frame_to_detect_object, frame = tracker.update2(frame, frame_size)
                frame = tracker.update(frame)#, frame_size)
                # MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2
                frame = C_PREPROCESSING.MOT_gt_show(frame, groundtruth_box[frame_number])

            cv2.imshow("output",frame)
            cv2.waitKey(1)
            continue
        else:

            frame = tracker.update(frame)
            # MOT.write(frame_number, tracker.Get_MOTChallenge_Format()) #version 2
            frame = C_PREPROCESSING.MOT_gt_show(frame, groundtruth_box[frame_number])
            # Output            
            cv2.imshow("output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break #esc to quit

    # MOT.close() # version 2
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
