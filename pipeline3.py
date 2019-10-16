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
import copy
import matplotlib.pyplot as plt
def main():
    #>>>>>>>>>>>>>>>>>>>>>> begin <<<<<<<<<<<<<<<<<<<<<<<<
    detection = C_DETECTION(DETECTION_METHOD)
    video = C_VIDEO_UNIT(INPUT_VIDEO_SNOURCE)

    tracker = []
    tracker_mask = []
    tracker_shapes = []
    
    frame_size = None
    frame_number = -1
    time_sum = 0

    while True:
        ret, frame = video.get_frame()
        if not ret:
            break
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)
        frame_number += 1
        
        stdout.write('\r%d'% frame_number)        
        
        detected_boxes, output_frame = detection.Detection_BoundingBox(frame)
        
        masks = np.zeros((len(tracker), len(detected_boxes)), dtype=np.int)
        if frame_size == None:
            rois = []
            frame_size = frame.shape
            for i, box in enumerate(detected_boxes):
                x,y,w,h = box
                tracker.append(cv2.createBackgroundSubtractorMOG2())
                roi = frame[y:y+h,x:x+w]
                tracker_shapes.append((w,h))
                # tracker_mask.append(tracker[i].apply(roi))                
                rois.append(roi)
        else:            
            for j, tr in enumerate(tracker):
                # cv2.imshow("roi in previous frame", rois[j])
                f = plt.figure(figsize=(4,4))
                f.add_subplot(4,4, 1)
                plt.imshow(rois[j])
                for i, box in enumerate(detected_boxes):
                    x,y,w,h = box
                    roi  = frame[y:y+h, x:x+w]
                    resized_roi = cdv2.resize(roi, tracker_shapes[j])
                    trkr = copy.deepcopy(tracker[j])
                    mask = trkr.apply(resized_roi)
                    masks[j][i] = np.sum(mask)
                    # showmessage = "roi " + str(i) + " in previous frame"                    
                    # cv2.imshow(showmessage, resized_roi)
                    f.add_subplot(4,4,2+i)
                    plt.imshow(resized_roi)
                
                min_idx = np.argmin(masks[j])                
                box = detected_boxes[min_idx]
                roi = frame[y:y+h, x:x+w]
                resized_roi = cv2.resize(roi, tracker_shapes[min_idx])
                # showmessage = "roi " + str(min_idx) + " in previous frame"                
                # cv2.imshow(showmessage, resized_roi)
                f.add_subplot(4,4, i+1)
                plt.imshow(resized_roi)
                plt.show()
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()



            # for i, box in enumerate(detected_boxes):
            #     x,y,w,h = box
            #     roi = frame[y:y+h, x:x+w]
                
            #     for j, tr in enumerate(tracker):
            #         # trcopy = copy.deepcopy(tr)
            #         resized_roi = cv2.resize(roi, tracker_shapes[j])
            #         mask = tr.apply(roi)
            #         masks[i][j] = np.sum(mask)
            #         showmessage= "roi "+str(j)+ " in previous frame"
            #         cv2.imshow(showmessage, resized_roi)
            #         cv2.imshow("roi in current frame", resized_roi)
            #     cv2.waitKey(0)
                
        if tracker is None:
            tracker = C_TRACKER(TRACKER_TYPE)

if __name__ == "__main__":
    main()