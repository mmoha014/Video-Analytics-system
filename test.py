import cv2
import pylab as pl

for i in range(20):
    img=cv2.imread('/home/mgharasu/Pictures/'+str(i+1)+'.png')
    pl.imshow(img)
    pl.pause(.1)
    pl.draw()


# from preprocessing import C_PREPROCESSING
# from tracking import C_TRACKER
# from  setting import *
# from utils import *
# from mylinear_assignment import linear_assignment
# from profiling_Pipeline import F1, targetSize, frame_size, detection, MOT, video,groundtruth_box
# from sys import stdout

# def F1_score(predicted_box, groundtruth, frame_number):   #version 3 
#     # global label_track_table
#     num_p = len(predicted_box)    
#     # gt = C_PREPROCESSING.MOT_read_frame_objects_from_groundtruth(groundtruth,frame_number)    
#     num_gt = len(groundtruth)
#     # compute number of True positives - find the bbox with max overlap (LAP) and label
#     # this section can be copied from tracker label assignment (Exactly the same method)
#     IOU_mat = np.zeros((num_gt, num_p), dtype=np.float32)
        
#     for i,p in enumerate(groundtruth):
#         for j,g in enumerate(predicted_box):
#             IOU_mat[i][j] = box_iou2(g,p)

#     matched_idx = linear_assignment(-IOU_mat)

#     P = np.divide(len(matched_idx), len(groundtruth), dtype=np.float)
#     R = np.divide(len(matched_idx), len(predicted_box), dtype= np.float)
#     # num_TP = 0
#     # for midx in matched_idx:
#     #     g_t = midx[0]
#     #     prdct = midx[1]
#     #     if len(label_track_table) < g_t:
#     #         if label_track_table[g_t] == prdct:
#     #             num_TP += 1
#     #         else:
#     #             label_track_table[g_t] = prdct
#     #     else: #------- assumption is that labels are in ascending order
#     #         label_track_table.append(prdct)
    
#     # num_of_objs = len(groundtruth)
#     # num_of_detections = len(predicted_box)
#     # P = np.divide(num_TP,num_of_objs, dtype=np.float)
#     # R = np.divide(num_TP,num_of_detections, dtype=np.float)
#     F1 = np.divide(2*P*R, P+R, dtype=np.float)
#     return F1

# def Set_configuration(C):
#     global targetSize, frame_size
#     #C [i,  j,   k]
#     # size, rate, model 
    
#     targetSize = C[0]#knobs['frame_size'][C[0]]
#     frame_rate = C[1]#knobs['frame_rate'][C[1]]    
#     model = C[2]#knobs['detection_model'][C[2]]
#     # tracker.switch_Tracker()    
#     detection.Switch_detection(model)

# def _Results(frame_number, bboxes, gt):
#         # bboxes = tracker.Get_MOTChallenge_Format()
#         MOT.write(frame_number, bboxes) #version 2
#         # place for calling F1_Score function
#         # gt = C_PREPROCESSING.MOT_read_frame_bboxes_from_groundtruth(groundtruth_box, start_frame) #version 3
#         F1.append(F1_score([bb[1:] for bb in bboxes], gt, frame_number)) #version 3 
#         # return start_frame +1

# def Process_segment(segment, config, isprofiling):

#     Set_configuration(config)
#     # global start_frame, frame_size, tracker, video, detection, groundtruth_box, search_for_frame_to_detect_object, MOT
#     # switching configs is done here before starting to 
#     start_frame, end_frame = segment
    
#     gt = None

#     if isprofiling:
#         tracker = None

#     while start_frame <= end_frame:
#         ret, frame = video.get_frame_position(start_frame)
#         if not ret:
#             break
#         # C_PREPROCESSING.MOT_gt_show(frame,groundtruth_box[start_frame])
#         stdout.write('\r%d'% start_frame)
#         # start_frame += 1
#         if isprofiling:
#             gt = groundtruth_box[start_frame]
#             # preprocessing
#             frame, gt = C_PREPROCESSING.resize(frame,targetSize,gt)
#             # cv2.rectangle(frame, (gt[0][0],gt[0][1]), (gt[0][2]+gt[0][0],gt[0][1]+gt[0][3]),(255,0,0),1)
#             # cv2.imshow("bbox", frame)
#             # cv2.waitKey(0)
#         else:
#             resize_image(frame, targetSize)
                
        
                
#         frame_size =  frame.shape
        
#         if start_frame % UPDATE_TRACKER == 0 or start_frame == segment[0]:  # each segment is determined here
#             search_for_frame_to_detect_object = True
        
#         start_frame = start_frame + 1

#         if search_for_frame_to_detect_object:
#             detected_boxes, frame = detection.Detection_BoundingBox(frame)
        
#             if len(detected_boxes) > 0:
#                 search_for_frame_to_detect_object = False
#                 if tracker is None:
#                     tracker = C_TRACKER(TRACKER_TYPE)
#                     tracker.Add_Tracker(frame, detected_boxes)
                    
#                     bboxes = tracker.Get_MOTChallenge_Format()
#                     _Results(start_frame, bboxes, gt)
#                     # MOT.write(start_frame, tracker.Get_MOTChallenge_Format()) #version 2
#                     # frame_number = 0                    
#                     continue
#                 else:
#                     tracker.update_pipeline(frame,detected_boxes)
#                     #------------- profiling -----------
#                     if isprofiling:
#                         bboxes = tracker.Get_MOTChallenge_Format()                        
#                         _Results(start_frame,bboxes, gt)                        
#                         # start_frame += 1

#                     continue
            
#             # if there is no detected boxes by object detection, the tracker keeps on tracking previous objects
#             if tracker is not None:
#                 frame = tracker.update(frame)
#                 #------------- profiling -----------
#                 if isprofiling:
#                     bboxes = tracker.Get_MOTChallenge_Format()
#                     _Results(start_frame,bboxes, gt)                        
#                     # start_frame += 1

#             cv2.imshow("output", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#             continue
#         else:
#             # in times when object detection is not used
#             frame = tracker.update(frame)
#             #------------- profiling -----------
#             if isprofiling:
#                 bboxes = tracker.Get_MOTChallenge_Format()
#                 _Results(start_frame,bboxes, gt)                
#                 # start_frame += 1

#             cv2.imshow("output", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
        
#         # start_frame += 1


#     cv2.destroyAllWindows()
    
#     if start_frame<=end_frame:
#         cont = 0 # process of segment is not completed because of finishing the frames in video
#     start_frame = start_frame - 1