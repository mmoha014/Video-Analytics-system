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
from mylinear_assignment import linear_assignment
from psutil import virtual_memory
# from Segment_process import Process_segment
from cpu_memory_track import monitor
import pylab as pl

video = C_VIDEO_UNIT(INPUT_VIDEO_SNOURCE, frame_rate=14)
groundtruth_box  = C_PREPROCESSING.MOT2017_read_groudntruth_file(FILE_ADDRESS_DEEP_GROUNDTHRUTH, video.get_number_of_frames())
detection = C_DETECTION(DETECTION_METHOD)
MOT = C_MOT_OUTPUT_GENERATER('./logs/'+FOLDER+'.txt')# version 2 



tracker = None
# label_track_table = []                                                #version 3
# top_k_config = []
# frame_size = None
# frame_number = -1
# time_sum = 0
start_frame = 0
search_for_frame_to_detect_object = True
targetSize = 1.0
#===================================version 3 ===================================================
total_memory = np.divide(virtual_memory().total, np.power(1024,2), dtype=np.float) #version 3
top_k = 5                                                               #version 3
frame_rate = 30   #from config file                                    #version 3
F1 = []                                                                 #version 3
knobs = {}                                                              #version 3
knobs['frame_size'] = [960,840,720,600,480]                             #version 3
knobs['frame_rate'] = [30,10,5,2,1]                                     #version 3
knobs['detection_model']=['Deep_Yolo','HOG_Pedestrian'] #version 3#'Deep_Yolo_Tiny',
configs = copy.deepcopy(knobs)

knobs_cpu_mem = [[[28.113,668.25],[28.95, 652.15],[29.7, 640.35],[30.8,639.93],[31.7, 637.67]],[[28.113,668.25],[28.23, 614.3],[25.21, 567.297],[25.895, 520.37],[27.042, 492.928]],[[28.113,668.25],[45.05, 306.914]]]
""" C is configuration to set in system """
        
        
def profiling(segment_number,S,C,K):  #version 3
    global knobs
    # def compute_F1_memory_cpu_temp(C):
        # cpu_percent, mem_usage = monitor(Process_segment(S, C, True)) #main_line
        # Process_segment(S, C, True) # temporary line
        # cpu = np.average(cpu_percent)                                                                                       main line
        # memory = np.divide(np.divide(np.average(mem_usage),np.power(1024,2), dtype=np.float), total_memory, dtype=np.float) main line
        
        # Both memory and cpu are percentage, i.e., a value in 
        # return cpu, memory
    def compute_F1_memory_cpu(C2):
        # ps = SEGMENT()
        cpu_percent, mem_usage = monitor(Process_segment, (segment_number, S, C2, True,)) #, detection, video, groundtruth_box, targetSize,)) #main_line        
        
        # cpu = np.divide(np.average(cpu_percent), 100, dtype=np.float)                                                                                # main line
        cpu = np.divide(np.sum(cpu_percent), 100, dtype=np.float)
        memory = np.sum(np.divide(np.divide(mem_usage,np.power(1024,2), dtype=np.float), total_memory, dtype=np.float))# main line
        print('\ncpu: ',cpu, 'memory:' ,memory)
        
        # Both memory and cpu are percentage, i.e., a value in 
        return cpu/frame_rate, memory/frame_rate

    # The below line  is for the time when we want to use golde configuration as the groundtruth, So in the current assumption 
    # it is not needed to be golden because the ground truth is obtained from dataset not from golden model.
    # But this is considered as a unique configuration because in other combinations in nested for 2 lines later it will be repeated for each knob.
    # Also it is mentioned in the chameleon article  that the knobs values are independent of each other, so we can use them seperately to see their performance
    golden_config = [C['frame_size'][0], C['frame_rate'][0], C['detection_model'][0]] 
    # golden_cpu, golden_mem = compute_F1_memory_cpu(golden_config) main line
    compute_F1_memory_cpu(golden_config)
    golden_cpu, golden_mem = knobs_cpu_mem[0][0]
    golden_cpu = golden_cpu/100
    golden_mem = np.divide(np.divide(np.average(golden_mem*1024*1024),np.power(1024,2), dtype=np.float), total_memory, dtype=np.float)
    # accuracy=max, cpu_usage=min and memory_usage=min. To create a minimum object function: accuracy = 1- accuracy (min). object function is finding minimum
    golden_F1 = np.average(F1)
    
    KnobValueToScore = [[] for j in range(len(C))]
    for i,knob in enumerate(C): 
        isnew_knob = True       
        for j,value in enumerate(C[knob][1:]):
            C_Vr = copy.deepcopy(golden_config)
            C_Vr[i] = value
            # here memory and cpu usage must be calculated            
            cpu, memory = compute_F1_memory_cpu(C_Vr) #Main line
            #========= temporary section ============================
            # compute_F1_memory_cpu_temp(C_Vr) # temporary line
            # cpu, memory1 = knobs_cpu_mem[i][j+1] # temporary line
            # cpu = cpu / 100 
            # # memory = np.divide(np.divide(np.average(memory1*1024*1024),np.power(1024,2), dtype=np.float), total_memory, dtype=np.float) # temporary line
            # memory = np.divide(memory1*1024*1024, total_memory, dtype=np.float)
            #========================================================

            # if isnew_knob:
            #     KnobValueToScore[i].append([np.average(F1), cpu, memory])
            
            KnobValueToScore[i].append([np.average(F1), cpu, memory])# KnobValueToScore[i].append(F1)
            # print([np.average(F1), cpu, memory], "\n")            
    # Currently, different configurations are evaluated and their results are stored as an 2D array.
    # Removing the configurations whose F1 Scores are less than 0.8 (Alpha)
    alpha = 0.49    
    for i in range(np.shape(KnobValueToScore)[0]):
        for j in range(len(KnobValueToScore[i])):
            if KnobValueToScore[i][j][0] < alpha:
                KnobValueToScore[i][j][0] = 0 # penalize the accuracy to get out of selection in minimal solutions (configurations in knobvaluetoscore)
    
    AccurateConfigs = 1000*np.ones((len(knobs['frame_size']),len(knobs['frame_rate']),len(knobs['detection_model'])), dtype=np.float)
    # AccuracteConfig_MEMCPU = np.ones((len(knobs['frame_size']),len(knobs['frame_rate']),len(knobs['detection_model'])), dtype=np.float)
    AC_shape = np.shape(AccurateConfigs)
    
    # Now it is the time for determining alpha_prime and remove configs based on their scores. 
    alpha_prime = 0.1 #considering the multiplication of all values bigger than or equal to 0.8, i.e., the minimal is 0.8*0.8*0.8= 0.512
                        # 0.8*0.79*0.81 = 0.51192,   0.8*0.78*0.85 = 0.5304,  

    for i in range(AC_shape[0]-1):        
        if i == 0:
            acci = golden_F1
            memi = golden_mem
            cpui = golden_cpu
        else:
            acci = KnobValueToScore[0][i-1][0]
            memi = KnobValueToScore[0][i-1][2]
            cpui = KnobValueToScore[0][i-1][1]
        
        for j in range(AC_shape[1]):
            if j == 0:
                accj = golden_F1
                memj = golden_mem
                cpuj = golden_cpu
            else:
                accj = KnobValueToScore[1][j-1][0]
                memj = KnobValueToScore[1][j-1][2]
                cpuj = KnobValueToScore[1][j-1][1]
            
            for k in range(AC_shape[2]): 
                if k == 0:
                    acck = golden_F1
                    memk = golden_mem
                    cpuk = golden_cpu
                else:
                    acck = KnobValueToScore[2][k-1][0]
                    memk = KnobValueToScore[2][k-1][2]
                    cpuk = KnobValueToScore[2][k-1][1]
            
                    
                temp_value = acci * accj * acck

                if temp_value>=alpha_prime:
                    cpu = np.min([cpui, cpuj, cpuk])
                    memory = np.min([memi,memj,memk])
                    # AccuracteConfig_MEMCPU[i,j,k] = [temp_value,cpu,memory]
                    AccurateConfigs[i,j,k] = (1-temp_value)+cpu+memory # normalization is very important
                # temp_value = KnobValueToScore[0][i][0] * KnobValueToScore[1][j][0] * (golden_F1 if k==0 else KnobValueToScore[2][0][0])#KnobValueToScore[2][k][0]
                # if temp_value>=alpha_prime:
                #     cpu = np.min([KnobValueToScore[0][i][1] , KnobValueToScore[1][j][1] , (golden_cpu if k==0 else KnobValueToScore[2][0][1])])
                #     memory = np.min([KnobValueToScore[0][i][2] , KnobValueToScore[1][j][2] , (golden_mem if k==0 else KnobValueToScore[2][0][2])])
                #     AccurateConfigs[i][j][k] = (1-temp_value) + cpu + memory # The goal is reducing memory and cpu, and increasing accuracy. In order to change the goal in accuracy as reduction target, it is subtracted by 1
                    
    # doing normalization on AccurateConfigs    
    top_k_config, scores = k_min_cell_index_3D_matrix(AccurateConfigs,top_k)    
    # print('==================================top_k configs=================================')
    # print(AccurateConfigs)
    # print('==================================top_k configs=================================')
    return top_k_config, scores




def F1_score(predicted_box, groundtruth, frame_number):   #version 3 
    # global label_track_table
    num_p = len(predicted_box)    
    # gt = C_PREPROCESSING.MOT_read_frame_objects_from_groundtruth(groundtruth,frame_number)    
    num_gt = len(groundtruth)
    # compute number of True positives - find the bbox with max overlap (LAP) and label
    # this section can be copied from tracker label assignment (Exactly the same method)
    IOU_mat = np.zeros((num_gt, num_p), dtype=np.float32)
        
    for i,p in enumerate(groundtruth):
        for j,g in enumerate(predicted_box):
            IOU_mat[i][j] = box_iou2(g,p)

    matched_idx = linear_assignment(-IOU_mat)

    P = np.divide(len(matched_idx), len(groundtruth), dtype=np.float)
    R = np.divide(len(matched_idx), len(predicted_box), dtype= np.float)
    # num_TP = 0
    # for midx in matched_idx:
    #     g_t = midx[0]
    #     prdct = midx[1]
    #     if len(label_track_table) < g_t:
    #         if label_track_table[g_t] == prdct:
    #             num_TP += 1
    #         else:
    #             label_track_table[g_t] = prdct
    #     else: #------- assumption is that labels are in ascending order
    #         label_track_table.append(prdct)
    
    # num_of_objs = len(groundtruth)
    # num_of_detections = len(predicted_box)
    # P = np.divide(num_TP,num_of_objs, dtype=np.float)
    # R = np.divide(num_TP,num_of_detections, dtype=np.float)
    F1 = np.divide(2*P*R, P+R, dtype=np.float)
    return F1

# def Set_configuration(C):
#     global targetSize#, frame_size
#     #C [i,  j,   k]
#     # size, rate, model 
    
#     targetSize = int(knobs['frame_size'][C[0]])
#     frame_rate = int(knobs['frame_rate'][C[1]])
#     model = knobs['detection_model'][C[2]]    
#     # tracker.switch_Tracker()    
#     detection.Switch_detection(model)
#     return str(targetSize)+'_'+str(frame_rate)+'_'+model+'.avi'

def Set_configuration(C):
    global targetSize, detection
    #C [i,  j,   k]
    # size, rate, model 
    
    targetSize = C[0]#knobs['frame_size'][C[0]]
    frame_rate = C[1]#knobs['frame_rate'][C[1]]    
    model = C[2]#knobs['detection_model'][C[2]]
    # tracker.switch_Tracker()        
    detection.Switch_detection(model)
    

def _Results(frame_number, bboxes, gt):
        global  MOT
        # bboxes = tracker.Get_MOTChallenge_Format()
        MOT.write(frame_number, bboxes) #version 2
        # place for calling F1_Score function
        # gt = C_PREPROCESSING.MOT_read_frame_bboxes_from_groundtruth(groundtruth_box, start_frame) #version 3
        F1.append(F1_score([bb[1:] for bb in bboxes], gt, frame_number)) #version 3 
        # return start_frame +1

def Process_segment(segment_number, segment, config, isprofiling):     
    print(segment)
    print(config)
    global targetSize, frame_rate, groundtruth_box,tracker, detection, segme
    Set_configuration(config)         
    targetSize = int(targetSize)

    filename = 'seg'+str(segment_number)+'_'+str(config[0])+'_'+str(config[1])+'_'+config[2]+'.avi'
    out = cv2.VideoWriter('/home/mgharasu/Documents/Purified codes/outputs/'+filename, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (targetSize,targetSize))
    
    # out = cv2.VideoWriter('output.avi', -1, frame_rate, (targetSize,targetSize))
    # global start_frame, frame_size, tracker, video, detection, groundtruth_box, search_for_frame_to_detect_object, MOT
    # switching configs is done here before starting to 
    start_frame, end_frame = segment
    
    gt = None

    if isprofiling:
        tracker = None
    tmp_counter = 0
    after_n__frames_read = np.int(np.divide(end_frame-start_frame+1, config[1]))
    # print(after_n__frames_read)
    # number_of_frames_to_skip = np.int((end_frame-start_frame+1)/ after_n__frames_read)
    # print("tmp_frames_rate: ", after_n__frames_read)
    # print("number_of_frames_to_skip", number_of_frames_to_skip)
    # while start_frame <= end_frame:
    while start_frame <= end_frame:
        ret, frame = video.get_frame_position(start_frame)
        if not ret:
            break
        # C_PREPROCESSING.MOT_gt_show(frame,groundtruth_box[start_frame])
        tmp_counter = tmp_counter + 1
        stdout.write('\r%d'% tmp_counter)
        # start_frame += 1
        if isprofiling:
            gt = groundtruth_box[start_frame]
            # preprocessing
            frame, gt = C_PREPROCESSING.resize(frame,targetSize,gt)
            # cv2.rectangle(frame, (gt[0][0],gt[0][1]), (gt[0][2]+gt[0][0],gt[0][1]+gt[0][3]),(255,0,0),1)
            # cv2.imshow("bbox", frame)
            # cv2.waitKey(0)
        else:
            resize_image(frame, targetSize)
                
        
                
        # frame_size =  frame.shape
        
        if start_frame % UPDATE_TRACKER == 0 or start_frame == segment[0]:  # each segment is determined here
            search_for_frame_to_detect_object = True
        
        start_frame = start_frame + after_n__frames_read

        if search_for_frame_to_detect_object:
            detected_boxes, frame = detection.Detection_BoundingBox(frame)
        
            if len(detected_boxes) > 0:
                search_for_frame_to_detect_object = False
                if tracker is None:
                    tracker = C_TRACKER(TRACKER_TYPE)
                    tracker.Add_Tracker(frame, detected_boxes)
                    
                    bboxes = tracker.Get_MOTChallenge_Format()
                    _Results(start_frame, bboxes, gt)
                    out.write(frame)
                    # MOT.write(start_frame, tracker.Get_MOTChallenge_Format()) #version 2
                    # frame_number = 0                    
                    continue
                else:
                    tracker.update_pipeline(frame,detected_boxes)
                    #------------- profiling -----------
                    if isprofiling:
                        bboxes = tracker.Get_MOTChallenge_Format()                        
                        _Results(start_frame,bboxes, gt)                        
                        # start_frame += 1
                        out.write(frame)

                    continue
            
            # if there is no detected boxes by object detection, the tracker keeps on tracking previous objects
            if tracker is not None:
                frame = tracker.update(frame)
                #------------- profiling -----------
                if isprofiling:
                    bboxes = tracker.Get_MOTChallenge_Format()
                    _Results(start_frame,bboxes, gt)                        
                    # start_frame += 1
                    out.write(frame)
            
            # cv2.imshow("output", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            continue
        else:
            # in times when object detection is not used
            frame = tracker.update(frame)
            #------------- profiling -----------
            if isprofiling:
                bboxes = tracker.Get_MOTChallenge_Format()
                _Results(start_frame,bboxes, gt)
                # start_frame += 1
                out.write(frame)

            out.write(frame)
            # cv2.imshow("output", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        # start_frame += 1


    # cv2.destroyAllWindows()
    out.release()
    if start_frame<=end_frame:
        cont = 0 # process of segment is not completed because of finishing the frames in video
    start_frame = start_frame - 1

def Sub_knob_creator(config):
    new_knob = {}
    new_knob['frame_size']=[]
    new_knob['frame_rate']=[]
    new_knob['detection_model']=[]

    for c in config:
        new_knob['frame_size'].append(knobs['frame_size'][c[0]])
        new_knob['frame_rate'].append(knobs['frame_rate'][c[1]])
        new_knob['detection_model'].append(knobs['detection_model'][c[2]])
    
    return new_knob

def pipeline():
    '''
    Here division of video is done and 
    choosing configuration (profiling) is done
    to process segments of video    
    '''
    global configs, frame_rate
    #each segement is 1 second or 14 frames because FPS = 14 in this video
    W = 5 # window includes 5 segments
    T = 4 #each segment 4 seconds
    t = 1 #seconds
    top_k_config = []
    segment_number = 0    
    cont = True
    start_frame = 0
    while cont:
        # segment = [start_frame, start_frame+frame_rate*T-1] #each segment is T seconds and each second is FPS=14  
        if segment_number % W == 0:
            # profiling            
            # print("\n profiling in whole space \n")
            # # sub_segment = [start_frame*(t-1), frame_rate*t - 1]
            # t = t + 1
            # top_k_config, scores = profiling(segment_number,sub_segment,knobs,5)
            end_frame = start_frame+frame_rate-1
            top_k_config, scores = profiling(segment_number,[start_frame, end_frame],knobs,5)
            sub_knobs = Sub_knob_creator(top_k_config)
            cont=False

            # sub_segment = [start_frame+1, frame_rate*t - 1]
            # t = t + 1
            # config = profiling(sub_segment, top_k_config,1)            
            
            # golden_config_in_sub_knobs = [sub_knobs['frame_size'][0], sub_knobs['frame_rate'][0], sub_knobs['detection_model'][0]] 
            # cont = Process_segment(sub_segment, golden_config_in_sub_knobs,False)

            # extract video for profiling
            #choosing a configuration
            # return 0
        else:
            # print("\n profiling in top-k \n")
            # process segment with profiling on top-k configs
            
            # sub_segment= [start_frame+1, frame_rate*t-1] # the first t second(s) of the segment
            # t = t + 1
            # golden_config_in_sub_knobs = [sub_knobs['frame_size'][0], sub_knobs['frame_rate'][0], sub_knobs['detection_model'][0]] 
            # cont = Process_segment(sub_segment, golden_config_in_sub_knobs,False)
            
            # config, scores = profiling(segment_number, sub_segment, sub_knobs, 1)
            
            end_frame = start_frame+frame_rate-1
            config, scores = profiling(segment_number, [start_frame, end_frame], sub_knobs, 1)
            
            # sub_segment = [start_frame, start_frame*(T-t)-1] # the remaining T-t seconds of the segment
            # cont = Process_segment(sub_segment, config,False)
        start_frame = start_frame+end_frame+1
        segment_number += 1

    MOT.close()
    return 0

if __name__ == "__main__":
    # from pipeline2 import main
    # cpu,mem = monitor(main)
    pipeline()