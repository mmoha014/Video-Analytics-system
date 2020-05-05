# This source code is used to generate and then process the segments in video analytics system. This source code 
# has solved some mistakes in computing objective function> Also, it is used to watch effect of features like luminance, object sizes and
# motion patterns to detect their effect in profiling. This file is in close usage with files in the other project whose name is opencv implementation.
# this file will be put beside the files in that project
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
import sys
from cpu_memory_track import monitor
import timeit
import pylab as pl
import multiprocessing as mp
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from multiprocessing.managers import BaseManager
from multiprocessing import Manager, Pool
from functools import partial
import psutil
import os
from memory_profiler import profile
from scipy.spatial import distance as dist

video = C_VIDEO_UNIT(INPUT_VIDEO_SNOURCE, frame_rate=14)
groundtruth_box  = C_PREPROCESSING.MOT2015_read_groundtruth_file(FILE_ADDRESS_DEEP_GROUNDTHRUTH)#
# groundtruth_box = C_PREPROCESSING.MOT2017_read_groudntruth_file(FILE_ADDRESS_DEEP_GROUNDTHRUTH, video.get_number_of_frames())
detection = C_DETECTION(DETECTION_METHOD)
MOT = C_MOT_OUTPUT_GENERATER('./logs/'+FOLDER+'.txt')# version 2 
cont_read_frame = False


tracker = None
# label_track_table = []                                                #version 3
# top_k_config = []
# frame_size = None
# frame_number = -1
# time_sum = 0
trckr_type = None
start_frame = 0
search_for_frame_to_detect_object = True
targetSize = 1.0
#=================================== version 3 ===================================================
total_memory = virtual_memory()[0]                                        #version 3
top_k = 5                                                               #version 3
frame_rate = 30   #from config file                                     #version 3
knobs = {}                                                              #version 3
knobs['frame_size'] = [960,840,720,600,480]                             #version 3
knobs['frame_rate'] = [30,10,5,2,1]                                     #version 3
knobs['detection_model']=['Deep_Yolo']#,'Deep_Yolo_Tiny'] #version 3#'HOG_Pedestrian',
knobs['tracker']=['CSRT','MEDIANFLOW', 'MOSSE','KCF']
configs = copy.deepcopy(knobs)
golden_F1 = golden_cpu = golden_mem = 0.0
plot_F1 =[]
plot_cpu=[]
plot_memory = []
plot_time = []
plot_num_objects = []
obj_size_avg = []
# plot_file = open('plot_info.txt','w')
# plot_file.write("30 fps, 960 resolution and YOLO model\n F1, time, mem_sum, mem_mean, cpu_sum, cpu_mean, number of objects, average_area of objects\n")

def obj_statistics(data, one_element_process):
    # print('obj_statistics', data)
    if one_element_process:
        number_objs = len(data)
        areas = [j[3]*j[4] for j in data]
        # print("obj statistics: ", areas)
    else:
        number_objs = np.sum([len(data[0][i]) for i in range(len(data[0]))])
        x=[[j[3]*j[4] for j in i] for i in data[0]]
        areas=[]
        [[areas.append(ii) for ii in k] for k in x]

    conflicts = []
    for i in range(1,number_objs):
        sum_conflicts = 0
        for j in range(i+1, number_objs):            
            sum_conflicts += box_iou2(correct_position(data[i][1:]), correct_position(data[j][1:]))
            # print("conflict between object ",i," and ", j," is ", str(data[i][1:]))
        conflicts.append(sum_conflicts)

    
    return number_objs, areas, conflicts

def velocity(new_objs, old_objs):
    # [[id0,x0,y0,w0,h0],[id1,x1,y1,w1,h1],[id2,x2,y2,w2,h2],...]            
    velocities = []
    for bbi in new_objs:
        for bbj in old_objs:
            if bbi[0]==bbj[0]:
                velocities.append(dist.euclidean(bbi[1:],bbj[1:]))
                
    return velocities

def compute_F1(segment_number, S, C2):    
    # F1 = mp.Manager().list()
    # F1 = manager.list()
    F1 = []
    detected_objs = []
    t1 = time.time()
    
    # print("tracker in computeF1:",tracker)
    detected_objs = Process_segment_frame_base(segment_number,S, C2, True,F1, detected_objs,None)
    t2 = time.time() 
    # print("compute F1 function-elapsed time:",t2-t1)  
    # end = timeit.timeit()
    # num_objs, average_area = obj_statistics(detected_objs)
    if not F1:
       F1=[1]
    
    # print(F1)
    return 0,0,F1,0,0
    # return cpu, memory, Temp_F1, np.average(cpu_percent), np.average(mem_usage)

def normalization(vect):
    minv = np.min(vect)
    maxv = np.max(vect)
    return [((x-minv)/(maxv-minv)) for x in vect]

def compute_F1_memory_cpu(segment_number,S,C2):        
        F1 = mp.Manager().list()
        detected_objs = mp.Manager().list()
        
        # shared_cont = mp.Manager().Value('d','1')
        t1 = time.time()
        cpu_percent, mem_usage = monitor(Process_segment, (segment_number, S, C2, True,F1, detected_objs, None,)) #, detection, video, groundtruth_box, targetSize,)) #main_line
        t2 = time.time()
        num_objs, mean_area_of_objs = obj_statistics(detected_objs, False)
        # x=[[j[2]*j[3] for j in i] for i in detected_objs[0]]
        # areas=[]
        # [[areas.append(ii) for ii in k] for k in x]
        # avg_obj_size = np.mean(areas)

        # print((detected_objs))
        # print("\n ", C2 ,str(t2-t1), num_objs, mean_area_of_objs)
        # cont_read_frame = shared_cont
        # print(cont_read_frame)
        # print("out of segment process: F1:",F1)
        if F1:
            Temp_F1 = np.mean(F1)
        else:
            Temp_F1 = 1
        
        # print('\n','cofig',C2,":", Temp_F1,',', np.mean(normalization(cpu_percent)),',', np.mean(normalization(mem_usage)))
        # print("\n",Temp_F1,',', np.mean(cpu_percent),',', np.mean(mem_usage))
        
        # cpu = np.divide(np.average(cpu_percent), 100, dtype=np.float)                                                                                # main line
        # cpu = np.divide(np.sum(cpu_percent), 100, dtype=np.float)
        # memory = np.sum(np.divide(np.divide(mem_usage,np.power(1024,2), dtype=np.float), total_memory, dtype=np.float))# main line
        cpu = np.sum(cpu_percent)
        memory= np.sum(mem_usage)
        # print('\ncpu: ',cpu, 'memory:' ,memory)
        # plot_file.write(str(mem_usage)+"\n")
        # plot_file.write(str(cpu_percent)+"\n")
        # plot_file.write(str(Temp_F1)+", "+str(t2-t1)+", "+str(memory)+", "+str(np.mean(mem_usage))+", "+str(cpu)+", "+str(np.mean(cpu_percent))+","+str(num_objs)+", "+str(mean_area_of_objs)+"\n")
        print("segment number:"+str(segment_number)+", F1: ",str(Temp_F1)+", time: "+str(t2-t1)+", sum_mem: "+str(memory)+", avg_mem: "+str(np.mean(mem_usage))+", cpu_sum: "+str(cpu)+", cpu_mean: "+str(np.mean(cpu_percent))+", number_objs:"+str(num_objs)+", areas: "+str(mean_area_of_objs)+"\n")
        
        # Both memory and cpu are percentage, i.e., a value in 
        return cpu, memory, Temp_F1, np.average(cpu_percent), t2-t1#np.average(mem_usage)

def subSpace_profiling(segment_number,S,C):
    # print("\n begin of subSpace_profiling \n",C)
    Results = []
    
    for i in range(len(C['frame_size'])):                        
        config = [C['frame_size'][i], C['frame_rate'][i], C['detection_model'][i]]
        cpu, memory, F1 = compute_F1_memory_cpu(segment_number,S,config)
        Results.append((1-F1) + np.divide(cpu,golden_cpu) + np.divide(memory,golden_mem))
        # print("F1: ", 1-F1, "cpu: ", np.divide(cpu,golden_cpu), "memory: ", np.divide(memory, golden_mem))
        
    # print(C)
    # print("Results", Results)
    
    best_idx = np.argmin(Results)
    worst_idx = np.argmax(Results)
    best_config = [C['frame_size'][best_idx], C['frame_rate'][best_idx], C['detection_model'][best_idx]]
    worst_config = [C['frame_size'][worst_idx], C['frame_rate'][worst_idx], C['detection_model'][worst_idx]]
    # print("best_idex before -1: ", best_idx)
    # best_idx = best_idx - 1
    # print("\nbest Index after -1: ", best_idx, " results length: ", len(Results))
    return best_config, worst_config, Results[best_idx]

def profiling(segment_number,S,C,K):  #version 3
    global knobs, golden_F1, golden_cpu, golden_mem
    # def compute_F1_memory_cpu_temp(C):
        # cpu_percent, mem_usage = monitor(Process_segment(S, C, True)) #main_line
        # Process_segment(S, C, True) # temporary line
        # cpu = np.average(cpu_percent)                                                                                       main line
        # memory = np.divide(np.divide(np.average(mem_usage),np.power(1024,2), dtype=np.float), total_memory, dtype=np.float) main line
        
        # Both memory and cpu are percentage, i.e., a value in 
        # return cpu, memory    

    # The below line  is for the time when we want to use golde configuration as the groundtruth, So in the current assumption 
    # it is not needed to be golden because the ground truth is obtained from dataset not from golden model.
    # But this is considered as a unique configuration because in other combinations in nested for 2 lines later it will be repeated for each knob.
    # Also it is mentioned in the chameleon article  that the knobs values are independent of each other, so we can use them seperately to see their performance
    golden_config = [C['frame_size'][0], C['frame_rate'][0], C['detection_model'][0]] 
    # golden_cpu, golden_mem = compute_F1_memory_cpu(golden_config) main line
    golden_cpu, golden_mem, golden_F1 = compute_F1_memory_cpu(segment_number,S,golden_config)
    # golden_cpu = 1 - golden_cpu
    # ====print("\nGolden_F1: ",1-golden_F1, "golden_cpu:", golden_cpu/golden_cpu, ' golden_memory:', golden_mem/golden_mem)
    # golden_cpu, golden_mem = knobs_cpu_mem[0][0]
    # golden_cpu = golden_cpu/100
    # golden_mem = np.divide(np.divide(np.average(golden_mem*1024*1024),np.power(1024,2), dtype=np.float), total_memory, dtype=np.float)
    
    # golden_F1 = np.average(GF1)    
    KnobValueToScore = [[] for j in range(len(C))]
    for i,knob in enumerate(C): 
        isnew_knob = True       
        for j,value in enumerate(C[knob][1:]):
            C_Vr = copy.deepcopy(golden_config)
            C_Vr[i] = value
            # here memory and cpu usage must be calculated            
            cpu, memory, F1 = compute_F1_memory_cpu(segment_number,S,C_Vr) #Main line
            #========= temporary section ============================
            # compute_F1_memory_cpu_temp(C_Vr) # temporary line
            # cpu, memory1 = knobs_cpu_mem[i][j+1] # temporary line
            # cpu = cpu / 100 
            # # memory = np.divide(np.divide(np.average(memory1*1024*1024),np.power(1024,2), dtype=np.float), total_memory, dtype=np.float) # temporary line
            # memory = np.divide(memory1*1024*1024, total_memory, dtype=np.float)
            #========================================================

            # if isnew_knob:
            #     KnobValueToScore[i].append([np.average(F1), cpu, memory])
            
            KnobValueToScore[i].append([np.average(F1), cpu/golden_cpu, memory/golden_mem])# KnobValueToScore[i].append(F1)
            # print('\nF1', F1, ' CPU: ', cpu/golden_cpu, 'Memory: ', memory/golden_mem)
            
    # Currently, different configurations are evaluated and their results are stored as an 2D array.
    # Removing the configurations whose F1 Scores are less than 0.8 (Alpha)
    alpha = 0.1    # no need to this section of code because the method removes the threshold requirement
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

    for i in range(AC_shape[0]):        
        if i == 0:
            acci = golden_F1
            memi = 1.#golden_mem
            cpui = 1.#golden_cpu
        else:
            acci = KnobValueToScore[0][i-1][0]
            memi = KnobValueToScore[0][i-1][2]
            cpui = KnobValueToScore[0][i-1][1]
        
        for j in range(AC_shape[1]):
            if j == 0:
                accj = golden_F1
                memj = 1.#golden_mem
                cpuj = 1.#golden_cpu
            else:
                accj = KnobValueToScore[1][j-1][0]
                memj = KnobValueToScore[1][j-1][2]
                cpuj = KnobValueToScore[1][j-1][1]
            
            for k in range(AC_shape[2]): 
                if k == 0:
                    acck = golden_F1
                    memk = 1.#golden_mem
                    cpuk = 1.#golden_cpu
                else:
                    acck = KnobValueToScore[2][k-1][0]
                    memk = KnobValueToScore[2][k-1][2]
                    cpuk = KnobValueToScore[2][k-1][1]
                # print("\n",i,",", j,",", k)
                # print("acci:", acci,", accj:", accj,", acck:", acck)
                # print("memi:", acci,", memj:", memj,", memk:", memk)
                # print("cpui:", cpui,", cpuj:", cpuj,", cpuk:", cpuk)
                temp_value = acci * accj * acck
                # print("temp_value: ", temp_value)

                if temp_value>=alpha_prime:
                    cpu = np.min([cpui, cpuj, cpuk])
                    memory = np.min([memi,memj,memk])
                    # AccuracteConfig_MEMCPU[i,j,k] = [temp_value,cpu,memory]                    
                    AccurateConfigs[i,j,k] = (1-temp_value)+cpu+memory # normalization is very important
                    # print("objective_function: ", AccurateConfigs[i,j,k] )
                # temp_value = KnobValueToScore[0][i][0] * KnobValueToScore[1][j][0] * (golden_F1 if k==0 else KnobValueToScore[2][0][0])#KnobValueToScore[2][k][0]
                # if temp_value>=alpha_prime:
                #     cpu = np.min([KnobValueToScore[0][i][1] , KnobValueToScore[1][j][1] , (golden_cpu if k==0 else KnobValueToScore[2][0][1])])
                #     memory = np.min([KnobValueToScore[0][i][2] , KnobValueToScore[1][j][2] , (golden_mem if k==0 else KnobValueToScore[2][0][2])])
                #     AccurateConfigs[i][j][k] = (1-temp_value) + cpu + memory # The goal is reducing memory and cpu, and increasing accuracy. In order to change the goal in accuracy as reduction target, it is subtracted by 1
                    
    # doing normalization on AccurateConfigs    
    # print(AccurateConfigs)
    
    # print(AccurateConfigs)
    exit()
    top_k_config, scores, worst = k_min_cell_index_3D_matrix(AccurateConfigs,top_k)
    # print("topK:", top_k_config)
    # print("\n worst: ", worst)
    # exit()
    # print("top_k: ", top_k_config)
    # print('==================================top_k configs=================================')
    # print(AccurateConfigs)
    # print('==================================top_k configs=================================')
    return golden_config, top_k_config, scores, [worst]

def F1_score(predicted_box, groundtruth, frame_number):   #version 3 
    # print(groundtruth)
    # print("inside F1_Score")
    # if all(groundtruth) and not all(predicted_box):
    #     return 0.0
    # elif predicted_box is None:
    #     predicted_box = [[0,0,0,0]]
    # gt = groundtruth[0]
    # pb = predicted_box[0]
    gt = groundtruth[0]
    pb = predicted_box[0]
    # print("gt:", gt)
    # print("pb:", pb)
    if gt[0]==gt[1]==gt[2]==gt[3]==0:
        if pb[0]==pb[1]==pb[2]==pb[3]==0:
            return 1.0
        else:
            return 0.0

    # if len(predicted_box)>0:
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
    # print("F1: =="+str(F1)+"==")
    return F1
    # else:
    #     # print("F1: ==0==")
    #     return 0

def Set_configuration(C):
    global targetSize, detection, trckr_type
    #C [i,  j,   k]
    # size, rate, model 
    
    targetSize = C[0]#knobs['frame_size'][C[0]]
    frame_rate = C[1]#knobs['frame_rate'][C[1]]    
    model = C[2]#knobs['detection_model'][C[2]]
    trckr_type = C[3]
    # tracker.switch_Tracker()        
    detection.Switch_detection(model) 

def _Results(frame_number, bboxes, gt,F1):
        global  MOT
        # print("result function")
        # print("inside _Results function ")
        # bboxes = tracker.Get_MOTChallenge_Format()
        MOT.write(frame_number, bboxes) #version 2
        
        F1.append(F1_score([bb[1:] for bb in bboxes], gt, frame_number)) #version 3
        # print("pass6")
        # F1[frame_number] = F1_score([bb[1:] for bb in bboxes], gt, frame_number)
        # return start_frame +1
        # print("F1:"+ str(F1))

def Process_segment(segment_number, segment, config, isprofiling, F1, detected_objs, shared_cont):  
    global targetSize, frame_rate, groundtruth_box,tracker, detection
    # print(targetSize)
    
    objs_list = []
    Set_configuration(config)         
    targetSize = int(targetSize)

    filename = 'seg'+str(segment_number)+'_'+str(config[0])+'_'+str(config[1])+'_'+config[2]+'.avi'
    out = None#cv2.VideoWriter('/home/mgharasu/Documents/Purified codes/outputs/'+filename, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (targetSize,targetSize))
    
    # out = cv2.VideoWriter('output.avi', -1, frame_rate, (targetSize,targetSize))
    # global start_frame, frame_size, tracker, video, detection, groundtruth_box, search_for_frame_to_detect_object, MOT
    # switching configs is done here before starting to 
    start_frame, end_frame = segment
    
    gt = None
    gt_idx = 0
    
    if isprofiling:
        tracker = None
    
    tmp_counter = 0
    after_n__frames_read = np.int(np.divide(end_frame-start_frame+1, config[1]))
    # print(after_n__frames_read)
    # number_of_frames_to_skip = np.int((end_frame-start_frame+1)/ after_n__frames_read)
    # print("tmp_frames_rate: ", after_n__frames_read)
    # print("number_of_frames_to_skip", number_of_frames_to_skip)
    # while start_frame <= end_frame:    
    t1 = time.time()
    while start_frame <= end_frame:
        # print(start_frame)
        
        ret, frame = video.get_frame_position(start_frame, shared_cont)        
        # === print(shared_cont)
        if not ret:
            # cont = False
            break
        # C_PREPROCESSING.MOT_gt_show(frame,groundtruth_box[start_frame])
        tmp_counter = tmp_counter + 1
        stdout.write('\r%d'% tmp_counter)
        # start_frame += 1
        if isprofiling:
            gt = groundtruth_box[start_frame]
            # print(gt)
            
            # if gt[0][0] == start_frame:
                # print("detected_gt")
            # preprocessing            
            frame, gt = C_PREPROCESSING.resize(frame,targetSize,gt)            
            # gt_idx += 1
            # else:
                # F1.append(0)
                # print("no groundtruth and F1: 0")
                # start_frame += 1
                # continue
            
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
            # print("detected objection")
            if len(detected_boxes) > 0:
                # print("detected objects: ",str(len(detected_boxes)))
                search_for_frame_to_detect_object = False
                if tracker is None:
                    # print("tracker empty: fill!!")
                    tracker = C_TRACKER(TRACKER_TYPE)
                    tracker.Add_Tracker(frame, detected_boxes)
                    # objs_list.append(detected_boxes)
                    # bboxes = tracker.Get_MOTChallenge_Format()
                    # _Results(start_frame, bboxes, gt, F1)
                    # out.write(frame)
                                     
                    # continue
                else:
                    # print("tracker is n't empty: update!!")
                    tracker.update_pipeline(frame,detected_boxes)
                    objs_list.append(detected_boxes)
                    #------------- profiling -----------
                    # if isprofiling:
                    #     bboxes = tracker.Get_MOTChallenge_Format()                        
                    #     _Results(start_frame,bboxes, gt,F1)                        
                    #     out.write(frame)

                    # continue
            
            # if there is no detected boxes by object detection, the tracker keeps on tracking previous objects
            if tracker is not None:
                # print("update tracker while there is not detected object")
                pb,frame = tracker.update(frame)
                objs_list.append(pb.tolist())
                #------------- profiling -----------
                # if isprofiling:
                #     bboxes = tracker.Get_MOTChallenge_Format()
                #     _Results(start_frame,bboxes, gt, F1)                        
                #     # start_frame += 1
                #     out.write(frame)            
            
            
            # cv2.imshow("output", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # print("no detected object no tracker initialized")            
            # continue
        else:
            # print("else section in case of using trackers before initialization")
            # in times when object detection is not used
            pb,frame = tracker.update(frame)
            objs_list.append(pb.tolist())
            #------------- profiling -----------
        if isprofiling:
            if tracker is not None:
                detected_boxes = tracker.Get_MOTChallenge_Format()
            else:
                detected_boxes=[[0,0,0,0,0]]
            _Results(start_frame,detected_boxes, gt, F1)
            # start_frame += 1
        
        # out.write(frame)

            # out.write(frame)
            # cv2.imshow("output", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        # start_frame += 1
        t2 = time.time()
        print("elpased time:", str(t2-t1))
    
    
    # cv2.destroyAllWindows()
    out.release()
    # print("\nF1_segmentProcess: ", F1)
    # if start_frame<=end_frame:
    # print("F1:",F1)
    #     cont = 0 # process of segment is not completed because of finishing the frames in video
    detected_objs.append(objs_list)
    start_frame = start_frame - 1

    return detected_objs

def Process_segment_frame_base_mp_bigTime(segment_number, segment, config, isprofiling, F1, detected_objs, shared_cont):
    global targetSize, frame_rate, groundtruth_box,tracker, detection
    def Process(frame, start_frame, tracker,after_n__frames_read, objs_list, out, tmp_counter,search_for_frame_to_detect_object,F1):
        # tracker = (C_TRACKER)tracker
        tmp_counter.value = tmp_counter.value + 1
        stdout.write('\r%d'% tmp_counter.value)
        # start_frame += 1
        if isprofiling:
            gt = groundtruth_box[start_frame.value]
            # preprocessing 
            frame, gt = C_PREPROCESSING.resize(frame,targetSize,gt)
            
        else:
            resize_image(frame, targetSize)

        if start_frame.value % UPDATE_TRACKER == 0 or start_frame.value == segment[0]: # each segment is determined here
            search_for_frame_to_detect_object.value = True
        
        start_frame.value = start_frame.value + after_n__frames_read

        if search_for_frame_to_detect_object.value:
            detected_boxes, frame = detection.Detection_BoundingBox(frame) 
            # print("detected objection")
            if len(detected_boxes) > 0:
                # print("detected objects: ",str(len(detected_boxes)))                
                search_for_frame_to_detect_object.value = False
                if tracker is None:#.Empty():
                    # print("tracker empty: fill!!")
                    tracker = C_TRACKER(TRACKER_TYPE)
                    tracker.Add_Tracker(frame, detected_boxes)  
                    # print("detected boxes", detected_boxes)                  
                    # print("initialized tracker")
                else:
                    tracker.update_pipeline(frame,detected_boxes)
                    objs_list.append(detected_boxes)
                    # print("not initialized")
                    
            # if there is no detected boxes by object detection, the tracker keeps on tracking previous objects
            # print("tracker.Empty():", tracker.Empty())
            tracker.Empty() # calling to change the variable value inside the function. 
                            # why it is happending (two time call to change the value of variable inside the function) is not clear.
            if not tracker.Empty(): #tracker is not None:
                pb,frame = tracker.update(frame)
                objs_list.append(pb.tolist())
                # print("tracker not empty")
                
        else:
            # in times when object detection is not used
            pb,frame = tracker.update(frame)
            # print("pb", pb)
            objs_list.append(pb.tolist())
            # print("tracker update")
            #------------- profiling -----------
        if isprofiling:
            if tracker is not None:
                detected_boxes = tracker.Get_MOTChallenge_Format()
            else:
                detected_boxes=[[0,0,0,0,0]]
            _Results(start_frame,detected_boxes, gt, F1)
            # start_frame += 1
        out.write(frame)
        
        # print("detected objects", objs_list)
        # print("tracker end of function:", tracker)
        # print("Pass 4")
        # return start_frame, tracker, objs_list, search_for_frame_to_detect_object, F1
    objs_list = []
    Set_configuration(config)         
    targetSize = int(targetSize)
    ret, frame = video.get_frame_position(0, shared_cont) 
    filename = 'seg'+str(segment_number)+'_'+str(config[0])+'_'+str(config[1])+'_'+config[2]+'.avi'
    out1 = cv2.VideoWriter('/home/mgharasu/Documents/Purified codes/outputs/'+filename, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (targetSize,targetSize))
    out = out1
    # out = mp.Manager().register('VideoWriter', cv2.VideoWriter)    
    # print("pass 1")
    out = out1
    fp = open("boxplot-video3.txt","a")
    st_frme, end_frame = segment    
    start_frame = mp.Manager().Value('start_frame',st_frme)
    # print("start_frame", start_frame.value)
    after_n__frames_read = np.int(np.divide(end_frame-st_frme+1, config[1])) 
    # end_frame = mp.Manager().Value('end_frame',0)

    
    gt = None
    gt_idx = 0
    
    # if isprofiling:
    #     tracker = None
    BaseManager.register('C_TRACKER', C_TRACKER)
    manager = BaseManager()
    manager.start()
    tracker = manager.C_TRACKER(TRACKER_TYPE)

    # tracker = mp.Manager().register('C_TRACKER', C_TRACKER)
    # mngr = mp.Manager()
    # print("tracker initialization: ", tracker)
    
    # tmp_counter = 0
    tmp_counter = mp.Manager().Value('temp_counter',0)
       
    
    # search_for_frame_to_detect_object = True
    search_for_frame_to_detect_object = mp.Manager().Value('search_for_frame_to_detect_object', True)
    while start_frame.value <= end_frame:  
        # print("start frame", start_frame.value)              
        ret, frame = video.get_frame_position(start_frame.value, shared_cont)        
        # print("frame type: ",type(frame))
        if not ret:
            break
        t1 = time.time()               
        # print("pass2")
        # F1 = []
        F1 = mp.Manager().list()#Value('F1',0.0)                
        detected_objs = []
        # detected_objs = mp.Manager().list()  
        # print("pass222")
        # frame_sh = mp.Manager().Array('i',frame)      
        
        # print("Pass 3")
        start_frame, tracker, detected_objs, search_for_frame_to_detect_object, F1 = Process(frame, start_frame, tracker,after_n__frames_read, detected_objs, out, tmp_counter, search_for_frame_to_detect_object, F1)#frame,start_frame,tracker,after_n__frames_read,objs_list,out, tmp_counter, search_for_frame_to_detect_object)
        # cpu_percent, mem_percent = monitor(target=Process, args=(frame, start_frame, tracker,after_n__frames_read, detected_objs, out, tmp_counter, search_for_frame_to_detect_object, F1))
        # print("cpu: ", np.mean(cpu_percent), " memory: ", np.mean(mem_percent))
        t2 = time.time()        
        # print("tracker out of monitor function: ", tracker)
        # print("out: ", out)
        # print("pass4")
        # print("\n F1:", str(F1.value))
        # print("F1: ",F1[-1])
        # print("detected objects", detected_objs)        
        # print("frame process time:", str(t2-t1),", F1: ",F1[-1],", num_detectedObjs:", len(detected_objs[-1]))        
        num_obj, avg_area = obj_statistics(detected_objs[-1], True)
        
        # print(num_obj,", ",str(t2-t1),", ",F1[-1],", ",avg_area)        
        # fp.write(str(num_obj)+","+str(t2-t1)+","+str(F1[-1])+","+str(avg_area)+","+str(np.mean(cpu_percent))+","+str(np.mean(mem_percent))+"\n")
        # print("average area:", str(avg_area))

        # for i, newbox in enumerate(detected_objs):
        #     p1 = (int(newbox[0]), int(newbox[1]))
        #     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #     cv2.rectangle(frame, p1, p2, self.colors[i][0:3], 2, 1)
    
    start_frame = start_frame -1
    # start_frame.value = start_frame.value -1
    fp.close()
    print("pass8")


    return detected_objs

def Get_ProcessID(a):
    # print("pass2",mp.current_process().pid)
    b = mp.current_process().pid
    return b

def Process(frame, start_frame,segment, isprofiling, tracker, after_n__frames_read, objs_list, tmp_counter,search_for_frame_to_detect_object,gt, F1):    
    global groundtruth_box, detection        
    cpu = []
    tmp_counter = tmp_counter + 1
    stdout.write('\r%d'% tmp_counter)
    p = psutil.Process(mp.current_process().pid)
    # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"1"+str(p.cpu_percent()))
    if search_for_frame_to_detect_object: 
        t1 = time.time()
        # print("object detection")
        # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"2:"+str(p.cpu_percent()))
        detected_boxes, frame = detection.Detection_BoundingBox(frame)
        objdetection = True
        # print("#objs: ", str(np.shape(frame)) +" for initialization")
        # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"3:"+str(p.cpu_percent()))
        t2 = time.time()
        if len(detected_boxes) > 0:
            # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"4:"+str(p.cpu_percent()))
            search_for_frame_to_detect_object = False            
            # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"5:"+str(p.cpu_percent()))
            if tracker.Empty():
                # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"6:"+str(p.cpu_percent()))
                tracker.Add_Tracker(frame, detected_boxes)  
                # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"7:"+str(p.cpu_percent()))
                # print("tracker initialization")
            else:
                # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"8:"+str(p.cpu_percent()))
                tracker.update_pipeline(frame,detected_boxes)
                # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"9:"+str(p.cpu_percent()))
                objs_list.append(detected_boxes)
                # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"10:"+str(p.cpu_percent()))
                # print("tracker update")
        # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"11:"+str(p.cpu_percent()))
        # if there is no detected boxes by object detection, the tracker keeps on tracking previous objects
        # print("tracker.Empty():", tracker.Empty())
        # tracker.Empty() # calling to change the variable value inside the function. 
                        # why it is happending (two time call to change the value of variable inside the function) is not clear.
        # print("update tracking: ", tracker.Empty())
        if not tracker.Empty() and len(detected_boxes)==0: #tracker is not None:
            print("pay attention") 
            # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"12:"+str(p.cpu_percent()))
            pb,frame = tracker.update(frame)
            # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"13:"+str(p.cpu_percent()))
            # print("pb: ", pb)
            objs_list.append(pb.tolist())
            # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"14:"+str(p.cpu_percent()))
            # print("tracker not empty")
        # cpu.append(p.cpu_percent)
            
    else:
        objdetection = False
        # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"15:"+str(p.cpu_percent()))
        # print("else section")
        # t1=time.time()
        # in times when object detection is not used        
        pb,frame = tracker.update(frame) 
        # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"16:"+str(p.cpu_percent()))
        # print("pb", pb)
        objs_list.append(pb.tolist()) 

    # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"17:"+str(p.cpu_percent()))
        # print("tracker update")
        # t2 = time.time()
        # print("tracker elapsed time: ", str(t2-t1))
        #------------- profiling -----------
    if isprofiling:
        # print("profiling")
        if not tracker.Empty():  
            # print("tracker Empty: False", tracker.Get_MOTChallenge_Format())
            detected_boxes = tracker.Get_MOTChallenge_Format()            
        else:
            # print("tracker empty: True") 
            detected_boxes=[[0,0,0,0,0]]
        
        # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"18:"+str(p.cpu_percent()))

        objs_list=detected_boxes
        # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"19:"+str(p.cpu_percent()))
        _Results(start_frame,detected_boxes, gt, F1)        

        # cpu.append(psutil.cpu_freq().current)#p.cpu_percent())#"17:"+str(p.cpu_percent()))

        # start_frame += 1
    # out.write(frame)
    cv2.imshow("output",frame)
    cv2.waitKey(1)
    
    
    return frame,start_frame, tracker, objs_list,     search_for_frame_to_detect_object, F1, tmp_counter, objdetection#, np.mean(cpu)
        #  frame,start_frame, tracker, detected_objs, search_for_frame_to_detect_object, F1, tmp_counter


def Process_segment_frame_base(segment_number, segment, config, isprofiling, F1, detected_objs, shared_cont):
    global trckr_type,targetSize, frame_rate, groundtruth_box, detection, pool, tracker, fp, pr

    objs_list = []
    Set_configuration(config)         
    targetSize = int(targetSize)
    print(config)
    fp.write(str(config)+"\n")
    # print("target size in process_segment_per_frame: ", targetSize)
    ret, frame = video.get_frame_position(0, shared_cont) 
    # filename = 'seg'+str(segment_number)+'_'+str(config[0])+'_'+str(config[1])+'_'+config[2]+'.avi'
    # out1 = cv2.VideoWriter('/home/mgharasu/Documents/Purified codes/outputs/'+filename, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (targetSize,targetSize))
    # out = out1
    # out = mp.Manager().register('VideoWriter', cv2.VideoWriter)    
    # print("pass 1")
    # out = out1
    # fp = open("boxplot-video2.txt","a")
    start_frame, end_frame = segment    
    # start_frame = mp.Manager().Value('start_frame',st_frme)
    # print("start_frame", start_frame.value)
    after_n__frames_read = np.int(np.divide(end_frame-start_frame+1, config[1])) 
    # end_frame = mp.Manager().Value('end_frame',0)    
    # print("before calling GetID")
    # func = partial(GetID,([3]))
    pid = pool.map(Get_ProcessID, [4])[0]
    p = psutil.Process(pid)

    ppid = os.getpid()
    parent_process = psutil.Process(ppid)    
  
    
    # p.nice(-19)
    # id = it.next()
    # print("id", pid)
    gt = None
    gt_idx = 0        
    
    tmp_counter = 0
    # tmp_counter = mp.Manager().Value('temp_counter',0)
        
    search_for_frame_to_detect_object = True
    old_objects = [[0,0,0,0,0]]
    # search_for_frame_to_detect_object = mp.Manager().Value('search_for_frame_to_detect_object', True)
    while start_frame <= end_frame:  
        # print("start frame", start_frame.value)              
        # print("before process the frame")
        ret, frame = video.get_frame_position(start_frame, shared_cont)        
        # print("frame type: ",type(frame))
        if not ret:
            break

        if isprofiling:
            gt = groundtruth_box[start_frame]
            # preprocessing
            # print("target_size in process function: ", targetSize)
            tmp = copy.deepcopy(frame)
            # print("frameSize: ", tmp.shape)
            frame, gt = C_PREPROCESSING.resize(frame,int(targetSize),gt)            
            tracker.switch_Tracker(frame, trckr_type,new_frame_size=int(targetSize))
            # print(targetSize) 
            # print(frame.shape)
            # cv2.imshow("output", frame)
            # cv2.waitKey(0)
        else:
            resize_image(frame, targetSize)

        if start_frame % UPDATE_TRACKER == 0 or start_frame == segment[0]: # each segment is determined here
            search_for_frame_to_detect_object = True
        
        start_frame = start_frame + after_n__frames_read
        # t1 = time.time()               
        # print("pass2")
        F1 = []
        # F1 = mp.Manager().list()#Value('F1',0.0)                
        """ detected_objs = []"""
        # detected_objs = mp.Manager().list()  
        # print("pass222")
        # frame_sh = mp.Manager().Array('i',frame)      
        
        # print("Pass 3")
        # start_frame, tracker, objs_list, search_for_frame_to_detect_object, F1, tmp_counter
        # it = pool.imap(Process,([]))
        # p.cpu_percent()
        t1 = time.time()  
        # p.is_running   
        # print("tracker in prcess per frame function:", tracker)      
        func = partial(Process, frame, start_frame,segment, isprofiling, tracker, after_n__frames_read, detected_objs, tmp_counter, search_for_frame_to_detect_object, gt)
        iterable = [[]]
        # print("moment 0.0001: ",p.cpu_percent(0.0001))            
        it = pool.imap(func, iterable)                
        frame, start_frame, tracker, detected_objs, search_for_frame_to_detect_object, F1, tmp_counter, objdetection = it.next() #Process(frame, start_frame, tracker,after_n__frames_read, detected_objs, out, tmp_counter, search_for_frame_to_detect_object, F1)        

        frame = draw_box(gt, frame)
        # print("moment 0.001: ",p.cpu_percent(0.01))
        # print(p.)
        cpu = p.cpu_percent()/psutil.cpu_count()/100
        # mem = (p.memory_info().rss+tracker.Get_trackerSize())/total_memory#tracker.Get_trackerSize()#
        mem_parent = parent_process.memory_info().rss
        mem_child = p.memory_info().rss
        # print("\nmmemory==>ain process:", parent_process.memory_info().rss,", child process: ", p.memory_info().rss)
        # print("cpu: ", cpu)        
        # start_frame, tracker, detected_objs, search_for_frame_to_detect_object, F1, tmp_counter = 
        # start_frame, tracker, detected_objs, search_for_frame_to_detect_object, F1, tmp_counter = Process(frame, start_frame, tracker,after_n__frames_read, detected_objs, out, tmp_counter, search_for_frame_to_detect_object, F1)#frame,start_frame,tracker,after_n__frames_read,objs_list,out, tmp_counter, search_for_frame_to_detect_object)
        t2 = time.time()        
        
        # cpu_percent, mem_percent = monitor(target=Process, args=(frame, start_frame, tracker,after_n__frames_read, detected_objs, out, tmp_counter, search_for_frame_to_detect_object, F1))
        # print("cpu: ", np.mean(cpu_percent), " memory: ", np.mean(mem_percent))
        
        # print("tracker out of monitor function: ", tracker)
        # print("out: ", out)
        # print("pass4")
        # print("\n F1:", str(F1.value))
        # print("F1: ",F1[-1])
        # print("detected objects", detected_objs)        
        
        # print("detected objs", detected_objs)
        num_obj, areas, conflicts = obj_statistics(detected_objs, True)
        velocities = velocity(detected_objs, old_objects)
        

        # print("frame process time:", str(t2-t1),", F1: ",F1[-1],", num_detectedObjs:", num_obj)#len(detected_objs[-1]))
        # print("time process: ",t2-t1, ", cpu: ", cpu, ", mem: ", mem,", num_0bjs: ", num_obj, ", avg_area: ", avg_area," , F1: ",F1)
        # print(num_obj,", ",str(t2-t1),", ",F1[-1],", ",avg_area)        
        # fp.write(str(num_obj)+","+str(t2-t1)+","+str(F1[-1])+","+str(avg_area)+",0.0,"+str(mem)+"\n")        
        num_objects = 0 if detected_objs==[[0,0,0,0,0]] else num_obj
        # print("num_Objects: ", num_objects,", memory: ",mem)        
        
        #====== number of objects, execution time, F1 score, Average Area, CPU, Memroy Parent, Memory Child, \n detected objects, conflicts \n velocities
        #====== object bounding box, conflict of object with others
        
        # print("Num_obj: "+str(num_objects)+", time: "+str(t2-t1)+", F1: "+str(F1[-1])+", CPU: "+str(cpu)+", Mem_parent:"+str(mem_parent)+", Mem_child:"+str(mem_child)+", ")
        print(F1)
        """
        if not objdetection:
            fp.write(str(num_objects)+","+str(t2-t1)+","+str(F1[-1])+","+str(cpu)+","+str(mem_parent)+","+str(mem_child)+", tr\n")
            # print("tr")
        else:
            fp.write(str(num_objects)+","+str(t2-t1)+","+str(F1[-1])+","+str(cpu)+","+str(mem_parent)+", "+str(mem_child)+", objD\n")
            # print("objD")

        for i in range(num_obj):
            fp.write(str(detected_objs[i])+"\n")
            # print("detected _objs: "+str(detected_objs[i])+"\n")
        fp.write(str(conflicts)+"\n")
        # print("conflicts:" , str(conflicts)+"\n")
        fp.write(str(velocities)+"\n")
        # print("Velocity: "+str(velocities))
        fp.write(str(areas)+"\n")
        # print("areas: "+str(areas))
        """
        # cv2.imshow("output",frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        old_objects = copy.deepcopy(detected_objs)
    start_frame = start_frame -1
    # start_frame.value = start_frame.value -1
    # fp.close()    


    return detected_objs


def Sub_knob_creator(config, ):
    # print("sub_knob_creator->config:", config)
    new_knob = {}
    new_knob['frame_size']=[]
    new_knob['frame_rate']=[]
    new_knob['detection_model']=[]
    if len(np.shape(config)) > 1:
        for c in config:
            new_knob['frame_size'].append(knobs['frame_size'][c[0]])
            new_knob['frame_rate'].append(knobs['frame_rate'][c[1]])
            # new_knob['detection_model'].append(knobs['detection_model'][c[2]])
    else:
        new_knob['frame_size'].append(knobs['frame_size'][config[0]])
        new_knob['frame_rate'].append(knobs['frame_rate'][config[1]])
        # new_knob['detection_model'].append(knobs['detection_model'][config[2]])
    return new_knob

def profiling_for_F1(segment_number, S,C):
    golden_config = [C['frame_size'][0], C['frame_rate'][0], C['detection_model'][0], C['tracker'][0]] 
    
    golden_cpu, golden_mem, golden_F1, c,m = compute_F1(segment_number,S,golden_config)    

    configs_measurements = np.zeros((len(knobs['frame_size']), len(knobs['frame_rate']), len(knobs['detection_model']), len(knobs['tracker'])))
    configs_measurements[0][0][0] = (1-np.average(golden_F1))
    
    for i,size in enumerate(knobs['frame_size']):
        for j, fps in enumerate(knobs['frame_rate']):
            for k, model in enumerate(knobs['detection_model']):
                for w, tr in enumerate(knobs['tracker']):
                    if i==j==k==w==0: #golden configuration
                        continue                
                    config = [C['frame_size'][i], C['frame_rate'][j], C['detection_model'][k], C['tracker'][w]]
                    cpu, mem, F1,c,m = compute_F1(segment_number,S,config)
    #             configs_measurements[i][j][k] = (1-np.average(F1))+ (cpu/golden_cpu)+ (mem/golden_mem)                
                
    # sorted_configs, sorted_scores, worst_config = k_min_cell_index_3D_matrix(configs_measurements,np.prod(np.shape(configs_measurements)))
    # return golden_config, sorted_configs, sorted_scores, [worst_config]
    return golden_config, golden_config, golden_F1, [golden_config]

def new_profiling(segment_number,S,C):    
    global knobs, golden_F1, golden_cpu, golden_mem
    
    golden_config = [C['frame_size'][0], C['frame_rate'][0], C['detection_model'][0]] 
    
    golden_cpu, golden_mem, golden_F1 = compute_F1_memory_cpu(segment_number,S,golden_config)
    
    KnobValueToScore = [[] for j in range(len(C))]
    for i,knob in enumerate(C): 
        isnew_knob = True       
        for j,value in enumerate(C[knob][1:]):
            C_Vr = copy.deepcopy(golden_config)
            C_Vr[i] = value
            # Here memory and cpu usage must be calculated            
            cpu, memory, F1 = compute_F1_memory_cpu(segment_number,S,C_Vr) #Main line            
            KnobValueToScore[i].append([np.average(F1), cpu/golden_cpu, memory/golden_mem])# KnobValueToScore[i].append(F1)                
    
    AccurateConfigs = 1000*np.ones((len(knobs['frame_size']),len(knobs['frame_rate']),len(knobs['detection_model'])), dtype=np.float)
    num_elements = len(knobs['frame_size'])*len(knobs['frame_rate'])*len(knobs['detection_model'])
    AC_shape = np.shape(AccurateConfigs)
    
    alpha_prime = 0.1
    # in fact, alpha_prime is not usable in our method and its value is so small that all configurations are considered in profiling
    for i in range(AC_shape[0]):
        # ======================================================
        if i == 0:
            acci, memi, cpui = golden_F1, 1, 1            
        else:
            acci, memi, cpui = KnobValueToScore[0][i-1][0], KnobValueToScore[0][i-1][2], KnobValueToScore[0][i-1][1]                   
        # =======================================================
        for j in range(AC_shape[1]):
            # ===================================================
            if j == 0:
                accj, memj, cpuj = golden_F1, 1, 1                
            else:
                accj, memj, cpuj = KnobValueToScore[1][j-1][0], KnobValueToScore[1][j-1][2], KnobValueToScore[1][j-1][1]
            # ===================================================
            for k in range(AC_shape[2]): 
                # ===================================================
                if k == 0:
                    acck, memk, cpuk = golden_F1,1,1                    
                else:
                    acck, memk, cpuk = KnobValueToScore[2][k-1][0], KnobValueToScore[2][k-1][2], KnobValueToScore[2][k-1][1]
                # ===================================================
                temp_value = acci * accj * acck

                if temp_value>=alpha_prime:
                    cpu = np.min([cpui, cpuj, cpuk])
                    memory = np.min([memi,memj,memk])
                    # AccuracteConfig_MEMCPU[i,j,k] = [temp_value,cpu,memory]                    
                    AccurateConfigs[i,j,k] = (1-temp_value)+cpu+memory # normalization is very important
                        
    top_k_config, scores, worst = k_min_cell_index_3D_matrix(AccurateConfigs,num_elements)
    # print(AccurateConfigs)
   
    return golden_config, top_k_config, scores, [worst]

def profiling_without_interpolation(segment_number, S, C):
    global knobs, golden_F1, golden_cpu, golden_mem
    # fp = open('plot.txt','w')
    """
    fp = open('outputs/seg'+str(segment_number)+'.txt','w')
    """
    # plt_config = []
    # plt_mem = []
    # plt_CPU = []
    golden_config = [C['frame_size'][0], C['frame_rate'][0], C['detection_model'][0], C['tracker'][0]] 
    golden_cpu, golden_mem, golden_F1, c,m = compute_F1_memory_cpu(segment_number,S,golden_config)        
    # plt_config.append(str(golden_config))
    # plt_mem.append((m/1024)/1024)
    # plt_CPU.append(c)

    # fp.writelines(str(golden_config)+" - "+str(c)+" - "+str((m/1024)/1024)+"\n")
    # fp.writelines(str(golden_config)+" - "+str(golden_cpu)+" - "+str((golden_mem/1024)/1024)+"\n")
    fp.writelines(str(golden_config)+":"+str(m))#str(golden_F1)+","+str(c)+","+ str(m)+"\n")
    configs_measurements = np.zeros((len(knobs['frame_size']), len(knobs['frame_rate']), len(knobs['detection_model'])))
    configs_measurements[0][0][0] = (1-np.average(golden_F1))+1+1
    
    return golden_config, golden_config, golden_F1, [golden_config]

    for i,size in enumerate(knobs['frame_size']):
        for j, fps in enumerate(knobs['frame_rate']):
            for k, model in enumerate(knobs['detection_model']):
                for w, tr in enumerate(knobs['tracker']):
                    if i==j==k==w==0:
                        continue
                    #print('\n-',counter,'-   => frame size:', size, ' frame rate:', fps,' model:', model)
                    config = [C['frame_size'][i], C['frame_rate'][j], C['detection_model'][k]]
                    cpu, mem, F1,c,m = compute_F1_memory_cpu(segment_number,S,config)
                    fp.writelines(str(config)+":"+str(np.mean(F1))+","+str(c)+","+ str(m)+"\n")
                    configs_measurements[i][j][k] = (1-np.average(F1))+ (cpu/golden_cpu)+ (mem/golden_mem)                
                    # plt_config.append(str(config))                
                    # plt_CPU.append(c)
                    # plt_mem.append((m/1024)/1024)
                    # fp.writelines(str(config)+" - "+str(c)+" - "+str((m/1024)/1024)+"\n")
                    # print(str(config)+" - cpu:"+str(c)+" - mem:"+str((m/1024)/1024)+"cpu:"+str(1-np.average(F1))+"\n")

    
    # print(configs_measurements)
    # exit()
    
    sorted_configs, sorted_scores, worst_config = k_min_cell_index_3D_matrix(configs_measurements,np.prod(np.shape(configs_measurements)))
    # import pandas as pd
    # print("before print")
    """
    fp.writelines("sorted:"+ str(sorted_configs))
    fp.close()
    """
    # df = pd.DataFrame({ 'cpu':plt_CPU}, index = plt_config)
    # ax = df.plot.barh()
    # plt.xlabel('cpu percent use')
    # plt.ylabel('configurations')
    # plt.show()
    # df = pd.DataFrame({ 'mem':plt_mem}, index = plt_config)
    # ax = df.plot.barh()
    # plt.xlabel('memory usage based on MB unit')
    # plt.ylabel('configurations')
    # plt.show()
    # exit()
    return golden_config, sorted_configs, sorted_scores, [worst_config]

def pipeline():
    '''
    Here division of video is done and 
    choosing configuration (profiling) is done
    to process segments of video    
    '''
    num_knobs = len(knobs['frame_size']) * len(knobs['frame_rate']) * len(knobs['detection_model']) * len(knobs['tracker'])
    
    
    # top_k_config = [[1, 4, 1], [3, 4, 1], [0, 4, 1], [2, 4, 1], [4, 4, 1], [1, 3, 1], [3, 3, 1], [0, 3, 1], [2, 3, 1], [1, 4, 0], [4, 3, 1], [3, 4, 0], [0, 4, 0], [2, 4, 0], [4, 4, 0], [1, 3, 0], [0, 3, 0], [3, 3, 0], [2, 3, 0], [4, 3, 0], [1, 2, 1], [3, 2, 1], [0, 2, 1], [2, 2, 1], [4, 2, 1], [1, 2, 0], [3, 2, 0], [0, 2, 0], [2, 2, 0], [4, 2, 0], [1, 1, 1], [0, 1, 1], [3, 1, 1], [2, 1, 1], [4, 1, 1], [1, 1, 0], [3, 1, 0], [0, 1, 0], [2, 1, 0], [4, 1, 0], [4, 0, 1], [2, 0, 1], [3, 0, 1], [4, 0, 0], [3, 0, 0], [2, 0, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0]]
    # top_k_score = [0.96340389, 0.96887819, 0.96887819, 0.97471525, 0.9759767 , 1.02039462,
    #                 1.0259633,  1.0259633 , 1.031901  , 1.03204826, 1.0331842 , 1.03563976,
    #                 1.03563976, 1.03946924, 1.04029684, 1.09022251, 1.09387593, 1.09387593,
    #                 1.09777144, 1.0986133 , 1.2090715 , 1.21475199, 1.21475199, 1.22080891,
    #                 1.22211788, 1.28030142, 1.28402818, 1.28402818, 1.28800191, 1.28886068,
    #                 1.47630344, 1.48199964, 1.48199964, 1.4880733 , 1.48938589, 1.54773032,
    #                 1.55146739, 1.55146739, 1.5554521 , 1.55631325, 2.21713089, 2.25196762,
    #                 2.28678496, 2.28913121, 2.3652064 , 2.36920306, 2.40537189, 2.4619179,
    #                 2.70217191, 2.85744487]
    # schdlr.Initialize_with_value(top_k_config, top_k_score)
    # schdlr.update_v2(0.9)    
    global configs, frame_rate, cont_read_frame
    cont_read_frame = True
    #each segement is 1 second or 14 frames because FPS = 14 in this video
    W = 1 # window includes 5 segments
    T = 4 #each segment 4 seconds
    t = 1 #seconds
    sequence_recorder = []
    top_k_config = []
    segment_number = 0    
    cont = True
    start_frame = 0

    best_configs = []
    worst_configs = []

    global tracker, pool, fp #, print
    pool = Pool(processes = 1)
    fp = open("features_v"+FOLDER+"_NEW.txt.junkfilefortest","w")
    BaseManager.register('C_TRACKER', C_TRACKER)
    manager = BaseManager()
    manager.start()
    tracker = manager.C_TRACKER(TRACKER_TYPE)
    
    while cont_read_frame:
        # segment = [start_frame, start_frame+frame_rate*T-1] #each segment is T seconds and each second is FPS=14  
        # if start_frame > 500:
        #     break
        if segment_number % W == 0:
        # if segment_number == 0:
            end_frame = start_frame+frame_rate-1
            print("[",start_frame, end_frame,"]")
            # golden_config, top_k_config, scores, worst = new_profiling(segment_number,[start_frame, end_frame],knobs,5)
            # change for using in scheduler
            all_configs = len(knobs['frame_size'])*len(knobs['frame_rate'])*len(knobs['detection_model'])*len(knobs['tracker'])
            golden_config, top_k_config, top_k_score, worst = profiling_for_F1(segment_number,[start_frame, end_frame],knobs) #profiling_without_interpolation(segment_number,[start_frame, end_frame],knobs)#profiling_for_F1(segment_number,[start_frame, end_frame],knobs)#new_profiling(segment_number,[start_frame, end_frame],knobs)profiling_without_interpolation(segment_number,[start_frame, end_frame],knobs)##new_profiling(segment_number,[start_frame, end_frame],knobs)#profiling_for_F1(segment_number,[start_frame, end_frame],knobs)##new_profiling(segment_number,[start_frame, end_frame],knobs)profiling_without_interpolation(segment_number,[start_frame, end_frame],knobs)#profiling_for_F1(segment_number,[start_frame, end_frame],knobs)##new_profiling(segment_number,[start_frame, end_frame],knobs)           
            # quit()
            # configs = Sub_knob_creator(top_k_config)
            # schdlr.Initialize_with_value(top_k_config, top_k_score)
            
            #================== before applying scheduler =======================
            # sub_knobs = Sub_knob_creator(top_k_config)
            # print(worst)
            # print(top_k_config)
            # worst_c = Sub_knob_creator(worst)
            # best_c = Sub_knob_creator([top_k_config[0]])
            # worst_configs.append(worst_c)
            # best_configs.append(best_c)
            # print('worst_c: ',worst_c)
            # print('best_c: ', best_c)  
            #==================================================================
            # print('filename: seg'+str(segment_number)+'_'+str(sub_knobs['frame_size'][0])+"_"+str(sub_knobs['frame_rate'][0])+"_"+sub_knobs['detection_model'][0]+".avi")
            # sequence_recorder.append('seg'+str(segment_number)+'_'+str(sub_knobs['frame_size'][0])+"_"+str(sub_knobs['frame_rate'][0])+"_"+sub_knobs['detection_model'][0]+".avi")
            # print('filename: seg'+str(segment_number)+'_'+str(golden_config[0])+"_"+str(golden_config[1])+"_"+golden_config[2]+".avi")
            # sequence_recorder.append('seg'+str(segment_number)+'_'+str(golden_config[0])+"_"+str(golden_config[1])+"_"+golden_config[2]+".avi")
            print("================================= segment ",segment_number," -whole space -=======================================")
            # cont=False

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
            if start_frame>900:
                break

            end_frame = start_frame+frame_rate-1
            print("[",start_frame, end_frame,"]")
            
            selected_config = schdlr.pop()
            sub_knobs = Sub_knob_creator(selected_config)
            config, _, score = subSpace_profiling(segment_number, [start_frame, end_frame], sub_knobs)
            # schdlr.update_v2(score)
            # ================================ before applying scheduler =====================================================
            # config, worst = subSpace_profiling(segment_number, [start_frame, end_frame], sub_knobs)
            # best_configs.append(config)
            # worst_configs.append(worst)
            # print('worst: ', worst)
            # print('best: ', config)
            # sequence_recorder.append('seg'+str(segment_number)+'_'+str(config[0])+"_"+str(config[1])+"_"+config[2]+".avi")
            # ================================ before applying scheduler =====================================================
            # print('filename: seg'+str(segment_number)+'_'+str(config[0])+"_"+str(config[1])+"_"+config[2]+".avi")
            # print("read_frame?  ", cont_read_frame)
            print("================================= segment ",segment_number," =======================================")
            
            # sub_segment = [start_frame, start_frame*(T-t)-1] # the remaining T-t seconds of the segment
            # cont = Process_segment(sub_segment, config,False)
        start_frame = end_frame+1#start_frame+end_frame+1
        segment_number += 1

    pool.terminate()
    MOT.close()
    fp.close()
    return 0

if __name__ == "__main__":
    # from pipeline2 import main
    # cpu,mem = monitor(main)
    pipeline()
    