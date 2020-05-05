from video_capture import C_VIDEO_UNIT
# from multitracking import box_iou2, correct_position
from preprocessing import C_PREPROCESSING
from Object_detection import C_DETECTION
from tracking import C_TRACKER
from utils import *
from MOT_File_Generator import C_MOT_OUTPUT_GENERATER # version 2
from Scheduler import C_SCHEDULER
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
import multiprocessing as mp
import time
import sys

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
start_frame = 0
search_for_frame_to_detect_object = True
targetSize = 1.0
#=================================== version 3 ===================================================
total_memory = np.divide(virtual_memory().total, np.power(1024,2), dtype=np.float) #version 3
top_k = 5                                                               #version 3
frame_rate = 30   #from config file                                     #version 3
# GF1 = mp.Array('d',0.0)                                               #version 3
knobs = {}                                                              #version 3
knobs['frame_size'] = [960,840,720,600,480]                             #version 3
knobs['frame_rate'] = [30,10,5,2,1]                                     #version 3
knobs['detection_model']=['Deep_Yolo']#,'Deep_Yolo_Tiny'] #version 3#'HOG_Pedestrian',
configs = copy.deepcopy(knobs)
golden_F1 = golden_cpu = golden_mem = 0.0

knobs_cpu_mem = [[[28.113,668.25],[28.95, 652.15],[29.7, 640.35],[30.8,639.93],[31.7, 637.67]],[[28.113,668.25],[28.23, 614.3],[25.21, 567.297],[25.895, 520.37],[27.042, 492.928]],[[28.113,668.25],[45.05, 306.914]]]
""" C is configuration to set in system """
def compute_F1_memory_cpu(segment_number,S,C2):
        # ps = SEGMENT()
        # print('frame_rate: ', C2[1])
        F1 = mp.Manager().list()#mp.Array('d', range(C2[1]+2))
        shared_cont = mp.Manager().Value('d','1')
        cpu_percent, mem_usage = monitor(Process_segment, (segment_number, S, C2, True,F1, shared_cont)) #, detection, video, groundtruth_box, targetSize,)) #main_line
        cont_read_frame = shared_cont
        # print(cont_read_frame)
        Temp_F1 = np.average(F1)
        print('\n','cofig',C2,' monitor_F1:', Temp_F1, ' cpu: ', np.sum(cpu_percent), 'memory: ', np.sum(mem_usage))
        
        # cpu = np.divide(np.average(cpu_percent), 100, dtype=np.float)                                                                                # main line
        # cpu = np.divide(np.sum(cpu_percent), 100, dtype=np.float)
        # memory = np.sum(np.divide(np.divide(mem_usage,np.power(1024,2), dtype=np.float), total_memory, dtype=np.float))# main line
        cpu = np.sum(cpu_percent)
        memory= np.sum(mem_usage)
        # print('\ncpu: ',cpu, 'memory:' ,memory)
        
        # Both memory and cpu are percentage, i.e., a value in 
        return cpu, memory, Temp_F1, np.average(cpu_percent), np.average(mem_usage)

def subSpace_profiling(segment_number,S,C):
    print("\n begin of subSpace_profiling \n",C)
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
    if len(predicted_box)>0:
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
        print("F1: =="+str(F1)+"==")
        return F1
    else:
        print("F1: ==0==")
        return 0

def Set_configuration(C):
    global targetSize, detection
    #C [i,  j,   k]
    # size, rate, model 
    
    targetSize = C[0]#knobs['frame_size'][C[0]]
    frame_rate = C[1]#knobs['frame_rate'][C[1]]    
    model = C[2]#knobs['detection_model'][C[2]]
    # tracker.switch_Tracker()        
    detection.Switch_detection(model)
    

def _Results(frame_number, bboxes, gt,F1):
        global  MOT
        print("inside _Results function ")
        # bboxes = tracker.Get_MOTChallenge_Format()
        MOT.write(frame_number, bboxes) #version 2
        if len(bboxes)>0:
            F1.append(F1_score([bb[1:] for bb in bboxes], gt, frame_number)) #version 3
        # F1[frame_number] = F1_score([bb[1:] for bb in bboxes], gt, frame_number)
        # return start_frame +1
        print("F1:"+ str(F1))

def Process_segment(segment_number, segment, config, isprofiling, F1, shared_cont):  
    # global F1       
    # print(segment)
    # print(config)
    global targetSize, frame_rate, groundtruth_box,tracker, detection
    # print(targetSize)
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
        shared_cont, frame = video.get_frame_position(start_frame, shared_cont)        
        # === print(shared_cont)
        if not shared_cont:
            # cont = False
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
                # if tracker is None:
                if tracker.Empty():
                    # tracker = C_TRACKER(TRACKER_TYPE)
                    tracker.Add_Tracker(frame, detected_boxes)
                    
                    # bboxes = tracker.Get_MOTChallenge_Format()
                    # _Results(start_frame, bboxes, gt, F1)
                    # out.write(frame)                   
                else:
                    tracker.update_pipeline(frame,detected_boxes)
                    #------------- profiling -----------
                    # if isprofiling:
                    #     bboxes = tracker.Get_MOTChallenge_Format()                        
                    #     _Results(start_frame,bboxes, gt,F1)                        
                    #     # start_frame += 1
                    #     out.write(frame)

                    # continue
            
            # if there is no detected boxes by object detection, the tracker keeps on tracking previous objects
            if not tracker.Empty() and len(detected_boxes)==0:#tracker is not None:
                frame = tracker.update(frame)
                #------------- profiling -----------
                # if isprofiling:
                #     bboxes = tracker.Get_MOTChallenge_Format()
                #     _Results(start_frame,bboxes, gt, F1)                        
                #     out.write(frame)
            
            # cv2.imshow("output", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            # continue
        else:
            # in times when object detection is not used
            frame = tracker.update(frame)
            #------------- profiling -----------
        if isprofiling:
            bboxes = tracker.Get_MOTChallenge_Format()
            _Results(start_frame,bboxes, gt, F1)
            # start_frame += 1
            out.write(frame)

            # out.write(frame)
            print()
            # cv2.imshow("output", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        # start_frame += 1


    # cv2.destroyAllWindows()
    out.release()
    print("\nF1_segmentProcess: ", np.average(F1))
    # if start_frame<=end_frame:
    #     cont = 0 # process of segment is not completed because of finishing the frames in video
    start_frame = start_frame - 1

def Sub_knob_creator(config, ):
    print("sub_knob_creator->config:", config)
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
    
    fp = open('plot.txt','w')

    plt_config = []
    plt_mem = []
    plt_CPU = []
    golden_config = [C['frame_size'][0], C['frame_rate'][0], C['detection_model'][0]] 
    golden_cpu, golden_mem, golden_F1, c,m = compute_F1_memory_cpu(segment_number,S,golden_config)    

    plt_config.append(str(golden_config))
    plt_mem.append((m/1024)/1024)
    plt_CPU.append(c)

    fp.writelines(str(golden_config)+" - "+str(c)+" - "+str((m/1024)/1024)+"\n")

    configs_measurements = np.zeros((len(knobs['frame_size']), len(knobs['frame_rate']), len(knobs['detection_model'])))
    configs_measurements[0][0][0] = (1-np.average(golden_F1))+1+1
    
    for i,size in enumerate(knobs['frame_size']):
        for j, fps in enumerate(knobs['frame_rate']):
            for k, model in enumerate(knobs['detection_model']):
                if i==j==k==0:
                    continue
                #print('\n-',counter,'-   => frame size:', size, ' frame rate:', fps,' model:', model)
                config = [C['frame_size'][i], C['frame_rate'][j], C['detection_model'][k]]
                cpu, mem, F1,c,m = compute_F1_memory_cpu(segment_number,S,config)
                configs_measurements[i][j][k] = (1-np.average(F1))+ (cpu/golden_cpu)+ (mem/golden_mem)                
                plt_config.append(str(config))                
                plt_CPU.append(c)
                plt_mem.append((m/1024)/1024)
                fp.writelines(str(config)+" - "+str(c)+" - "+str((m/1024)/1024)+"\n")
                # print(str(config)+" - cpu:"+str(c)+" - mem:"+str((m/1024)/1024)+"cpu:"+str(1-np.average(F1))+"\n")

    
    # print(configs_measurements)
    # exit()
    fp.close()
    sorted_configs, sorted_scores, worst_config = k_min_cell_index_3D_matrix(configs_measurements,np.prod(np.shape(configs_measurements)))
    # import pandas as pd

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
    num_knobs = len(knobs['frame_size']) * len(knobs['frame_rate']) * len(knobs['detection_model'])
    schdlr = C_SCHEDULER(num_knobs,3)
    
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
    while cont_read_frame:
        # segment = [start_frame, start_frame+frame_rate*T-1] #each segment is T seconds and each second is FPS=14  
        if segment_number % W == 0:
        # if segment_number == 0:
            end_frame = start_frame+frame_rate-1
            print("[",start_frame, end_frame,"]")
            # golden_config, top_k_config, scores, worst = new_profiling(segment_number,[start_frame, end_frame],knobs,5)
            # change for using in scheduler
            all_configs = len(knobs['frame_size'])*len(knobs['frame_rate'])*len(knobs['detection_model'])
            golden_config, top_k_config, top_k_score, worst = profiling_without_interpolation(segment_number,[start_frame, end_frame],knobs)#new_profiling(segment_number,[start_frame, end_frame],knobs)
            configs = Sub_knob_creator(top_k_config)
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
            schdlr.update_v2(score)
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

    MOT.close()
    return 0

if __name__ == "__main__":
    # from pipeline2 import main
    # cpu,mem = monitor(main)
    pipeline()