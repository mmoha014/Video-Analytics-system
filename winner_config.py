# from scipy.scipy.spatial import distance
from encoder import seq2vec, seq2vec_readSource
import numpy as np
import cv2
from video_capture import C_VIDEO_UNIT
# from test2 import seq2vec
import pickle
import torch
# import scipy.spatial
from scipy.spatial import distance
from scipy.spatial import distance_matrix

# def adjusted_cosine()
sizeidx ={'480':0, '600':1, '720':2, '840':3, '960':4}
rateidx = {'1':0, '2':1,'5':2,'10':3,'30':4}
modelidx = {'faster_rcnn_X_101_32x8d_FPN_3x':0,'faster_rcnn_R_101_DC5_3x':1,'retinanet_R_101_FPN_3x':2, 'faster_rcnn_R_50_FPN_3x':3}


dataset = pickle.load(open('dataset_6output.p','rb'))    

#########################################################################################
################################ Read Total Configs #######################################
#########################################################################################

# l_seg = list()
# l_fr_size = list()
# l_rate = list()
# l_model = list()
# l_ET = list()
# l_F1 = list()
# l_fraction = list()
# l_of = list()
# l_line = list()
# Total = list()

# def add_new_entry():
#     l_seg.append(list())
#     l_fr_size.append(list())
#     l_rate.append(list())
#     l_model.append(list())
#     l_ET.append(list())
#     l_F1.append(list())
#     l_fraction.append(list())
#     l_of.append(list())
#     l_line.append(list())
#     Total.append(list())

# def append2lists(segm,fr_size,fr_rate,model,et,f1,frac,of,line):
#     # print(line)
#     l_seg[segm].append(segm)
#     l_fr_size[segm].append(fr_size)
#     l_rate[segm].append(fr_rate)
#     l_model[segm].append(model)
#     l_ET[segm].append(et)
#     l_F1[segm].append(f1)
#     # if frac is not None:
#     #     l_fraction[segm].append(frac)
#     # else:
#     l_fraction[segm].append(frac)
#     l_of[segm].append(of)
#     l_line[segm].append(line)
#     Total[segm].append([line,segm,fr_size,fr_rate,model,et,f1,frac,of])



#     number = '1'
#     segm_count = -1
#     line_count = 0
#     alpha=0.2
#     # rfp = open('gt'+number+'_detectron2_960_record.txt','r')
#     rfp = open('','r')#'/home/morteza/Videos/traffic camera/profile/CAM822_Virginia/profile_on_all_configs/gt_'+number+'.txt.record.check')#'gt'+str(number)+'_profile.txt','r')
#     for line in  rfp.readlines():
#         line_count += 1
#         # if line_count == 12:
#         #     a=0
#         tmp = line.split(',')
#         segm = np.int16(tmp[0])    
#         fr_size = tmp[1]
#         fr_rate = tmp[2]
#         model = tmp[3]
#         if True:#int(number)>2:
#             ET = np.float(tmp[-3])
#             F1 = np.float(tmp[-2])
#             fraction = np.float(tmp[-1][:-1])
#         else:
#             ET = np.float(tmp[-2])
#             F1 = np.float(tmp[-1])
#         if True: #fraction>=0.8 and ET<=1.0:
#             if True:#ET<=1.0:
#                 obj_func = (alpha*(7-ET))+((1-alpha)*7*F1)
#             else:
#                 obj_func = 0.0
#             # print('line:',line_count,', of: ',obj_func)
#             if segm_count<segm:
#                 add_new_entry()
#                 segm_count+=1
#             if True:#int(number)>2:
#                 Total.append(segm,fr_size,fr_rate,model,ET,F1,fraction,obj_func,line_count)
#             else:
#                 Total.append(segm,fr_size,fr_rate,model,ET,F1,None,obj_func,line_count)#,isfraction=False)
            
#     rfp.close()

def find_config(configs,segm,framesize,framerate,model):
    for conf in configs[segm]:#range(len(Total[segm])):        
        if conf[0]==framesize and conf[1]==framerate and conf[2]==model:
            return conf#l_ET[segm][i],l_F1[segm][i]
    return None
#########################################################################################
################################ Read Top Configs #######################################
#########################################################################################
fp_key = open('/home/mgharasu/Videos/traffic_lstm/config files/top_configs_1.txt.newObj','r')
top_configs = list()
segm = -1
num_read_top_configs = 0
for line in fp_key.readlines():
    if "###" in line:
        segm += 1
        top_configs.append(list())
        num_read_top_configs = 0
        continue
    
    
    # if num_read_top_configs == 10:
    #     continue
    num_read_top_configs += 1

    tmp = line.split(',')
    seg_n = tmp[1][1:]
    frSize = tmp[2][2:-1]
    frRate = tmp[3][2:-1]
    mdl = tmp[4][2:-1]
    et = tmp[5][1:]
    F1 = tmp[6][1:]
    frac = tmp[7][1:]
    objf = tmp[8][1:-2]
    top_configs[segm].append([frSize,frRate,mdl,et,F1,frac,objf])


def fps_list(start,step,segment):
            fps = list()
            counter = start
            while counter % segment != 0 or counter == start:
                fps.append(counter)
                counter+=step
            return fps

def find_config_winner(segm, prev_sim_segm):
    framesize = np.zeros(5)
    framesize_w = np.zeros(5)
    framerate = np.zeros(5)
    framerate_w = np.zeros(5)
    model = np.zeros(4)
    model_w = np.zeros(4)
    # close_segment = top_configs[similar_segm_idx]
    # for i in range(len(close_segment)):        
    #     if i<1:#segm>close_segments[i]:#
    # prev_segm = close_segments[i]
    prev_top = top_configs[prev_sim_segm][:10]
    # curr_top = top_configs[segm]
    for j in range(len(prev_top)):
        frsz = prev_top[j][0]
        frrt = prev_top[j][1]
        mdl  = prev_top[j][2]
        framesize[sizeidx[str(frsz)]]+=1
        framesize_w[sizeidx[str(frsz)]]+=(100-j)
        framerate[rateidx[str(frrt)]]+=1
        framerate_w[rateidx[str(frrt)]]+=(100-j)
        model[modelidx[mdl]]+=1
        model_w[modelidx[mdl]]+=(100-j)
        # if j==10:
        #     break
    #find max voting configs
    # print(segm,"  sim to ", prev_segm)
    vote_framesize = np.argsort(framesize_w)[::-1]
    vote_framerate = np.argsort(framerate_w)[::-1]
    vote_model = np.argsort(model_w)[::-1]    
    winner_framesize1 = list(sizeidx.keys())[list(sizeidx.values()).index(vote_framesize[0])]
    winner_framerate1 = list(rateidx.keys())[list(rateidx.values()).index(vote_framerate[0])]
    winner_model1 = list(modelidx.keys())[list(modelidx.values()).index(vote_model[0])]
    found_config = find_config(top_configs,segm,winner_framesize1, winner_framerate1, winner_model1)
    return found_config
    
def winner_segment(video,segment,segment_number):
    a=0
    start, end = segment
    imgs=list()

    fs = fps_list(start,5,30)
    
    # for f in fs:
    #     imgs.append(video.get_frame_position_PIL(f)[1])
    
    # output = seq2vec_readSource(fs,video)
    output = seq2vec_readSource(segment_number)
    # reading dataset and find closest
    
    min_dist = 0
    max_sim=0
    first_sim = 0
    second_sim = 0
    sim_l = []
    os1=[]
    # pdist = torch.nn.PairwiseDistance(p=2)
    # cos = torch.nn.CosineSimilarity(dim=1,eps=1e-6)
    for i in range(len(dataset[:143])):
        #Euclidean distance, chebyshev, canberra, cosine
        sim = distance.cdist(output.view(1,-1).cpu().data.numpy(), dataset[i].view(1,-1).cpu().data.numpy(),'cityblock')
        # same as torch.cosine_similarity =>sim = distance_matrix(output.view(1,-1).cpu().data.numpy(), dataset[i].view(1,-1).cpu().data.numpy())#
        # sim = torch.pairwise_distance(output.view(1,-1), dataset[i].view(1,-1))
        old_sim = torch.cosine_similarity(output.view(1,-1), dataset[i].view(1,-1) ).item()
        sim_l.append(sim[0][0])
        os1.append(old_sim)
        # if max_sim<sim:
        #     second_sim = first_sim
        #     first_sim = i
        #     max_sim = sim
    order = np.array(sim_l).argsort()#[::-1]
    ord2=np.array(os1).argsort()[::-1]
    return order[0], order[1]#first_sim, second_sim

# def creating dataset()