import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#read top_configs1.txt
fp_key = open('/home/morteza/Videos/traffic camera/profile/youtube/5_top_configs/top_configs(of=0.8*ET+0.2*F1)/top_configs1.txt','r')
top_configs = list()
segm = -1

for line in fp_key.readlines():
    if "###" in line:
        segm += 1
        top_configs.append(list())
        continue

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

def match_top_config(segm,size,rate,model):
    for i in range(len(top_configs[segm])):
        if top_configs[segm][i][0]==size and top_configs[segm][i][1]==rate and top_configs[segm][i][2]==model:
            return top_configs[segm][i]
    return None


#read keyframe_categorization
fp = open('/home/morteza/Videos/traffic camera/profile/youtube/keyframe_categorization.txt','r')
segm = -1
keyframe_categorization = list()
for line in fp.readlines():
    if segm == -1:
        segm +=1
        # similar_segs.append(list())
        continue
    tmp = line.split(' ')
    if int(tmp[0][3:-1])>142:
        break
    data = list()
    for j in tmp[1:]:
        if (j!='' and j!='['):
            if '\n' in j:
                data.append(int(j[:-2]))
            elif '[' in j:
                data.append(int(j[1:]))
            else:
                data.append(int(j))
            
    # seg_n = tmp[0][:-1]
    
    keyframe_categorization.append(data)#[int(tmp[2]),int(tmp[4]),int(tmp[7]),int(tmp[10]),int(tmp[13]),int(tmp[15]),int(tmp[18]),int(tmp[20]),int(tmp[22]),int(tmp[24]),int(tmp[25][:-2])])
a=0
#3 read new_gt1_detectron2.txt.record.check

l_seg = list()
l_fr_size = list()
l_rate = list()
l_model = list()
l_ET = list()
l_F1 = list()
l_fraction = list()
l_of = list()
l_line = list()
Total = list()

def add_new_entry():
    l_seg.append(list())
    l_fr_size.append(list())
    l_rate.append(list())
    l_model.append(list())
    l_ET.append(list())
    l_F1.append(list())
    l_fraction.append(list())
    l_of.append(list())
    l_line.append(list())
    Total.append(list())

def append2lists(segm,fr_size,fr_rate,model,et,f1,frac,of,line):
    # print(line)
    l_seg[segm].append(segm)
    l_fr_size[segm].append(fr_size)
    l_rate[segm].append(fr_rate)
    l_model[segm].append(model)
    l_ET[segm].append(et)
    l_F1[segm].append(f1)
    # if frac is not None:
    #     l_fraction[segm].append(frac)
    # else:
    l_fraction[segm].append(frac)
    l_of[segm].append(of)
    l_line[segm].append(line)
    Total[segm].append([line,segm,fr_size,fr_rate,model,et,f1,frac,of])



number = '1'
segm_count = -1
line_count = 0
# rfp = open('gt'+number+'_detectron2_960_record.txt','r')
rfp = open('/home/morteza/Videos/traffic camera/profile/youtube/profile_on_all_configs/new_gt'+number+'_detectron2_960.txt.record.check')#'/home/morteza/Videos/traffic camera/profile/CAM822_Virginia/profile_on_all_configs/gt_'+number+'.txt.record.check')#'gt'+str(number)+'_profile.txt','r')
for line in  rfp.readlines():
    line_count += 1
    # if line_count == 12:
    #     a=0
    tmp = line.split(',')
    segm = np.int16(tmp[0])    
    fr_size = tmp[1]
    fr_rate = tmp[2]
    model = tmp[3]
    if True:#int(number)>2:
        ET = np.float(tmp[-3])
        F1 = np.float(tmp[-2])
        fraction = np.float(tmp[-1][:-1])
    else:
        ET = np.float(tmp[-2])
        F1 = np.float(tmp[-1])
    if True: #fraction>=0.8 and ET<=1.0:
        if True:#ET<=1.0:
            obj_func = (0.8*(7-ET))+(.2*7*F1)
        else:
            obj_func = 0.0
        # print('line:',line_count,', of: ',obj_func)
        if segm_count<segm:
            add_new_entry()
            segm_count+=1
        if True:#int(number)>2:
            append2lists(segm,fr_size,fr_rate,model,ET,F1,fraction,obj_func,line_count)
        else:
            append2lists(segm,fr_size,fr_rate,model,ET,F1,None,obj_func,line_count)#,isfraction=False)
        
rfp.close()

def find_config(segm,framesize,framerate,model):
    for i in range(len(Total[segm])):        
        if l_fr_size[segm][i]==framesize and l_rate[segm][i]==framerate and l_model[segm][i]==model:
            return Total[segm][i][2:]#l_ET[segm][i],l_F1[segm][i]
        
    return None

################## voting ##################
sizeidx ={'480':0, '600':1, '720':2, '840':3, '960':4}
rateidx = {'1':0, '2':1,'5':2,'10':3,'30':4}
modelidx = {'faster_rcnn_X_101_32x8d_FPN_3x':0,'faster_rcnn_R_101_DC5_3x':1,'retinanet_R_101_FPN_3x':2, 'faster_rcnn_R_50_FPN_3x':3}
framesize = np.zeros(5)
framerate = np.zeros(5)
model = np.zeros(4)
et1=list()
et2=list()
opt_et = list()
f1_1 = list()
f1_2 = list()
opt_f1 = list()
for segm in range(143):
    close_frames = keyframe_categorization[segm]
    # framesize['480']=framesize['600']=framesize['600']=framesize['720']=framesize['840']=framesize['960']=0
    # framerate['1']=framerate['2']=framerate['5']=framerate['10']=framerate['30'] = 0
    # model['faster_rcnn_X_101_32x8d_FPN_3x']=model['faster_rcnn_R_101_DC5_3x']=model['retinanet_R_101_FPN_3x']=model['faster_rcnn_R_50_FPN_3x'] = 0
    framesize = np.zeros(5)
    framerate = np.zeros(5)
    model = np.zeros(4)
    for i in range(len(close_frames)):        
        if True:#segm>close_frames[i]:
            prev_segm = close_frames[i]
            prev_top = top_configs[prev_segm]
            curr_top = top_configs[segm]
            for j in range(len(prev_top)):
                frsz = prev_top[j][0]
                frrt = prev_top[j][1]
                mdl  = prev_top[j][2]
                framesize[sizeidx[str(frsz)]]+=1
                framerate[rateidx[str(frrt)]]+=1
                model[modelidx[mdl]]+=1
            #find max voting configs
            vote_framesize = np.argsort(framesize)[::-1]
            vote_framerate = np.argsort(framerate)[::-1]
            vote_model = np.argsort(model)[::-1]
            winner_framesize1 = list(sizeidx.keys())[list(sizeidx.values()).index(vote_framesize[0])]
            winner_framerate1 = list(rateidx.keys())[list(rateidx.values()).index(vote_framerate[0])]
            winner_model1 = list(modelidx.keys())[list(modelidx.values()).index(vote_model[0])]                       
            
            top_c1 = match_top_config(segm,winner_framesize1,winner_framerate1,winner_model1)
            
            optimal_et,optimal_f1 = top_configs[segm][0][3],top_configs[segm][0][4]
            opt_et.append(float(optimal_et))
            opt_f1.append(float(optimal_f1))
            if top_c1 is None: #if the winner config of the previous segment is not among top configs in the current segment, find the winner config in the total configs of the current segment
                top_c1 = find_config(prev_segm,winner_framesize1,winner_framerate1,winner_model1)
            
            c1_et,c1_f1 = top_c1[3],top_c1[4]
            et1.append(float(c1_et))
            f1_1.append(float(c1_f1))
            
            # ##### new policy ####
            # winner_model2 = winner_model1
            # winner_framerate2 = winner_framerate1
            # winner_framesize2 = winner_framesize1
            # if vote_model[0]==vote_model[1] or model[vote_model[1]]==0:
            #     if vote_framerate[0]==vote_framerate[1] or framerate[vote_framerate[1]]==0:
            #         winner_framesize2 = list(sizeidx.keys())[list(sizeidx.values()).index(vote_framesize[1])]
            #     else:
            #         winner_framerate2 = list(rateidx.keys())[list(rateidx.values()).index(vote_framerate[1])]
            # else:
            #     winner_model2 = list(modelidx.keys())[list(modelidx.values()).index(vote_model[1])]
            
            # top_c2 = match_top_config(segm,winner_framesize2,winner_framerate2,winner_model2)
            # if top_c2 is None:
            #     top_c2 = find_config(segm,winner_framesize2,winner_framerate2,winner_model2)
            # c2_et,c2_f1 =top_c2[3],top_c2[4]
            # et2.append(float(c2_et))
            # f1_2.append(float(c2_f1))
                    
            if True:#vote_model[0] == vote_model[1]:# if two models among top configs of the previous segment have the same frequency, look at the second model that is winner
                if vote_model[1]>0:
                    winner_model2 = list(modelidx.keys())[list(modelidx.values()).index(vote_model[1])]
                else:
                    winner_model2 = winner_model1[:]
                top_c2 = match_top_config(segm,winner_framesize1,winner_framerate1,winner_model2)
                if top_c2 is None:
                    top_c2 = find_config(segm,winner_framesize1,winner_framerate1,winner_model2)
                c2_et,c2_f1 =top_c2[3],top_c2[4]
                et2.append(float(c2_et))
                f1_2.append(float(c2_f1))
            break
                        
# Currently, I have three numbers for F1: optimal_f1, c1_f1, c2_f1
# and there is three numbers for execution time: optimal_et, c1_et, c2_et
# data = {'optimal':np.array(opt_f1),'first_top_config':np.array(f1_1),'second_top_config':np.array(f1_2)}
# df4 = pd.DataFrame.from_dict(data).T
# cols = list()
# for i in range(len(f1_1)):
#     cols.append('segm_'+str(i))
# df4.columns=cols#['optimal','first_top_config','second_top_config']
# plt.figure()
# df4.plot.hist(alpha = 0.5) 
# plt.show()
# x=np.arange(139)
# df = pd.DataFrame(zip(x*3, ["optimalF1"]*3+["firstTopF1"]*3+["secTopF1"]*3),columns=["segment","data","F1"])
# plt.figure()
# import seaborn as sns# seaborn import sns
# sns.barplot(x="segment", hue="data", y="F1", data=df)
# plt.show()
def F1_statistics(f1_1,opt_f1):
    eighty = 0
    ninety = 0
    under8 = 0
    optm = 0

    for i in range(len(f1_1)):
        if f1_1[i]>=0.9 and f1_1[i]<=1.0:
            ninety += 1
        elif f1_1[i]<0.9 and f1_1[i]>=0.8:
            eighty += 1
        elif f1_1[i]<0.8:
            under8 += 1
        
        if f1_1[i]==opt_f1[i]:
            optm+=1
    
    return eighty, ninety, under8, optm


def ET_statistics(et,opt_et):
    def compare(x,y):
        result=x-y
        if abs(result)<0.05:
            return 0
        elif result>0:
            return 1
        else:
            return -1
    bigger = 0
    smaller = 0
    same = 0
    for i in range(len(et)):
        if compare(et[i],opt_et[i])==0:#et[i]==opt_et[i]:
            same += 1
        elif compare(et[i],opt_et[i])>0:#et[i]>opt_et[i]:
            bigger += 1
        else:
            smaller += 1
    return bigger, smaller, same

bc8,bc9,bcu8,bcop = F1_statistics(f1_1,opt_f1)
sc8,sc9,scu8,scop = F1_statistics(f1_2,opt_f1)

bcbggr, bcsml, bcsame = ET_statistics(et1,opt_et)
scbggr, scsml, scsame = ET_statistics(et2,opt_et)




ax=plt.subplot(111)
a=[opt_f1,f1_1,f1_2]
ax.boxplot([opt_f1,f1_1,f1_2])# names=c('Optimal configs','First winner configs','Second winner configs'))
ax.set_xticklabels(['Optimal configs','First winner configs','Second winner configs'])
ax.set_ylabel('F1 Score')
ax.set_title('distribution the number of F1 scores')

plt.show()


ax=plt.subplot(111)
x=np.arange(143)#range(0,140)
# for i in x:
#     print(i)
ax.bar(x,opt_f1, width=0.8,color='b',align='center',alpha=0.5)#scatter(x,opt_et,color='b',marker='_')#bar(x,opt_et, width=0.8,color='b',align='center')
ax.scatter(x,f1_1, color='y',marker='x')#bar(x,et1, width=0.5, color='y', align='center')#, edgecolor='white')
ax.scatter(x,f1_2, color='r',marker='_')#bar(x,et2, width=0.1, color='r',align='center')#, edgecolor='white')#,alpha=0.5)
ax.set_xlabel("segment number")
ax.set_ylabel("F1 Score")#"Execution Time (Sec)")#
ax.set_xticks(x,['seg'+str(i) for i in x])
ax.set_title('choosing config frame the most similar previous segment in one video')
ax.legend(['first winner config','second winner config','Optimal config '],loc=2)
plt.show()



                
            