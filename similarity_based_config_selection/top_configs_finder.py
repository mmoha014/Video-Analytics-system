"""
number='2'
rfp = open('records/new_gt'+number+'_detectron2_960.txt.record','r')
wfp = open('tmp_gt'+number+'_detectron2_960_record.txt','w')
line_count = 0 
for line in rfp.readlines():
    line_count += 1
    
    if line.find(',,') is not -1:
        tmp1=line.split(',,')
        split=tmp1[0].split('[')        
        t1=split[1].split(',')
        # t2 = tmp1[1].split(',')
        wfp.write(split[0]+','+t1[0]+','+t1[1][1:]+','+t1[2][2:-2]+','+tmp1[1])#tmp1[0]+','+tmp2[0][1:]+','+tmp2[1]+','+tmp2[2][2:-2]+','+tmp2[3]+','+tmp2[4])
    else:
        tmp1=line.split(',')
        # tmp2=tmp1[1].split(',')
        wfp.write(tmp1[0]+','+tmp1[1][1:]+','+tmp1[2][1:]+','+tmp1[3][2:-2]+','+tmp1[4]+','+tmp1[5])#tmp1[0]+','+tmp2[0]+','+tmp2[1][1:]+','+tmp2[2][2:-2]+','+tmp2[4]+','+tmp2[5])
    
    # if line.find(',,') is not -1:        
    #     tmp1=line.split(',,')
    #     tmp2 = tmp1[1][:-2]
    #     tmp1 = tmp1[0].split(',')
    #     tmp2 = tmp2.split(',')
    #     wfp.write(tmp1[0]+','+tmp1[1]+','+tmp1[2][1:]+','+tmp1[3][2:-1]+','+tmp2[0]+','+tmp2[1][1:]+','+tmp2[2][1:]+'\n')
    # else:
    #     tmp1=line.split(',')
    #     # tmp2 = tmp1[1][:-2]
    #     # tmp1 = tmp1[0].split(',')
    #     # tmp2 = tmp2.split(',')
    #     # wfp.write(tmp1[0]+','+tmp1[1]+','+tmp1[2][1:]+','+tmp1[3][2:-1]+','+tmp1[5]+','+tmp2[0]+','+tmp2[1][1:]+'\n')
    #     wfp.write(tmp1[0]+','+tmp1[1]+','+tmp1[2][1:]+','+tmp1[3][2:-1]+','+tmp1[4]+','+tmp1[5][1:]+','+tmp1[6][1:-2]+'\n')
    

rfp.close()
wfp.close()
"""

##################### Definition of Objective Function #################################

import numpy as np
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
    print(line)
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

def topFinder_main():
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
                obj_func = F1#(0.8*(7-ET))+(.2*7*F1)
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

    sorted_configs = list()
    for i in range(len(l_of)):
        sorted_configs.append(sorted(Total[i],key=lambda x:(x[8],x[5]))[::-1])
        # sorted_configs.append(sorted(Total[i],key=lambda x:(x[8]))[::-1])

    # sorted_configs = sorted_configs[::-1]
    f = open('/home/morteza/Videos/traffic camera/profile/youtube/5_top_configs/top_configs(of=F1&ET<0)/top_configs_youtube'+str(number)+'_F1.txt.test2','w')#'/home/morteza/Videos/traffic camera/profile/CAM822_Virginia/profile_on_all_configs/top_configs(of=F1)/top_configs_gt'+str(number)+'.txt','w')
    for i in range(len(sorted_configs)):
        f.write('################ segment '+str(i)+' ###############\n')
        for j in sorted_configs[i]:
            f.write(str(j)+'\n')
    f.close()
    # sort_idx = list()
    # for i in range(len(l_of)):
    #     sort_idx.append(np.argsort(l_of[i],key=lambda x:(x[4],x[5])))

    # l =  [list() for k in range(len(l_ET))]
    # segm_count = -1
    # for i in range(len(l_of)):
    #     for j in range(len(l_of[i])):
    #         idx = sort_idx[i][j]
    #         l[i].append([l_line[i][idx],l_seg[i][idx], l_fr_size[i][idx],l_rate[i][idx],l_model[i][idx],l_ET[i][idx],l_F1[i][idx],l_fraction[i][idx],l_of[i][idx]])

    # f = open('/home/morteza/Videos/traffic camera/profile/youtube/5_top_configs/top_configs(of=F1&ET<0)/top_configs_gt'+str(number)+'.txt.test','w')#'/home/morteza/Videos/traffic camera/profile/CAM822_Virginia/profile_on_all_configs/top_configs(of=F1)/top_configs_gt'+str(number)+'.txt','w')
    # for i in range(len(l)):
    #     f.write('################ segment '+str(i)+' ###############\n')
    #     f.write(str(l[i][-1])+'\n')
    #     f.write(str(l[i][-2])+'\n')
    #     f.write(str(l[i][-3])+'\n')
    #     f.write(str(l[i][-4])+'\n')
    #     f.write(str(l[i][-5])+'\n')

    f.close()

if if __name__ == "__main__":
    topFinder_main()
