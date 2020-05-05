"""
# Here I consider to use boxplot and do frame-based process instead of segment-based
# Feb 5, 2020

import numpy as np
import matplotlib.pyplot as plt
fp = open('info_boxplot_v1.txt.back', 'r')
#============================================================== Number of Objects is X-Axis ===========================================
# num_obj , time , F1 , avg_area , cpu_avg , mem_avg 
num_objs = []
time = []
F1 = []
avg_area = []
cpu = []
mem = []
det = []
for line in fp.readlines():
    v = line.split(',')
    
    num_objs.append(int(v[0]))
    time.append(float(v[1]))
    F1.append(float(v[2]))
    avg_area.append(float(v[3]))
    cpu.append(float(v[4]))
    mem.append(float(v[5]))
    det.append(v[6])

indces = [i for i, data in enumerate(det) if 'tr' in data]#np.where(det=='tr\n')
# num_objs = np.array(num_objs)
num_objs = np.array(num_objs)[indces]
time = np.array(time)[indces]
F1 = np.array(F1)[indces]
avg_area = np.array(avg_area)[indces]
cpu = np.array(cpu)[indces]
mem = np.array(mem)[indces]

#cpu process
for i in range(len(cpu)):
        if i>0 and cpu[i]==0:
                cpu[i] = cpu[i-1]
        
# cpu = [cpu[i-1] for i in range(len(cpu)) if cpu[i]==0]

max_v = max(num_objs)
data = [[] for i in range(max_v)]

for i in range(max_v):
    idces = [j for j in range(len(num_objs)) if num_objs[j]==i]
    if len(idces)>0:
        data[i].append([time[i] for i in idces])

# x = np.arange(max_v)
# for i in range(max_v):
#     if len(data[i])>0:
#         x= np.ones(len(data[i][0]))*i
        # plt.boxplot(x,data[i][0])
plt.boxplot(data)
plt.xlabel("Number of objects")
plt.ylabel("time (second)")
plt.title("video 1-Tracker")
plt.show()
a = 0
"""
#============================================================== Average Area is X-Axis ===========================================

# import numpy as np
# import matplotlib.pyplot as plt
# # fp = open('info_boxplot_v1.txt', 'r')
# # num_obj , time , F1 , avg_area , cpu_avg , mem_avg 
# num_objs = []
# time = []
# F1 = []
# avg_area = []
# cpu = []
# mem = []
# det = []
# lines = []
# counter =0
# fp=open('info_boxplot_v1_revision.txt','r')
# # for line in fp.readlines():
# #         if counter%2!=0:
# #                 fp2.write(line)
# #         counter += 1
# # fp2.close()
# for line in fp.readlines():
#     v = line.split(',')
    
#     num_objs.append(int(v[0]))
#     time.append(float(v[1]))
#     F1.append(float(v[2]))
#     avg_area.append(float(v[3]))
#     cpu.append(float(v[4]))
#     mem.append(float(v[5]))
#     det.append(v[6])

# indces = [i for i, data in enumerate(det) if 'tr' in data]#np.where(det=='tr\n')

# num_objs = np.array(num_objs)[indces]
# time = np.array(time)[indces]
# F1 = np.array(F1)[indces]
# avg_area = np.array(avg_area)[indces]
# cpu = np.array(cpu)[indces]
# mem = np.array(mem)[indces]

# minArea = np.min(avg_area)
# maxArea = np.max(avg_area)
# print("min:", minArea, ", max: ", maxArea, ", diff:", maxArea-minArea)

# data = [[] for i in range(247)]

# for i in range(len(avg_area)):
#     data[int(avg_area[i]/1000)].append(mem[i])
# a=0
# plt.boxplot(data)
# plt.xlabel("Mean of average area per frame-categorization based on 1000 pixels")
# plt.ylabel("mem (percent)")
# plt.title("video 1-Tracker")
# plt.show()

# ==================================== avergae area versus memory usage (revised  memory allocation observation) ========================
"""
import numpy as np
import matplotlib.pyplot as plt
# fp = open('info_boxplot_v1.txt', 'r')
# num_obj , time , F1 , avg_area , cpu_avg , mem_avg 
num_objs = []
time = []
F1 = []
avg_area = []
cpu = []
tracker_mem = []
parent_mem = []
child_mem = []
det = []
lines = []
counter =0
fp=open('info_boxplot_v1_revision.txt','r')
# for line in fp.readlines():
#         if counter%2!=0:
#                 fp2.write(line)
#         counter += 1
# fp2.close()
for line in fp.readlines():
    v = line.split(',')
    
    num_objs.append(int(v[0]))
    time.append(float(v[1]))
    F1.append(float(v[2]))
    avg_area.append(float(v[3]))
    cpu.append(float(v[4]))
    tracker_mem.append(int(v[5]))
    parent_mem.append(int(v[6]))
    child_mem.append(int(v[7]))
    det.append(v[8])

indces = [i for i, data in enumerate(det) if 'obj' in data]#np.where(det=='tr\n')

num_objs = np.array(num_objs)[indces]
time = np.array(time)[indces]
F1 = np.array(F1)[indces]
avg_area = np.array(avg_area)[indces]
cpu = np.array(cpu)[indces]
tracker_mem = np.array(tracker_mem)[indces]
parent_mem = np.array(parent_mem)[indces]
child_mem = np.array(child_mem)[indces]

minArea = np.min(avg_area)
maxArea = np.max(avg_area)
print("min:", minArea, ", max: ", maxArea, ", diff:", maxArea-minArea)

data = [[] for i in range(247)]

for i in range(len(avg_area)):
    data[int(avg_area[i]/1000)].append(tracker_mem[i]+child_mem[i])
a=0
plt.boxplot(data)
plt.xlabel("Mean of average area per frame-categorization based on 1000 pixels")
plt.ylabel("Memory Usage of tracker and child process (Byte)")
plt.title("video 1-Memory usage")
plt.show()
"""
# ==================================== number of objects versus memory usage (revised  memory allocation observation) ========================
import numpy as np
import matplotlib.pyplot as plt
# fp = open('info_boxplot_v1.txt', 'r')
# num_obj , time , F1 , avg_area , cpu_avg , mem_avg 
num_objs = []
time = []
F1 = []
avg_area = []
cpu = []
tracker_mem = []
parent_mem = []
child_mem = []
det = []
lines = []
counter =0
fp=open('info_boxplot_v1_revision.txt','r')
# for line in fp.readlines():
#         if counter%2!=0:
#                 fp2.write(line)
#         counter += 1
# fp2.close()
for line in fp.readlines():
    v = line.split(',')
    
    num_objs.append(int(v[0]))
    time.append(float(v[1]))
    F1.append(float(v[2]))
    avg_area.append(float(v[3]))
    cpu.append(float(v[4]))
    tracker_mem.append(int(v[5]))
    parent_mem.append(int(v[6]))
    child_mem.append(int(v[7]))
    det.append(v[8])

indces = [i for i, data in enumerate(det) if 'obj' in data]#np.where(det=='tr\n')

num_objs = np.array(num_objs)[indces]
time = np.array(time)[indces]
F1 = np.array(F1)[indces]
avg_area = np.array(avg_area)[indces]
cpu = np.array(cpu)[indces]
tracker_mem = np.array(tracker_mem)[indces]
parent_mem = np.array(parent_mem)[indces]
child_mem = np.array(child_mem)[indces]

max_v = max(num_objs)
data = [[] for i in range(max_v)]

for i in range(max_v+1):
    idces = [j for j in range(len(num_objs)) if num_objs[j]==i]
    if len(idces)>0:
        data[i].append([child_mem[i] for i in idces])
        if i== 14:
                plt.boxplot(data)
                plt.xlabel("Number of objects")
                plt.ylabel("time (second)")
                plt.title("video 1-Tracker")
                plt.show()
# x = np.arange(max_v)
# for i in range(max_v):
#     if len(data[i])>0:
#         x= np.ones(len(data[i][0]))*i
        # plt.boxplot(x,data[i][0])
