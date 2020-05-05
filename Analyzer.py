# #!/usr/bin/env python

# # Matthieu Brucher
# # Last Change : 2008-11-19 19:05

# import subprocess
# import threading
# import datetime

# names = [("pid", int),
#          ("comm", str),
#          ("state", str),
#          ("ppid", int),
#          ("pgrp", int),
#          ("session", int),
#          ("tty_nr", int),
#          ("tpgid", int),
#          ("flags", int),
#          ("minflt", int),
#          ("cminflt", int),
#          ("majflt", int),
#          ("cmajflt", int),
#          ("utime", int),
#          ("stime", int),
#          ("cutime", int),
#          ("cstime", int),
#          ("priority", int),
#          ("nice", int),
#          ("0", int),
#          ("itrealvalue", int),
#          ("starttime", int),
#          ("vsize", int),
#          ("rss", int),
#          ("rlim", int),
#          ("startcode", int),
#          ("endcode", int),
#          ("startstack", int),
#          ("kstkesp", int),
#          ("kstkeip", int),
#          ("signal", int),
#          ("blocked", int),
#          ("sigignore", int),
#          ("sigcatch", int),
#          ("wchan", int),
#          ("nswap", int),
#          ("cnswap", int),
#          ("exit_signal", int),
#          ("processor", int),]

# colours = ['b', 'g', 'r', 'c', 'm', 'y']

# def getPageSize():
#   import resource
#   f = open("/proc/meminfo")
#   mem = f.readline()
#   f.close()
#   return resource.getpagesize() / (1024 * float(mem[10:-3].strip()))

# pagesizepercent = getPageSize()

# def collectData(pid, task):
#   """
#   Collect process list
#   """
#   f1 = open("/proc/%d/task/%s/stat"%(pid,task))
#   f2 = open("/proc/%d/task/%s/statm"%(pid,task))
#   t = datetime.datetime.now()
#   stat = f1.readline().split()
#   mem = f2.readline().split()
#   d = dict([(name[0], name[1](el)) for (name, el) in zip(names, stat)])
#   d["pmem"] = 100 * float(mem[1]) * pagesizepercent
#   return t, d

# def getTime(key):
#   """
#   Returns the time in microseconds
#   """
#   return (((key.weekday() * 24 + key.hour) * 60 + key.minute) * 60 + key.second) * 1000000 + key.microsecond
  
# class MonitorThread(threading.Thread):
#   """
#   The monitor thread saves the process info every 5 seconds
#   """
#   def __init__(self, pid):
#     import collections

#     self.pid = pid
#     threading.Thread.__init__(self)
#     self.data = collections.defaultdict(dict)
#     self.process = True
    
#   def run(self):
#     import os
#     import time

#     while self.process:
#       threads = os.listdir("/proc/%d/task/" % self.pid)
#       for thread in threads:
#         t, d = collectData(self.pid, thread)
#         d["current_time"] = t
        
#         if "now" in self.data[thread]:
#           now = self.data[thread]["now"]
#           d['pcpu'] = 1e6 * ((d['utime'] + d['stime']) - (now['utime'] + now['stime'])) / float((getTime(t) - getTime(now["current_time"])))

#         self.data[thread][getTime(t)] = d
#         self.data[thread]["now"] = d
#       time.sleep(1)

# def displayCPU(data, pid):
#   """
#   Displays and saves the graph
#   """
#   import pylab
#   import numpy
  
#   spid = str(pid)
  
#   c = 0
#   threads = data.keys()
#   threads.sort()
#   for thread in threads:
#     d = data[thread]
#     keys = d.keys()
#     keys.remove("now")
#     keys.sort()
#     mykeys = numpy.array(keys)/1e6
#     mykeys -= mykeys[0]
  
#     pylab.plot(mykeys[2:], [d[key]['pcpu'] for key in keys[2:]], colours[c], label = thread)
#     c = c+1
#     if spid == thread:
#       pylab.plot(mykeys[2:], [d[key]['pmem'] for key in keys[2:]], 'k', label = 'MEM')

#   pylab.ylim([-5, 105])
#   pylab.legend(loc=6)
  
#   pylab.savefig('%d.svg' % pid)
#   pylab.savefig('%d.png' % pid)
#   pylab.close()

# if __name__ == "__main__":
#   import sys
#   import os
#   import pickle
  
#   stdin = open(sys.argv[1])
#   stdout = open(sys.argv[2], "w")
  
#   process = subprocess.Popen(sys.argv[3:], stdin = stdin, stdout = stdout)
  
#   thread = MonitorThread(process.pid)
#   thread.start()

#   process.wait()

#   thread.process = False
#   thread.join()
  
#   f = open('%d.data' % process.pid, 'w')
#   pickle.dump(thread.data, f)
  
#   try:
#     displayCPU(thread.data, process.pid)
#   except:
#     pass
# ========================================== check memory allocation behavior in python 
# import numpy as np
# import sys
# import copy
# def get_size(obj, seen=None):
#     """Recursively finds size of objects"""
#     size = sys.getsizeof(obj)
#     if seen is None:
#         seen = set()
#     obj_id = id(obj)
#     if obj_id in seen:
#         return 0
#     # Important mark as seen *before* entering recursion to gracefully handle
#     # self-referential objects
#     seen.add(obj_id)
#     if isinstance(obj, dict):
#         size += sum([get_size(v, seen) for v in obj.values()])
#         size += sum([get_size(k, seen) for k in obj.keys()])
#     elif hasattr(obj, '__dict__'):
#         size += get_size(obj.__dict__, seen)
#     elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
#         size += sum([get_size(i, seen) for i in obj])
#     return size

# class checkmemory():
#     def __init__(self):
#         self.__a = []
#     def add(self,x):
#         for i in x:
#             self.__a.append(i)
    
#     def get_size(self):
#         return sys.getsizeof(self.__a)

# obj = checkmemory()
# obj.add([1,2,3,4,5,6,7,8,9])
# l=[]
# l.append(obj)
# print("size: ", sys.getsizeof(l))

# l.append(copy.deepcopy(obj))
# print("size: ", sys.getsizeof(l))

# obj2=checkmemory()
# obj2.add([1,2,3,4,5,6,7,8,99,0,1,2,3,4,5,6,7,8,9,0])
# print(sys.getsizeof(obj2))
# l.append(obj2)
# print('list size: ', sys.getsizeof(l))
# l.append(obj2)
# print('list size: ', sys.getsizeof(l))
# l.append(obj2)
# print('list size: ', sys.getsizeof(l))
#============================================== check size of tracker class ==================================
from tracking import C_TRACKER
from setting import *
import cv2
import sys

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


frame = cv2.imread('/home/mgharasu/Videos/traffic camera/5/1.jpg')
tracker = C_TRACKER(TRACKER_TYPE)
print(get_size(tracker))
tracker.Add_Tracker(frame, [[1,2,3,4],[67,89,12,111], [45,90,54,70],[66,77,100,100]])
print(get_size(tracker))