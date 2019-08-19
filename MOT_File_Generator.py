
# version 2
class C_MOT_OUTPUT_GENERATER:
    def __init__(self, outputfile):
        # self.__outputfile = outputfile
        self.__fp = open(outputfile, 'w')
    
    def write(self, frame_number, MOT_data_per_frame):
        for data in MOT_data_per_frame:            
            self.__fp.write(str(frame_number)+' '+str(int(round(data[0])))+' '+str(int(round(data[1])))+' '+ str(int(round(data[2])))+' '+str(int(round(data[3])))+' '+str(int(round(data[4])))+' -1 -1 -1 -1\n')
    
    def close(self):
        self.__fp.close()