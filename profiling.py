
class C_PROFILING:
    def __init__(self, video, frame_number, number_of_frame_for_profiling):
        self.video = video
        # self.frame_number = frame_number
        # self.number_frames_for_profiling = number_of_frame_for_profiling
    
    '''
        segment = the format is [start_frame, end_frame]. it reads frame numbers from start_frame to end_frame
        configs = knobs
        top_k = returining k configurations that are the best 
    '''
    def profiling(self, segment, configs, top_k):
        start_frame, end_frame = segment[0], segment[1]
        ret, frame, start_frame = self.video.get_frame_position(start_frame, end_frame)
        while ret:
            # process frame
