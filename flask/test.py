import json
import numpy as np
class state:
    NO_REP = 1
    IN_REP = 2
    COOLDOWN = 3


with open('test.json', 'r') as f:
    data = json.load(f)

initial = globals().copy()
print initial
globals().update(data)


in_time = 0
out_time = 0
cooldown_in_time = 0
cooldown_out_time = 0
frame_rate = 15
#num_of_vids = 25

#set variables

#global in_time, out_time, cooldown_in_time, cooldown_out_time
#global global_counter, winner_stride, cur_state, in_frame_num, actions_counter
med_out_label = 0
global_counter = 0
winner_stride = 7
in_frame_num = 1
actions_counter = 0
cur_state = state.NO_REP

glob = globals().copy()
for key,value in initial.items() :
	del glob[key]
del glob['initial']
#del glob['variables']
#variables.update(glob)
#del glob['std_arr']
print glob

obj = glob
filename='global'
with open(filename+'.json','w') as outfile:
    json.dump(obj,outfile,ensure_ascii=False)#,encoding='utf-8')
    outfile.write('\n')