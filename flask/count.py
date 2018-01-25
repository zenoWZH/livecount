import json
import numpy as np
import sys


class state:
    NO_REP = 1
    IN_REP = 2
    COOLDOWN = 3
data = {}
dic = {}
output_label = []
pYgivenX = []
def softmax(x):
#    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
init = globals().copy()


static_th = 10
norep_std_th = 13
norep_ent_th = 1.0
inrep_std_th = 13
inrep_ent_th = 1.1
lastsix_ent_th = 1.1
history_num = 9

stride_number = 0
rep_count = 0
frame_residue = 0
st_entropy = 0
st_std = 0
std_arr = np.zeros(history_num)
ent_arr = np.zeros(history_num)+2
label_array = np.zeros(history_num)
count_array = np.zeros(history_num)
cur_std = 0
cur_entropy = 0



private = globals().copy()

# global vars
#global in_time, out_time, cooldown_in_time, cooldown_out_time
#global global_counter, winner_stride, cur_state, in_frame_num, actions_counter
initflag = True
in_time = 0
out_time = 0
cooldown_in_time = 0
cooldown_out_time = 0
frame_rate = 15
#num_of_vids = 25

#set variables
med_out_label = 0
global_counter = 0
winner_stride = 0
cur_state = state.NO_REP;
in_frame_num = -1
actions_counter = 0



def load_data():
    global med_out_label,cur_std,stride_number,output_label,pYgivenX
    largest = 0
    for i in range(8):
        pYgivenX.append(data[str(i+3)])
    #pYgivenX = softmax(pYgivenX)
    output_label.append(np.argmax(pYgivenX))
    #for key,value in data.items():
    #    if key!='interval':
    #        if value>=largest :
    #            largest = value
    #            med_out_label = int(key)
    cur_std = data['cur_std']
    stride_number = data['interval']
    #print output_label,cur_std

def save_data(interval,filename,glob):
    #print initial
    #glob = globals().copy()
    for key,value in init.items() :
        del glob[key]
    #loca = locals().copy()
    #print loca
    #for key,value in loca.items() :
    #    del loca[key]
    #del glob['data']
    del glob['cur_std']
    del glob['std_arr']
    del glob['ent_arr']
    del glob['label_array']
    del glob['count_array']
    del glob['med_out_label']
    #del glob['data']
    del glob['loadfile']
    del glob['f']
    del glob['init']
    del glob['variables']
    del glob['save_data']
    del glob['load_data']
    del glob['count']
    del glob['do_local_count']
    del glob['private']
    if 'globalvar' in glob:
        del glob['globalvar']

    if '__warningregistry__' in glob:
        del glob['__warningregistry__']
    print glob
    obj = glob
    #filename='variables'+str(interval)
    with open(filename,'w') as outfile:
        json.dump(obj,outfile,ensure_ascii=False)#,encoding='utf-8')
        outfile.write('\n')
    outfile.close()
    if initflag:
        np.save("std_arr"+str(interval)+".npy",np.zeros(history_num))
        np.save("ent_arr"+str(interval)+".npy",np.zeros(history_num)+2)
        np.save("label_array"+str(interval)+".npy",np.zeros(history_num))
        np.save("count_array"+str(interval)+".npy",np.zeros(history_num))
    else:
        np.save("std_arr"+str(stride_number)+".npy",std_arr)
        np.save("ent_arr"+str(stride_number)+".npy",ent_arr)
        np.save("label_array"+str(stride_number)+".npy",label_array)
        np.save("count_array"+str(stride_number)+".npy",count_array)

    for key,value in private.items() :
        if key in glob:
            del glob[key]
    #print glob
    with open('global.json','w') as outfile:
        json.dump(obj,outfile,ensure_ascii=False)#,encoding='utf-8')
        outfile.write('\n')
    outfile.close()

def do_local_count(initial):
    global cur_entropy,cur_std,med_out_label,label_array,output_label, pYgivenX,rep_count,frame_residue
    #framesArr = get_boundingbox()
        # classify
    #test_set_x.set_value(framesArr, borrow=True)
    #output_label , pYgivenX  = classify(0)

    largest = 0
    for i in range(8):
        pYgivenX.append(data[str(i+3)])
    #pYgivenX = softmax(pYgivenX)
    output_label.append(np.argmax(pYgivenX))
    #for key,value in data.items():
    #    if key!='interval':
    #        if value>=largest :
    #            largest = value
    #            med_out_label = int(key)
    cur_std = data['cur_std']
    stride_number = data['interval']
    print "output_label",output_label,"cur_std",cur_std

    cur_entropy = - (pYgivenX*np.log(pYgivenX)).sum()#-0.2
    # count
    output_label = output_label[0] + 3
    print(output_label,"cur_entropy",cur_entropy)
    #if initflag:
    #    print "init label_array"
    #    label_array = np.arange(output_label,output_label,label_array.shape)
    label_array = np.delete(label_array,0,axis=0)
    label_array = np.insert(label_array, history_num-1 , output_label, axis=0)
        #take median of the last frames
    print("label_array",label_array)
    med_out_label = np.ceil(np.median(label_array[history_num-4:history_num]))
    med_out_label = med_out_label.astype('int32')
    print("med_out_label",med_out_label)

    if initial:
        med_out_label = label_array[history_num-1]
        rep_count = 20 // (med_out_label)
        frame_residue = 20 % (med_out_label)
    else:
        #print(frame_residue)
        frame_residue += 1
        if (frame_residue >= med_out_label):
            rep_count += 1;
            frame_residue = 0;
    print "initial",initial,"rep_count",rep_count,"frame_residue",frame_residue


def count():

        # globals
    global in_time, out_time, cooldown_in_time, cooldown_out_time
    global global_counter, winner_stride, cur_state, in_frame_num, actions_counter
    global stride_number, rep_count, frame_residue, st_entropy, st_std, std_arr, ent_arr, label_array, count_array, cur_std, cur_entropy,cur_state
    detector_strides = [5,7,9]
    # insert new frame
    #frame_set = np.delete(frame_set,0,axis=0)
    #frame_set = np.insert(frame_set, 19 , proFrame, axis=0)

    if (cur_state == state.NO_REP):
        do_local_count(True)
    if ((cur_state == state.IN_REP) and (winner_stride == stride_number)):
        do_local_count(False)
    if (cur_state == state.COOLDOWN):
        do_local_count(True)
    # common to all state
    if (cur_std < static_th):
        cur_entropy = 2
    count_array = np.delete(count_array,0,axis=0)
    count_array = np.insert(count_array, history_num-1 , rep_count, axis=0)
    ent_arr = np.delete(ent_arr,0,axis=0)
    ent_arr = np.insert(ent_arr, history_num-1 , cur_entropy, axis=0)
    std_arr = np.delete(std_arr,0,axis=0)
    std_arr = np.insert(std_arr, history_num-1 , cur_std, axis=0)
    st_std = np.median(std_arr)
    st_entropy = np.median(ent_arr)

    print "count_array",count_array
    print "ent_arr",ent_arr
    print "std_arr",std_arr

    print "cur_state:",cur_state
    print "st_std:",st_std,">norep_std_th=13","st_entropy:",st_entropy,"<norep_ent_th=1.0"
    if (cur_state == state.NO_REP):
        # if we see good condition for rep take the counting and move to rep state
        print "st_std:",st_std,">norep_std_th=13","st_entropy:",st_entropy,"<norep_ent_th=1.0"
        if ((st_std > norep_std_th) and (st_entropy < norep_ent_th)):
            print "start counting!"
            actions_counter += 1
            cur_state = state.IN_REP
            global_counter = rep_count
            winner_stride = stride_number
            in_time = in_frame_num/frame_rate
    if ((cur_state == state.IN_REP) and (winner_stride == stride_number)):
        lastSixSorted = np.sort(ent_arr[history_num-8:history_num])
        print("lastSixSorted",lastSixSorted)
        # if we see good condition for rep take the counting and move to rep state
        # also, if there were 2 below th in the last entropies, don't stop.
        if (((st_std > inrep_std_th) and (st_entropy < inrep_ent_th)) or  (lastSixSorted[1] < lastsix_ent_th)):
            # continue counting
            global_counter = rep_count
            print"global_counter",rep_count
        else:
            out_time = in_frame_num/frame_rate
            if (((out_time-in_time)<4) or (rep_count<5)):
                # fast recovery mechnism, start over
                actions_counter -= 1
                global_counter = 0
                cur_state = state.NO_REP
                print('fast recovery applied !!')
            else:
                # rewind redundant count mechanism
                # find how many frames pass since we have low entropy
                frames_pass = 0
                reversed_ent = ent_arr[::-1]
                for cent in reversed_ent:
                    if (cent > inrep_ent_th):
                        frames_pass += 1
                    else:
                        break
                # calc if and how many global count to rewind
                reversed_cnt = count_array[::-1]
                frames_pass = min(frames_pass, reversed_cnt.shape[0]-1)
                new_counter = reversed_cnt[frames_pass]
                print('couting rewinded by %i' %(global_counter-new_counter))
                global_counter = new_counter
                # stop counting, move to cooldown
                cur_state = state.COOLDOWN
                # init cooldown counter
                cooldown_in_time = in_frame_num/frame_rate
    if (cur_state == state.COOLDOWN):
        cooldown_out_time = in_frame_num/frame_rate
        if ((cooldown_out_time-cooldown_in_time)>4):
            global_counter = 0
            cur_state = state.NO_REP

loadfile = 'init.json'
with open(loadfile,'r') as f:
    data = json.load(f)
    f.close()
initflag = data['initflag']
print "Load initflag", initflag

loadfile = 'recieved.json'
with open(loadfile,'r') as f:
    data = json.load(f)
    f.close()
#print(data)
#load_data()
variables = {}
if not(initflag):
    print('Load')
    stride_number = data['interval']
    loadfile = 'variables'+str(data['interval'])+'.json'
    #loadfile = 'test.json'
    with open(loadfile,'r') as f:
        variables = json.load(f)
        f.close()
    globals().update(variables)
    std_arr = np.load("std_arr"+str(stride_number)+".npy")
    ent_arr = np.load("ent_arr"+str(stride_number)+".npy")
    label_array = np.load("label_array"+str(stride_number)+".npy")
    count_array = np.load("count_array"+str(stride_number)+".npy")
    loadfile = 'global.json'
    with open(loadfile,'r') as f:
        globalvar = json.load(f)
        f.close()
    globals().update(globalvar)
    initflag = False
    if globalvar['winner_stride']!=0 and globalvar['winner_stride']!=stride_number :
        sys.exit(2)

#print("Counting")
#print("Load frame_residue",frame_residue)
in_frame_num+=1
count()
#print("Counted")
if initflag:
    save_data(5,'variables5.json',globals().copy())
    save_data(7,'variables7.json',globals().copy())
    save_data(9,'vareables9.json',globals().copy())
    print "initflag", initflag
    initflag = False
    dic = {}
    dic['initflag'] = False
    outfile = 'init.json'
    #print(dic)
    with open(outfile,'w') as f:
        #json.dumps(dic,outfile)
        f.write(json.dumps(dic,outfile)+'\n')
        f.close()  
else:
    save_data(data['interval'],'variables'+str(data['interval'])+'.json',globals().copy())


try:
    sys.stdout.close()
except:
    pass
try:
    sys.stderr.close()
except:
    pass



