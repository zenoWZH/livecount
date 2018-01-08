'''
live repetition counting system
Ofir Levy, Lior Wolf
Tel Aviv University
'''
import pickle
#from skimage import transform
import numpy
import cv2
from scipy.ndimage import filters
#import theano
import tensorflow as tf
import scipy
#import theano.tensor as T
import sys, getopt
from layers_tf import LogisticRegression, HiddenLayer, LeNetConvPoolLayer

global test_set_x, test_set_y, shared_test_set_y

class state:
    NO_REP = 1
    IN_REP = 2
    COOLDOWN = 3

# global vars
detector_strides = [5,7,9]
static_th = 10
norep_std_th = 13
norep_ent_th = 1.0
inrep_std_th = 13
inrep_ent_th = 1.1
lastsix_ent_th = 1.1
history_num = 9

in_time = 0
out_time = 0
cooldown_in_time = 0
cooldown_out_time = 0
#frame_rate = 30
#num_of_vids = 25

global_counter = 0
winner_stride = 0
cur_state = state.NO_REP;
in_frame_num = -1
actions_counter = 0


class RepDetector:

    def __init__(self, proFrame, st_number):
        self.stride_number = st_number
        self.frame_set = numpy.zeros([20,proFrame.shape[0],proFrame.shape[1]])
        for i in range(20):
            self.frame_set[i,:,:] = proFrame

        self.rep_count = 0
        self.frame_residue = 0
        self.st_entropy = 0
        self.st_std = 0
        self.std_arr = numpy.zeros(history_num)
        self.ent_arr = numpy.zeros(history_num)+2
        self.label_array = numpy.zeros(history_num)
        self.count_array = numpy.zeros(history_num)
        self.cur_std = 0
        self.cur_entropy = 0

        
    # receive a list of 20 frames and return the ROI scaled to 50x50
    def get_boundingbox(self):

        fstd = numpy.std(self.frame_set,axis=0)
        framesstd = numpy.mean(fstd)
        th = framesstd
        ones = numpy.ones(10)
        big_var = (fstd>th)


        if (framesstd==0): # no bb, take full frame
            frameROIRes = numpy.zeros([20,50,50])
            for i in range(20):
                frameROIRes[i,:,:] = scipy.misc.imresize(self.frame_set[i,:,:], size=(50,50),interp='bilinear')

                frameROIRes = numpy.reshape(frameROIRes, (1,frameROIRes.shape[0]*frameROIRes.shape[1]*frameROIRes.shape[2]))
                frameROIRes = frameROIRes.astype(numpy.float32)

            return (frameROIRes)

        big_var = big_var.astype(numpy.float32)    
        big_var = filters.convolve1d(big_var, ones, axis=0)
        big_var = filters.convolve1d(big_var, ones, axis=1)

        th2 = 80
        i,j = numpy.nonzero(big_var>th2)

        if (i.size > 0):

            si = numpy.sort(i)
            sj = numpy.sort(j)


            ll = si.shape[0]
            th1 = round(ll*0.02)
            th2 = int(round(numpy.floor(ll*0.98)))
            print(th2)
            y1 = si[th1]
            y2 = si[th2]
            x1 = sj[th1]
            x2 = sj[th2]

            # cut image ROI
            if (((x2-x1)>0) and ((y2-y1)>0)):
                framesRoi = self.frame_set[:,y1:y2,x1:x2]
            else: 
                framesRoi = self.frame_set[:,:,:]
        else:
            framesRoi = self.frame_set[:,:,:]

        # resize to 50x50    
        frameROIRes = numpy.zeros([20,50,50])
        for i in range(20):
            frameROIRes[i,:,:] = scipy.misc.imresize(framesRoi[i,:,:], size=(50,50),interp='bilinear')

        riofstd = numpy.std(frameROIRes,axis=0)
        self.cur_std = numpy.mean(riofstd)

        frameROIRes = numpy.reshape(frameROIRes, (1,frameROIRes.shape[0]*frameROIRes.shape[1]*frameROIRes.shape[2]))
        frameROIRes = frameROIRes.astype(numpy.float32)

        return (frameROIRes)

    
    def do_local_count(self, classify, initial):

        framesArr = self.get_boundingbox()    
        # classify    
        #test_set_x.set_value(framesArr, borrow=True)             
        #output_label , pYgivenX  = classify(0)
        test_set_x = framesArr
        output_label , pYgivenX  = classify(0)
        self.cur_entropy = - (pYgivenX*numpy.log(pYgivenX)).sum()
        # count
        output_label = output_label[0] + 3  
        self.label_array = numpy.delete(self.label_array,0,axis=0)
        self.label_array = numpy.insert(self.label_array, history_num-1 , output_label, axis=0)
        #take median of the last frames
        med_out_label = numpy.ceil(numpy.median(self.label_array[history_num-4:history_num]))    
        med_out_label = med_out_label.astype('int32')

        if initial:
            self.rep_count = 20 / (med_out_label)
            self.frame_residue = 20 % (med_out_label)     
        else:
            self.frame_residue += 1
            if (self.frame_residue >= med_out_label):
                self.rep_count += 1;
                self.frame_residue = 0;


    
    def count(self, proFrame):

        # globals               
        global in_time,     out_time, cooldown_in_time, cooldown_out_time
        global global_counter, winner_stride, cur_state, in_frame_num, actions_counter    

        # insert new frame
        self.frame_set = numpy.delete(self.frame_set,0,axis=0)
        self.frame_set = numpy.insert(self.frame_set, 19 , proFrame, axis=0)

        if (cur_state == state.NO_REP):
            self.do_local_count(classify, True) 
        if ((cur_state == state.IN_REP) and (winner_stride == self.stride_number)):
            self.do_local_count(classify, False)
        if (cur_state == state.COOLDOWN): 
            self.do_local_count(classify, True)

        # common to all states
        if (self.cur_std < static_th):
            self.cur_entropy = 2
        self.count_array = numpy.delete(self.count_array,0,axis=0) 
        self.count_array = numpy.insert(self.count_array, history_num-1 , self.rep_count, axis=0)
        self.ent_arr = numpy.delete(self.ent_arr,0,axis=0)
        self.ent_arr = numpy.insert(self.ent_arr, history_num-1 , self.cur_entropy, axis=0)
        self.std_arr = numpy.delete(self.std_arr,0,axis=0)
        self.std_arr = numpy.insert(self.std_arr, history_num-1 , self.cur_std, axis=0)
        self.st_std = numpy.median(self.std_arr)
        self.st_entropy = numpy.median(self.ent_arr)

        if (cur_state == state.NO_REP):
            # if we see good condition for rep take the counting and move to rep state
            if ((self.st_std > norep_std_th) and (self.st_entropy < norep_ent_th)):
                # start counting!
                actions_counter += 1
                cur_state = state.IN_REP
                global_counter = self.rep_count
                winner_stride = self.stride_number
                in_time = in_frame_num/30
        if ((cur_state == state.IN_REP) and (winner_stride == self.stride_number)):
            lastSixSorted = numpy.sort(self.ent_arr[history_num-8:history_num])
            # if we see good condition for rep take the counting and move to rep state
            # also, if there were 2 below th in the last entropies, don't stop.
            if (((self.st_std > inrep_std_th) and (self.st_entropy < inrep_ent_th)) or  (lastSixSorted[1] < lastsix_ent_th)):
            # continue counting
                global_counter = self.rep_count
            else:
                out_time = in_frame_num/30
                if (((out_time-in_time)<4) or (self.rep_count<5)):
                    # fast recovery mechnism, start over
                    actions_counter -= 1
                    global_counter = 0
                    cur_state = state.NO_REP 
                    print('fast recovery applied !!')
                else:
                    # rewind redundant count mechanism
                    # find how many frames pass since we have low entropy
                    frames_pass = 0
                    reversed_ent = self.ent_arr[::-1]
                    for cent in reversed_ent:
                        if (cent > inrep_ent_th):
                            frames_pass += 1
                        else:
                            break   
                    # calc if and how many global count to rewind
                    reversed_cnt = self.count_array[::-1]
                    frames_pass = min(frames_pass, reversed_cnt.shape[0]-1)
                    new_counter = reversed_cnt[frames_pass]     
                    print('couting rewinded by %i' %(global_counter-new_counter))
                    global_counter = new_counter
                    # stop counting, move to cooldown       
                    cur_state = state.COOLDOWN
                    # init cooldown counter
                    cooldown_in_time = in_frame_num/30
        if (cur_state == state.COOLDOWN):    
            cooldown_out_time = in_frame_num/30
            if ((cooldown_out_time-cooldown_in_time)>4):    
                global_counter = 0
                cur_state = state.NO_REP
                
def shared_dataset(data_xy, borrow=True):
    
    data_x, data_y = data_xy
    shared_x = numpy.asarray(data_x,dtype='float32')
    shared_y = numpy.asarray(data_y,dtype='float32')
    
    return shared_x, tf.cast(shared_y, 'int32'), shared_y


def process_single_frame(frame):

    # convert to gray scal
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # downscale by 2
    gray_frame = gray_frame[::2,::2]  
    return gray_frame

def draw_str(dst, xypair, s, color, scale):
    x = xypair[0]
    y = xypair[1]
    if (color[0]+color[1]+color[2]==255*3):
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness = 4, lineType=10)
    else:
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness = 4, lineType=10)
        
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), lineType=11)


if __name__ == '__main__':
    
    rng = numpy.random.RandomState(23455)
        
    ######################## build start ########################    

    # create an empty shared variables to be filled later
    data_x = numpy.zeros([1,20*50*50])
    data_y = numpy.zeros(20)
    train_set = (data_x, data_y)
    test_set_x, test_set_y, shared_test_set_y  = shared_dataset(train_set)         
    
    print('building ... ')           
    batch_size = 1
    
    
    # image size
    layer0_w = 50
    layer0_h = 50
    layer1_w = (layer0_w-4)/2
    layer1_h = (layer0_h-4)/2
    layer2_w = (layer1_w-2)/2
    layer2_h = (layer1_h-2)/2
    layer3_w = (layer2_w-2)/2
    layer3_h = (layer2_h-2)/2

    
    ######################
    # BUILD ACTUAL MODEL #
    ######################    
    
    # image sizes
    batchsize     = batch_size
    in_channels   = 20
    in_width      = 50
    in_height     = 50
    #filter sizes
    flt_channels  = 40
    flt_time      = 20
    flt_width     = 5
    flt_height    = 5

    # allocate symbolic variables for the data
    #index = T.lscalar() 
    #x = T.matrix('x')   
    #y = T.ivector('y')                        
    x = tf.placeholder(dtype='float32')
    y = tf.placeholder(dtype='float32')
    
    signals_shape = (batchsize, in_channels, in_height, in_width)
    filters_shape = (flt_channels, in_channels, flt_height, flt_width)
    
    print('signals_shape=',signals_shape)
    print('filters_shape=',filters_shape)
    
    # load weights  
    print ('loading weights state')
    f = open('weights.save', 'rb')
    loaded_objects = []
    for i in range(5):
        loaded_objects.append(pickle.load(f,encoding='bytes'))
    f.close()
        
    layer0_input = tf.reshape(x,signals_shape)
    
    #print('layer0_input=',tf.shape(layer0_input))
    print('layer0')
    layer0 = LeNetConvPoolLayer(rng, inputs=layer0_input,
                image_shape=signals_shape,
                    filter_shape=filters_shape,W=loaded_objects[0][0],b=loaded_objects[0][1], poolsize=(2, 2))
    print('layer1')
    layer1 = LeNetConvPoolLayer(rng, inputs=layer0.output,
            image_shape=(batch_size, flt_channels, layer1_w, layer1_h),
            filter_shape=(60, flt_channels, 3, 3),W=loaded_objects[1][0],b=loaded_objects[1][1], poolsize=(2, 2))

    print('layer2')
    layer2 = LeNetConvPoolLayer(rng, inputs=layer1.output,
                image_shape=(batch_size, 60, layer2_w, layer2_h),
                filter_shape=(90, 60, 3, 3),W=loaded_objects[2][0],b=loaded_objects[2][1], poolsize=(2, 2))
    layer3_input = tf.reshape(layer2.output, [-1])

    print('layer3')
    layer3 = HiddenLayer(rng, inputs=layer3_input, n_in=90 * layer3_w * layer3_h  ,
                         n_out=500, W=loaded_objects[3][0],b=loaded_objects[3][1],activation=tf.tanh)
  
    print('layer4')
    layer4 = LogisticRegression(inputs=layer3.output, n_in=500, n_out=8, W=loaded_objects[4][0],b=loaded_objects[4][1])   # change the number of output labels


    cost = layer4.negative_log_likelihood(y)
    
    outputs = layer4.get_output_labels(y)

    #classify = theano.function([index], outputs=layer4.get_output_labels(y),
    #                           givens={
    #                               x: test_set_x[index * batch_size: (index + 1) * batch_size],
    #                               y: test_set_y[index * batch_size: (index + 1) * batch_size]})
    
    print("test_set_x shape= ",test_set_x.shape)
    print("test_set_y shape= ",test_set_y.shape)
    
    print('CNN Finished')
    #print('classify = ', classify)
    #layer0.__setstate__(loaded_objects[0])
    #layer1.__setstate__(loaded_objects[1])
    #layer2.__setstate__(loaded_objects[2])
    #layer3.__setstate__(loaded_objects[3])
    #layer4.__setstate__(loaded_objects[4])
    

    def classify(index):    
        with tf.Session() as sess :
            init = tf.global_variables_initializer()
            sess.run(init)
            classify, __cost = sess.run([outputs,cost],feed_dict={
                                   x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                   y: numpy.zeros((index + 1) * batch_size)})
        sess.close()
        return classify, __cost
    
    ######################## build done ########################
    
    
    inputfile = ''
    fromCam = True

    try:
        opts, args = getopt.getopt(sys.argv[1:],'i:h:',['ifile=','help='])
    except getopt.GetoptError:
        print ('invalid arguments. for input from camera, add -i <inputfile> for input from file')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-i', '--ifile'):
            inputfile = arg
            fromCam = False
            
    if (fromCam):
        print ('using camera input')
        cap = cv2.VideoCapture(0)
    else:
        print ('using input file: ', inputfile)         
        cap = cv2.VideoCapture(inputfile)
        
            
    # my timing 
    frame_rate = 30
    frame_interval_ms = 1000/frame_rate
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    
    video_writer = cv2.VideoWriter("../out/live_out.avi", fourcc, frame_rate, (640, 480))

    frame_counter = 0
    ret, frame = cap.read()

    proFrame = process_single_frame(frame)
    
    # init detectors    
    st_a_det = RepDetector(proFrame, detector_strides[0])
    st_b_det = RepDetector(proFrame, detector_strides[1])
    st_c_det = RepDetector(proFrame, detector_strides[2])
        
    while True:
        
        in_frame_num += 1
        if (in_frame_num%2 ==1):
            continue
        
        ret, frame = cap.read()
        if (ret == 0):
            print ('unable to read frame')
            break
        proFrame = process_single_frame(frame)

        # handle stride A       
        if (frame_counter % st_a_det.stride_number == 0):
            st_a_det.count(proFrame)
        # handle stride B
        if (frame_counter % st_b_det.stride_number == 0):
            st_b_det.count(proFrame)
        # handle stride C
        if (frame_counter % st_c_det.stride_number == 0):
            st_c_det.count(proFrame)        
        
        # display result on video           
        blue_color = (130, 0, 0)
        green_color = (0, 130, 0)
        red_color = (0, 0, 130)
        orange_color = (0,140,255)
        
        out_time = in_frame_num/60
        if  ((cur_state == state.IN_REP) and (((out_time-in_time)<4) or (global_counter<5))):
            draw_str(frame, (20, 120), " new hypothesis (%d) " % global_counter, orange_color, 1.5)
        if ((cur_state == state.IN_REP) and ((out_time-in_time)>=4) and (global_counter>=5)):
            draw_str(frame, (20, 120), "action %d: counting... %d" % (actions_counter, global_counter), green_color, 2)
        if ((cur_state == state.COOLDOWN) and (global_counter>=5)):
            draw_str(frame, (20, 120), "action %d: done. final counting: %d" % (actions_counter, global_counter), blue_color, 2)
            #print 'action %d: done. final counting: %d' % (actions_counter, global_counter)

        video_writer.write(frame)
        cv2.namedWindow('video', cv2.WINDOW_NORMAL)         
        cv2.imshow('video', frame)                    
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
        frame_counter = frame_counter + 1

    
    cap.release()        
    video_writer.release()
    cv2.destroyAllWindows()