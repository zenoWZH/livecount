'''
live repetition counting system, ICCV2015
Ofir Levy, Lior Wolf

transfer to Tensorflow
'''
import numpy
import tensorflow as tf
class LogisticRegression(object):

    def __init__(self, inputs, n_in, n_out):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = tf.variable(np.zeros((n_in, n_out),dtype='float32'),name='W')
        # initialize the baises b as a vector of n_out 0s
        self.b = tf.variable(np.zeros((n_out),dtype='float32'),name='b')

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = tf.nn.softmax(tf.dot(inputs, self.W) + self.b)

        self.p_y_given_x_printed = tf.Print(data = self.p_y_given_x, message = 'p_y_given_x = ')
        #self.p_y_given_x_printed = self.p_y_given_x

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = tf.arg_max(self.p_y_given_x, axis=1)  
       # parameters of the model
        self.params = [self.W, self.b]

    def __getstate__(self):
        return (self.W.get_variable(), self.b.get_variable())
    
    def __setstate__(self, state):
        W, b = state
        self.W.assign(W)
        self.b.sassign(b)

    def negative_log_likelihood(self, y):
        return -tf.mean(tf.log(self.p_y_given_x)[tf.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', targetf.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the tf.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return tf.mean(tf.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def get_output_labels(self, y):
        return (((self.y_pred-y)*0 + self.y_pred), self.p_y_given_x)




class HiddenLayer(object):
    def __init__(self, rng, inputs, n_in, n_out, W=None, b=None,
                 activation=tf.tanh):
        self.inputs = inputs

        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype='float32')
            if activation == tf.sigmoid:
                W_values *= 4

            W = tf.Variable(value=W_values, name='W')
            self.W = W
            
        else:
            W = tf.Variable(value=W, name='W')

        if b is None:
            b_values = np.zeros((n_out,), dtype='float32')
            b = tf.Variable(value=b_values, name='b')
            self.b = b
            
        else:
            b = tf.Variable(value=b, name='b')
            self.b = b
        
        

        lin_output = tf.dot(inputs, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))                       
                       #else tf.maximum(0.0, lin_output)) #activation(lin_output))                       
        # parameters of the model
        self.params = [self.W, self.b]

    def __getstate__(self):
        return (self.W.get_variable(), self.b.get_variable())
    
    def __setstate__(self, state):
        W, b = state
        self.W.set_value(W)
        self.b.set_value(b)

    def get_output_vector(self):
        return (self.output)




class LeNetConvPoolLayer(object):

    def __init__(self, rng, inputs, filter_shape, image_shape, W=None, b=None, poolsize=(2, 2)):
        #data_format= "NCHW"
        assert image_shape[1] == filter_shape[1]
        self.inputs = tf.transponse(inputs, parm=[0,2,3,1])

        # there are "num inputs feature maps * filter height * filter width"
        # inputss to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        if W is None:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = tf.transponse(tf.Variable(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),dtype='float32')), perm=[2,3,1,0])
        else:
            self.W = tf.transponse(tf.Variable(value=W),perm=[2,3,1,0])
        # the bias is a 1D tensor -- one bias per output feature map
        if b is None:
            b_values = np.zeros((filter_shape[0],), dtype='float32')
            self.b = tf.Variable(value=b_values)
        else:
            self.b = tf.Variable(value=b)

        # convolve inputs feature maps with filters   ********
        conv_out = tf.add(tf.nn.conv2d(inputs, self.W, strides=(1,1,1,1), padding="VALID"),self.b)

        # downsample each feature map individually, using maxpooling   *******
        pooled_out = tf.nn.max_pool(conv_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        self.output = tf.maximum(0.0, pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
                
        # store parameters of this layer
        self.params = [self.W, self.b]

    def __getstate__(self):
        return (self.W.get_variable(), self.b.get_variable())
    
    def __setstate__(self, state):
        W, b = state
        self.W.set_value(W)
        self.b.set_value(b) 