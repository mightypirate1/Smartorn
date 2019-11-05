import tensorflow as tf
import numpy as np
import time
import utils.utils as utils

class smartorn_layer(tf.keras.layers.Layer):
    def __init__(
                 self,
                 input_shape,
                 output_shape,
                 dtype=tf.float16,
                 name="smartorn_layer",
                 n_neurons=100,
                 n_dimentions=3,
                 radial_decay=False,
                 depth_decay=False,
                 renormalize_activations=False,
                 trainable_bias=False,
                 trainable_input_position=False,
                 reapply_inputs=True,
                 reg_amount= {
                                "spatial" : 1.1,
                                "direction" : 0.01,
                                "output"   : 1.0,
                                "entropy" : None,
                              },
                 ):
        super(smartorn_layer, self).__init__(dtype=dtype)
        #Shapes!
        self.DEBUG = False
        self._name = name
        self.EPSILON = tf.constant(10**-5, dtype=dtype)
        self.trainable_bias = trainable_bias
        self.trainable_input_position = trainable_input_position
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.n_input = np.prod(input_shape[1:])
        self.n_output = np.prod(output_shape[1:])
        self.n = n_neurons + self.n_input + self.n_output
        self.dim = n_dimentions
        self.radial_decay=radial_decay
        self.depth_decay=depth_decay
        self.reg_amount = reg_amount

    def build(self, input_shapes):
        self.create_neurons()
        self.create_regularizers()

    def call(self, x):
        new_activations = self.apply_neurons(x)
        return new_activations

    def apply_neurons(self, x):
        def strength(x, w=1.0): # x in [-1, 1]
            ret = ( tf.math.exp(w*x) - (np.e**-w) ) / ( np.e**w-np.e**-w )
            #Original thought was a relu so signals are not "felt" if receptor and sender are not pointing the same way. This is a "fuzzied" version which has a gradient always...
            # (e^x - e^-1) / (e^1-e^-1)
            return ret
        def activation(x):
            return tf.nn.elu(x)
        #############
        #############
        x = tf.expand_dims(x,2)
        x = tf.expand_dims(x,2)
        outdir = tf.expand_dims( self.outdir_tf,  2)
        indir  = tf.expand_dims( self.indir_tf,  1)
        power = tf.expand_dims( self.power_tf, 2)
        #####
        #######
        #########
        delta_p = tf.expand_dims(self.position_tf, 1) - tf.expand_dims(self.position_tf, 2)
        delta_p_norm = tf.math.sqrt( tf.maximum(tf.reduce_sum( tf.math.square(delta_p), axis=-1, keepdims=True), self.EPSILON ) )
        delta_p_bar = tf.math.divide( delta_p, delta_p_norm )
        delta_p_bar = utils.remove_nan(delta_p_bar, value=0.0)
        #########
        #######
        #####
        alpha  = strength( utils.dot( delta_p_bar, outdir) ) #Send strength
        if self.radial_decay:
            print("Radial decay code has not been debugged: there MAY be errors...")
            exponent = 1.0 + self.radial_decay_tf[:,:,tf.newaxis,tf.newaxis]
            alpha = tf.math.pow( alpha, exponent )
        beta   = strength( utils.dot( delta_p_bar, indir) ) #Receive strength
        if self.depth_decay:
            print("Depth decay code has not been debugged: there MAY be errors...")
            decay_factor =  self.depth_decay_tf[:,:,tf.newaxis,tf.newaxis]
            beta = beta * tf.math.exp( -decay_factor * delta_p_norm )
        Z = x * power * alpha * beta
        new_activation = activation( tf.reduce_sum(Z, axis=1, keepdims=False) - tf.expand_dims(self.bias_tf,2))
        return tf.squeeze(new_activation, axis=-1)

    def create_neurons(self, debug=True):
        # np.random.seed(int(time.time()))
        def _reset_input_position(x):
            if self.trainable_input_position:
                return x
            else:
                return x[:,:self.n_input,:].assign(input_init_pos)
        def _positive(x):
            if tf.is_tensor(x):
                return tf.nn.relu(x)
            assert False, "_positive not implemented for non-tensor type objects"
        def _normalize(x):
            if tf.is_tensor(x):
                return x / tf.maximum(tf.sqrt(tf.reduce_sum( tf.math.square(x), axis=-1, keepdims=True)), self.EPSILON)
            return x / np.linalg.norm(x, axis=2, keepdims=True)
        #dist
        init_dist = np.ones((1,self.n,self.n))
        #positions
        init_pos = np.random.randn(1,self.n, self.dim)
        init_pos[:,:self.n_input,:]  = input_init_pos = utils.init_pos_from_shape(self._input_shape, is_input=True)
        init_pos[:,-self.n_output:,:]                 = utils.init_pos_from_shape(self._output_shape, is_input=False)
        #in_dir
        init_indir = np.random.randn(1,self.n, self.dim)
        init_indir[:,:self.n_input,:]   = utils.init_dir_from_shape(self._input_shape)
        init_indir[:,-self.n_output:,:] = utils.init_dir_from_shape(self._output_shape)
        init_indir = _normalize(init_indir)
        #out_dir
        init_outdir = np.random.randn(1,self.n, self.dim)
        init_outdir[:,:self.n_input,:]   = utils.init_dir_from_shape(self._input_shape)
        init_outdir[:,-self.n_output:,:] = utils.init_dir_from_shape(self._output_shape)
        init_outdir = _normalize(init_outdir)
        #bias
        init_bias = np.ones((1,self.n))
        #decays
        init_depth_decay = np.zeros((1,self.n))
        init_radial_decay = np.zeros((1,self.n))
        #power
        init_power = np.ones((1,self.n))
        ##
        if self.DEBUG:
            init_pos[:,self.n_input,:] = [-1,0,0]
            init_indir[:,self.n_input,:] = [0,0,1]
        #Create tensors
        self.power_tf         = tf.Variable(init_power, dtype=self._dtype, name="power"                                     )
        self.position_tf      = tf.Variable(init_pos, dtype=self._dtype, name="position"  , constraint=_reset_input_position)
        self.indir_tf         = tf.Variable(init_indir, dtype=self._dtype, name="indir"   , constraint=_normalize           )
        self.outdir_tf        = tf.Variable(init_outdir, dtype=self._dtype, name="outdir" , constraint=_normalize           )
        self.bias_tf          = tf.Variable(init_bias, dtype=self._dtype, name="bias", trainable=self.trainable_bias        )
        self.radial_decay_tf  = tf.Variable(init_radial_decay, dtype=self._dtype, name="radial_decay"                       ) if self.radial_decay else None
        self.depth_decay_tf   = tf.Variable(init_depth_decay, dtype=self._dtype, name="depth_decay", constraint=_positive   ) if self.depth_decay else None

    def create_regularizers(self, output=True, direction=True, spatial=True, directional_mode='both',spatial_mode='hard', norm_ord=2, reg_type=tf.math.square):
        #Create them!
        regularizers = {}
        if spatial: #Keeps nerons from flying off!
            assert spatial_mode in ['hard', 'soft'], "spatial regularizer modes are: soft, hard"
            dist = tf.norm(self.position_tf, axis=-1, ord=norm_ord)
            if spatial_mode == 'soft':
                x = tf.reduce_mean( (dist - 1),           axis=-1)
            if spatial_mode == 'hard':
                x = tf.reduce_mean( tf.nn.relu(dist - 1), axis=-1)
            regularizers['spatial'] = reg_type(x, name='spatial_regularizer')
        if direction: #Biases
            assert directional_mode in ['indir', 'outdir', 'both'], "directional regularizer modes are: indir, outdir, both"
            centroid = tf.reduce_mean( self.position_tf, axis=1, keepdims=True )
            reg_dir = centroid - self.position_tf
            if directional_mode == 'indir':
                x = tf.nn.relu(utils.dot(reg_dir, self.indir_tf))
            if directional_mode == 'outdir':
                x = tf.nn.relu(utils.dot(-reg_dir, self.outdir_tf))
            if directional_mode == 'both':
                x = tf.nn.relu(utils.dot(reg_dir, self.indir_tf)) + tf.nn.relu(utils.dot(-reg_dir, self.outdir_tf))
            regularizers['direction'] = reg_type( tf.reshape(tf.reduce_mean(x, axis=1), [1,]), name='directional_regularizer' )
        #Add them!
        for reg_name in regularizers:
            self.add_loss( lambda : (self.reg_amount[reg_name] * regularizers[reg_name]))
