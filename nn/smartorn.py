import tensorflow as tf
import numpy as np

import utils.utils as utils

EPSILON = 10**-7

class smartorn:
    def __init__(
                 self,
                 input_shape,
                 output_shape,
                 n_neurons=100,
                 n_layers=3,
                 n_dimentions=3,
                 dtype=tf.float16,
                 name="smartorn",
                 radial_decay=True,
                 depth_decay=True,
                 normalized_activations=False,
                 trainable_bias=True,
                ):
        self.DEBUG = False
        self.dbg_tensors = []
        #Store some numbers
        self.input_shape = input_shape if type(input_shape) is list else input_shape.as_list()
        self.output_shape = output_shape if type(output_shape) is list else output_shape.as_list()
        self.n_input = np.prod(input_shape[1:])
        self.n_output = np.prod(output_shape[1:])
        self.n = n_neurons + self.n_input + self.n_output
        self.dim = n_dimentions
        self.dtype = dtype
        self.radial_decay = radial_decay
        self.depth_decay = depth_decay
        self.normalized_activations = normalized_activations
        self.trainable_bias = trainable_bias
        #Initialize
        self.input = tf.placeholder(dtype, input_shape)
        self.init_activation_tf = tf.placeholder(self.dtype, shape=[None, self.n], name="initial_activation")
        self.scope = name
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.create_neurons()
            #Unroll brain
            self.activations = []
            self.outputs = []
            x = self.init_activation_tf
            for _ in range(n_layers):
                print("X:", x)
                x = self.apply_neurons(x)
                self.activations.append(x)
                self.outputs.append(x[:,-self.n_output:])
            self.create_regularizers(spatial=True, direction=True, spatial_mode='hard')
            self.create_init_op()

    def apply_neurons(self, current_activation):
        def strength(x, w=2.0):
            # x in [-1, 1]
            ret = ( tf.math.exp(w*x) - (np.e**-1) ) / ( np.e**w-np.e**-1 ) + 0.01
            # ret = tf.math.exp(0*x)
            #Original thought was a relu so signals are not "felt" if receptor and sender are not pointing the same way. This is a "fuzzied" version which has a gradient always...
            # (e^(2x) - e^-1) / (e^2-e^(-1))
            # f = lambda x : 1.0 + tf.nn.elu( x - 1.0 )
            # ret = f(x)
            self.dbg_tensors += [ret]
            return ret

        def activation(x):
            return tf.nn.elu(x)
        x = self.renormalize_activations(current_activation)
        x = tf.expand_dims(x,2)
        x = tf.expand_dims(x,2)
        outdir_bar = tf.expand_dims( self.outdir_bar_tf,  2)
        indir_bar  = tf.expand_dims( self.indir_bar_tf,  1)
        power = tf.expand_dims( self.power_tf, 2)

        #####
        #######
        #########
        delta_p_norm = tf.reduce_sum( tf.math.square(self.delta_p_tf), axis=-1, keepdims=True)
        delta_p_bar = tf.math.divide( self.delta_p_tf, tf.math.maximum( 0.0001, delta_p_norm) )
        delta_p_bar = utils.remove_nan(delta_p_bar, value=0.0)
        #########
        #######
        #####
        alpha  = strength( utils.dot( delta_p_bar, outdir_bar) ) #Send strength
        if self.radial_decay:
            print("Radial decay code has not been debugged: there MAY be errors...")
            exponent = 1.0 + self.radial_decay_tf[:,:,tf.newaxis,tf.newaxis]
            alpha = tf.math.pow( alpha, exponent )
        beta   = strength( utils.dot( delta_p_bar, indir_bar) ) #Receive strength
        if self.depth_decay:
            print("Depth decay code has not been debugged: there MAY be errors...")
            decay_factor =  self.depth_decay_tf[:,:,tf.newaxis,tf.newaxis]
            beta = beta * tf.math.exp( -decay_factor * delta_p_norm )
        Z = x * power * alpha * beta
        new_activation = activation( tf.reduce_mean(Z, axis=1, keepdims=False) - tf.expand_dims(self.bias_tf,2))
        output = self.reapply_inputs(tf.squeeze(new_activation, axis=-1))
        # self.dbg_tensors = [outdir_bar, indir_bar, delta_p_bar, alpha, beta, x, output, delta_p_norm]
        self.dbg_tensors = []
        return output

    def renormalize_activations(self, x):
        if self.normalized_activations:
            sum = tf.reduce_sum(x+EPSILON, axis=-1, keepdims=True)
            return tf.math.divide(x,sum)
        else:
            return x

    def reapply_inputs(self, x):
        mode = 'fixed'
        # mode = 'none'
        if mode == 'none':
            return x
        if mode == 'fixed':
        #Each "layer" has the same activation for the input neurons
            mask = np.zeros((1,self.n))
            mask[0,:self.n_input] = 1
            mask_tf = self._variable('input_mask', mask, trainable=False)
            x = x * (1-mask_tf) + self.init_activation_tf * mask_tf
            return x

    def create_neurons(self, debug=True):
        #dist
        init_dist = np.ones((1,self.n,self.n))
        #positions
        init_pos = 2 * np.random.random(size=[1,self.n, self.dim]) - 1
        init_pos[:,:self.n_input,:]   = utils.init_pos_from_shape(self.input_shape, is_input=True)
        init_pos[:,-self.n_output:,:] = utils.init_pos_from_shape(self.output_shape, is_input=False)
        #in_dir
        init_indir = 2 * np.random.random(size=[1,self.n, self.dim]) - 1
        init_indir[:,:self.n_input,:]   = utils.init_dir_from_shape(self.input_shape)
        init_indir[:,-self.n_output:,:] = utils.init_dir_from_shape(self.output_shape)
        #out_dir
        init_outdir = 2 * np.random.random(size=[1,self.n, self.dim]) - 1
        init_outdir[:,:self.n_input,:]   = utils.init_dir_from_shape(self.input_shape)
        init_outdir[:,-self.n_output:,:] = utils.init_dir_from_shape(self.output_shape)
        #bias
        init_bias = np.ones((1,self.n))
        #decays
        init_depth_decay = np.zeros((1,self.n))
        init_radial_decay = np.zeros((1,self.n))
        #power
        init_power = np.zeros((1,self.n))
        ##
        if self.DEBUG:
            init_pos[:,self.n_input,:] = [-1,0,0]
            init_indir[:,self.n_input,:] = [0,0,1]
        #Create tensors
        self.power_tf         = self._variable("power"       , init_power                 , collections=['signal',])
        self.position_tf = p  = self._variable("position"    , init_pos                   , collections=['signal', 'position',])
        self.indir_tf         = self._variable("indir"       , init_indir,  normalize=True, collections=['direction',])
        self.outdir_tf        = self._variable("outdir"      , init_outdir, normalize=True, collections=['direction',])
        self.bias_tf          = self._variable("bias"        , init_bias                  , collections=['signal',], trainable=self.trainable_bias)
        self.radial_decay_tf  = self._variable("radial_decay", init_radial_decay          , collections=['decay',]) if self.radial_decay else None
        self.depth_decay_tf   = self._variable("depth_decay" , init_depth_decay           , collections=['decay',])  if self.depth_decay else None
        self.delta_p_tf = tf.expand_dims(p, 1) - tf.expand_dims(p, 2)
        self.outdir_bar_tf = tf.nn.l2_normalize(self.outdir_tf, axis=-1)
        self.indir_bar_tf = tf.nn.l2_normalize(self.indir_tf, axis=-1)

    def _variable(self, name, initval, trainable=True, normalize=False, collections=[]):
        if normalize:
            initval = initval / np.linalg.norm(initval, axis=2, keepdims=True)
        return tf.get_variable(
                                name,
                                initval.shape,
                                dtype=self.dtype,
                                initializer=tf.constant_initializer(initval),
                                trainable=trainable,
                                collections=[tf.GraphKeys.GLOBAL_VARIABLES, "smartorn", *collections],
                                )

    def create_init_op(self):
        self.init_op = tf.variables_initializer(tf.get_collection(self.scope))

    def create_regularizers(self, direction=False, spatial=True, directional_mode='both',spatial_mode='soft', norm_ord=2, reg_type=tf.math.square):
        self.regularizers = []
        if spatial:
            #spatial regularizer
            assert spatial_mode in ['hard', 'soft'], "spatial regularizer modes are: soft, hard"
            dist = tf.norm(self.position_tf, axis=-1, ord=norm_ord)
            if spatial_mode == 'soft':
                x = tf.reduce_mean( (dist - 1),           axis=-1)
            if spatial_mode == 'hard':
                x = tf.reduce_mean( tf.nn.relu(dist - 1), axis=-1)
            spatial_reg_tf = reg_type(x)
            self.regularizers.append(spatial_reg_tf)
        if direction:
            #directional regularizer
            assert directional_mode in ['indir', 'outdir', 'both'], "directional regularizer modes are: indir, outdir, both"
            centroid = tf.reduce_mean( self.position_tf, axis=1, keepdims=True )
            reg_dir = centroid - self.position_tf
            if directional_mode == 'indir':
                x = tf.nn.relu(utils.dot(reg_dir, self.indir_bar_tf))
            if directional_mode == 'outdir':
                x = tf.nn.relu(utils.dot(reg_dir, self.outdir_bar_tf))
            if directional_mode == 'both':
                x = tf.nn.relu(utils.dot(reg_dir, self.indir_bar_tf)) + tf.nn.relu(utils.dot(reg_dir, self.outdir_bar_tf))
            directional_reg_tf = reg_type( tf.reshape(tf.reduce_mean(x, axis=1), [1,]) )
            self.regularizers.append(directional_reg_tf)

    def input_pad(self, x):
        batch_size = x.shape[0]
        n = x.shape[1]
        assert n in [self.n, self.n_input], "Invalid initial activations passed to brain"
        #If only n_input activations are specified, those are assumed to be fore the input activations, and the rest is zero.
        if n < self.n:
            _x = np.zeros((batch_size, self.n))
            _x[:,:n] = x
            x = _x
        return x

    @property
    def output(self):
        return self.outputs[-1]

    @property
    def activation(self):
        return self.activations[-1]
