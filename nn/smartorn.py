import tensorflow as tf
import numpy as np

from nn.smartorn_layer import smartorn_layer

# class smartorn(tf.keras.Model):
#     def __init__(self, output_shape, **kwargs):
#         super(smartorn, self).__init__(dtype=kwargs['dtype'], name=kwargs["name"])
#         self.stack = smartorn_layer_stack(output_shape, **kwargs)
#     def call(self, x):
#         self.stack(x)
#
# class smartorn_layer_stack(tf.keras.layers.Layer):
class smartorn(tf.keras.Model):
    def __init__(
                 self,
                 output_shape,
                 #
                 dtype=tf.float16,
                 name="smartorn_stack",
                 trainable_bias=False,
                 trainable_input_position=False,
                 #
                 n_neurons=100,
                 n_layers=3,
                 n_dimentions=3,
                 #
                 radial_decay=True,
                 depth_decay=True,
                 #
                 renormalize_activations=True,
                 reapply_inputs=True,
                 #
                 reg_amount= {
                                "spatial" : 1.1,
                                "direction" : 0.01,
                                "output"   : 1.0,
                                "entropy" : None,
                              },
                 #
                ):
        super(smartorn, self).__init__(dtype=dtype, name=name)

        #DBG...
        self.tracked_gradients = []
        self.tracked_tensors = []
        self.dbg_tensors = []
        self.EPSILON = tf.constant(10**-6, dtype=dtype)

        #Shapes!
        self._output_shape = [None, *output_shape] if type(output_shape) is list else [None, output_shape]

        #SAVE ALL INPUTS!
        self.n_layers = n_layers
        self.trainable_bias = trainable_bias
        self.trainable_bias = trainable_input_position
        self.n_neurons = n_neurons
        self.dim = n_dimentions
        self.radial_decay = radial_decay
        self.depth_decay = depth_decay
        self.reg_amount = reg_amount
        self._renormalize_activations = renormalize_activations
        self._reapply_inputs = reapply_inputs

    def build(self, input_shape):
        #Input/output sizes etc
        self.n_input = np.prod(input_shape[1:])
        self.n_output = np.prod(self._output_shape[1:])
        self.n = self.n_neurons + self.n_input + self.n_output

        #Initialize!
        self.layer = smartorn_layer(
                                    input_shape,
                                    self._output_shape,
                                    #
                                    dtype=self.dtype,
                                    name=self.name+"_layer",
                                    trainable_bias=self.trainable_bias,
                                    #
                                    n_neurons=self.n_neurons,
                                    n_dimentions=self.dim,
                                    #
                                    depth_decay=self.depth_decay,
                                    radial_decay=self.radial_decay,
                                    #
                                    )

        if self._reapply_inputs:
            mask = np.zeros((1,self.n), dtype=self.dtype)
            mask[0,:self.n_input] = 1.0
            self.input_mask_tf = tf.Variable(mask, name='input_mask', trainable=False)

    def call(self, x, training=False):
        x = self.input_pad(x)
        self.init_activation_tf = x
        self.activations = []
        self.outputs = []
        #Unroll brain
        for idx in range(self.n_layers):
            #(Optional) Fix the activation of the input neurons to their initial value
            if self._reapply_inputs:
                x = self.reapply_inputs(x)
            ### Update activations based on the previous time-step
            x = self.layer(x)
            #(Optional) Renormalize the activations [unclear what the  renormalization-scheme "should" be. its currently y = (x-mu)/sigma + mu]
            if self._renormalize_activations:
                x = self.renormalize_activations(x)
            #Store output tensors
            self.activations.append(x)
            self.outputs.append(x[:,-self.n_output:])

            # if 'output' in self.reg_amount and self.reg_amount['output'] > 0.0:
            #     outputreg = self.output_regularizer(self.outputs)
            #     self.layer.add_loss(outputreg)
            # return loss + outputreg

        return self.outputs

    def renormalize_activations(self, x):
        mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        std  = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return (x - mean)/(std+self.EPSILON) + mean

    def reapply_inputs(self, x):
        mode = 'fixed'
        if mode == 'fixed':
        #Each "layer" has the same activation for the input neurons
            x = x * (1-self.input_mask_tf) + self.init_activation_tf * self.input_mask_tf
            return x

    def input_pad(self, x):
        n = x.shape[1]
        batch_size = x.shape[0]
        assert type(x) is np.ndarray or tf.is_tensor(x), "Smartorn called with invalid type: {} (expected tensor or ndarray)".format(type(x))
        assert n in [self.n, self.n_input], "Invalid initial activation passed to brain"
        #If we got a full-sized object, we dont need to pad...
        if n == self.n:
            return x
        #PAD!
        #(If only n_input activations are specified, those are assumed to be fore the input activations, and the rest is zero.)
        if tf.is_tensor(x):
            if x.dtype is not self.dtype:
                x = tf.cast(x, self.dtype)
            ret = tf.pad(x, [[0,0], [0, self.n - n]])
            assert ret.shape[1] == self.n, "your padding is incorrect. this is a bug. fix the bug!"
            return ret
        else:
            _x = np.zeros((batch_size, self.n))
            _x[:,:n] = x
            x = _x
        return x

    def smartorn_loss(self, y_true, y_pred):
        loss = self._L( y_true - y_pred[-1] )
        outputreg = 0.0
        if 'output' in self.reg_amount and self.reg_amount['output'] > 0.0:
            outputreg = self.output_regularizer(self.outputs)
        return loss + outputreg

    def output_regularizer(self, outputs):
        W, w, reg = 1.0, 0.02 ** (1/self.n_layers), 0.0
        mean = tf.add_n(outputs)/len(outputs)
        for y in outputs[:-1]:
            reg *= w
            reg += self._L( y - mean )
            W *= w
            W += 1
        reg = 1 * self.reg_amount['output'] * reg / W

    def _L(self, x):
        return tf.reduce_mean(tf.math.square(x))

    @property
    def output(self):
        return self.outputs[-1]

    @property
    def activation(self):
        return self.activations[-1]
