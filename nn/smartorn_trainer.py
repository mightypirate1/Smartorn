import tensorflow as tf
import numpy as np

from visualize.o3d import visualizer

class trainer:
    def __init__(self, smartorn, session):
        REG_AMMOUNT = 1.1, 0.01 #Spatial, directional
        self.S = smartorn
        self.sess = session
        self.target_tf = tf.placeholder(self.S.dtype, self.S.output_shape)

        ###
        ######
        #########

        # Error loss!
        # #Only last iteration?
        # self.loss_error_tf = tf.reduce_mean(tf.square(self.target_tf - self.S.output))

        #Weighted average?
        exp_decay, W = 0.02**(1/len(self.S.outputs)), 0
        print("Error weighs")
        self.loss_error_tf = 0
        for output in self.S.outputs:
            self.loss_error_tf *= exp_decay
            self.loss_error_tf += tf.reduce_mean(tf.square(self.target_tf - output))
            W *= exp_decay
            W += 1
            print(W)
        self.loss_error_tf /= W

        #########
        ######
        ###

        ## Regularizers
        self.loss_spacereg_tf = REG_AMMOUNT[0] * self.S.regularizers[0]
        self.loss_dirreg_tf = REG_AMMOUNT[1] * self.S.regularizers[1]
        self.loss_reg_tf = self.loss_spacereg_tf + self.loss_dirreg_tf
        self.loss_tf = self.loss_error_tf + self.loss_reg_tf

        self.S.dbg_tensors = [tf.gradients(self.loss_error_tf, self.S.position_tf)]
        optimizer = tf.train.AdamOptimizer(learning_rate=1*10**-4)
        self.training_ops = [optimizer.minimize(self.loss_tf)]

    def train(self, data, targets, minibatch_size=32, epochs=1, visualize=False, debug=False):
        if visualize:
            self.visualizer = visualizer(window_name='Smartorn', width=1024, height=760, left=50, top=50, visible=True)
        n = data.shape[0]
        for t in range(epochs):
            totloss, spaceregloss, dirregloss, regloss = 0.0, 0.0, 0.0, 0.0
            for idx in range(0,n,minibatch_size):
                pi = np.random.permutation(n)
                x =    data[pi[idx:idx+minibatch_size],:]
                y = targets[pi[idx:idx+minibatch_size],:]
                _, _totloss, _spaceregloss, _dirregloss, _dbg = self.sess.run(
                                                                                        [self.training_ops, self.loss_tf, self.loss_spacereg_tf, self.loss_dirreg_tf, self.S.dbg_tensors],
                                                                                        feed_dict={
                                                                                                    self.S.init_activation_tf : self.S.input_pad(x),
                                                                                                    self.target_tf : y,
                                                                                                  }
                                                                      )
                totloss += _totloss
                spaceregloss += _spaceregloss
                dirregloss += _dirregloss
                # break
            if debug:
                for tensor,value in zip(self.S.dbg_tensors, _dbg):
                    print(tensor, ":", value)

            print('-----epoch{}-----'.format(t))
            print('totloss : ', totloss)
            print('spaceregloss : ', spaceregloss)
            print('dirregloss : ', dirregloss)

            y_pred, activation, pos, indir, outdir = self.sess.run([self.S.output, self.S.activation, self.S.position_tf, self.S.indir_tf, self.S.outdir_tf], feed_dict={self.S.init_activation_tf : self.S.input_pad(x)})
            for i in range(min(minibatch_size,4)):
                print("x:", x[i], "y_pred:", y_pred[i], "y_true:", y[i,:])

            if visualize:
                visualize_activations = False
                amin = np.amin(activation)
                a = activation[0] - amin
                a = a / np.amax(a)
                color = (outdir[0]+0.001) / (np.linalg.norm(outdir[0], axis=1, keepdims=True)+0.001)
                if visualize_activations:
                    color = color * ( 0.3 + 0.7* a.reshape((-1,1)) )
                self.visualizer.draw_np(pos[0], col=color)
