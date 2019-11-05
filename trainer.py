import tensorflow as tf
import numpy as np

from nn.smartorn import smartorn
import visualize.pointcloud_visualizer as pcv
import data.logic
import utils.utils as utils
from utils.eval_callback import eval_callback

class trainer:
    def __init__(self, model, data, lr=1*10**-4):
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer, model.smartorn_loss)
        self.optimizer = optimizer
        self.model = model #type(model) == smartorn
        self.data = data
        self.eval_x, i = np.unique(data[0], axis=0, return_index=True)
        self.eval_y = data[1][i]
        #
        ###
        #####
        self.callbacks = [
                            tf.keras.callbacks.TensorBoard(log_dir="./logs"),
                            pcv.pointcloud_callback(dir_delta=0.01),
                            eval_callback(self.eval_x, self.eval_y)
                         ]
    def train(self, epochs, batch_size=256, verbose=1):
        self.model.fit(*self.data, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=self.callbacks)
    def eval(self):
        y_pred = self.model(self.eval_x)[-1].numpy()
        print("X\ty_true\ty_pred")
        for x,yt,yp in zip(self.eval_x,self.eval_y,y_pred):
            print(x,yt,yp)
    def label(self,samples):
        _ret = []
        for sample in samples:
            for i,w in enumerate(zip(*self.data)):
                x,y = w
                if (x == sample).all():
                    _ret.append(y)
                    break
        return np.array(_ret)
    def print_weights(self):
        for weight in self.model.trainable_weights:
            print("-----<{}>-----".format(weight.name))
            print(weight.numpy())
#####
#####
#####

if __name__ == "__main__":
    D = data.logic.make(100000,'xor')
    M = smartorn(
                 1, #n outputs
                 #
                 dtype=tf.float32,
                 trainable_bias=False,
                 trainable_input_position=False,
                 #
                 n_neurons=10,
                 n_layers=3,
                 n_dimentions=3,
                 #
                 radial_decay=False,
                 depth_decay=False,
                 #
                 renormalize_activations=False,
                 reapply_inputs=True,
                 #
                 reg_amount= {
                                "spatial" : 3.1,
                                "direction" : 0.05,
                                "output"   : 0.0,
                                "entropy" : None,
                              },
                )

    T = trainer(M,D)
    T.train(1000)
    T.eval()
    T.print_weights()

    #DEBUG TESTS...
    #Variables
    # print(b.losses);input("losses?")
