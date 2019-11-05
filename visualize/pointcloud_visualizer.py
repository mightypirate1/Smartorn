import tensorflow as tf
from visualize.o3d import visualizer

'''
Common methods for training/testing/predicting
For training, testing, and predicting, following methods are provided to be overridden.

on_(train|test|predict)_begin(self, logs=None)
Called at the beginning of fit/evaluate/predict.

on_(train|test|predict)_end(self, logs=None)
Called at the end of fit/evaluate/predict.

on_(train|test|predict)_batch_begin(self, batch, logs=None)
Called right before processing a batch during training/testing/predicting. Within this method, logs is a dict with batch and size available keys, representing the current batch number and the size of the batch.

on_(train|test|predict)_batch_end(self, batch, logs=None)
Called at the end of training/testing/predicting a batch. Within this method, logs is a dict containing the stateful metrics result.

Training specific methods
In addition, for training, following are provided.

on_epoch_begin(self, epoch, logs=None)
Called at the beginning of an epoch during training.

on_epoch_end(self, epoch, logs=None)
'''

class pointcloud_callback(tf.keras.callbacks.Callback):
    def __init__(self, dir_delta=0.01):
        super(pointcloud_callback,self).__init__()
        self.dir_delta = dir_delta
        self.initialized = False
    def initialize(self):
        self.pos = self.model.layer.position_tf[0]
        self.dir = self.model.layer.outdir_tf[0]
        self.col = self.model.layer.indir_tf[0]
        #
        ###
        self.V = visualizer(
                            dir_delta=self.dir_delta,
                            window_name='Smartorn',
                            width=1024,
                            height=760,
                            left=50,
                            top=50,
                            visible=True
                            )
        self.initialized = True
        ###
        #
    def on_epoch_end(self, epoch, logs):
        if not self.initialized:
            self.initialize()
        self.V.draw_np(
                        self.pos.numpy(),
                        col=self.col.numpy(),
                        dir=self.dir.numpy()
                        )
