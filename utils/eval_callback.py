import tensorflow as tf
import numpy as np
class eval_callback(tf.keras.callbacks.Callback):
    def __init__(self, x,y):
        super(eval_callback,self).__init__()
        self.x, self.y = x, y
        self.n = len(self.x) if type(self.x) is list else self.x.shape[0]
    def on_epoch_begin(self, epoch, logs):
        if self.n > 10:
            idx = np.random.permutation(n)[:10]
        else:
            idx = list(range(self.n))
        X = self.x[idx]
        Y = self.y[idx]
        y_pred = self.model(X).numpy()
        print(flush=True)
        print("Epoch {} eval:\nX\ty_true\ty_pred".format(epoch))
        for x,yt,yp in zip(X,Y,y_pred):
            print(x,yt,yp)
