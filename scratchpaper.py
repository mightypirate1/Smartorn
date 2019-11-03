import tensorflow as tf
import numpy as np

from nn.smartorn import smartorn
from nn.smartorn_trainer import trainer
import data.logic
import utils.utils as utils

b = smartorn(
             [None, 2],
             [None, 1],
             n_neurons=10,
             n_layers=1,
             depth_decay=True,
             radial_decay=False,
             dtype=tf.float32,
             trainable_bias=False,
            )
with tf.Session() as session:
    #Setup
    T = trainer(b,session)
    session.run(tf.global_variables_initializer())

    x,y = data.logic.make(10000,'xor')
    T.train(
            x,
            y,
            epochs=1000,
            minibatch_size=32,
            visualize=True,
            debug=False
            )


    #DEBUG TESTS...
    #Variables
    x = np.array([[1,1], [0,1], [1,0], [0,0]])
    y_true = [[1], [0], [0], [0]]

    #Feed
    feed_dict = {
                 b.init_activation_tf : b.input_pad(x),
                 T.target_tf : y_true,
                }

    loss, target, output, activation, dbg = session.run( [T.loss_error_tf, T.target_tf, b.output, b.activation, b.dbg_tensors], feed_dict=feed_dict)

    for o,t,a in zip(output, target, activation):
        print('---sample---')
        print('activation:',a)
        print('output:',o)
        print('target:',t)

    print(loss)
    # exit()
    print("-----")
    for d in dbg:
        print(d)
