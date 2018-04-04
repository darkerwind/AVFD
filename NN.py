import numpy        as np
import tensorflow   as tf

from Globals        import *

class NeuralNetwork():
    __CLS_DIC_NNVARIABLES = {}

    def __init__(self):
        pass
    
    @classmethod
    def __makeNeuralNetwork(cls):
        cls.__CLS_DIC_NNVARIABLES[NN_SESSION] = tf.Session()

        X = tf.placeholder(tf.float32, [1, 720, 1280, 6], name="X")
        Y = tf.placeholder(tf.float32, [1, 720, 1280, 3], name="Y")
        KP = tf.placeholder(tf.float32, name="KP")

        X_Norm = tf.divide(X, 255.0)

        D1W1 = tf.Variable(tf.random_normal([3, 3, 6, 6], stddev=0.1), name="D1W1")
        D1L1 = tf.nn.conv2d(X_Norm, D1W1, strides=[1, 1, 1, 1], padding='SAME', name="D1L1")
        D1W2 = tf.Variable(tf.random_normal([3, 3, 6, 6], stddev=0.1), name="D1W1")
        D1L2 = tf.nn.conv2d(D1L1  , D1W2, strides=[1, 1, 1, 1], padding='SAME', name="D1L1")
        D1L2 = tf.concat([X_Norm, D1L2], axis=3, name="DENSE1")

        D2W1 = tf.Variable(tf.random_normal([3, 3, 12, 12], stddev=0.1), name="D2W1")
        D2L1 = tf.nn.conv2d(D1L2  , D2W1, strides=[1, 1, 1, 1], padding='SAME', name="D2L1")
        D2W2 = tf.Variable(tf.random_normal([3, 3, 12, 12], stddev=0.1), name="D2W2")
        D2L2 = tf.nn.conv2d(D2L1  , D2W2, strides=[1, 1, 1, 1], padding='SAME', name="D2L2")
        D2L2 = tf.concat([D1L2, D2L2], axis=3, name="DENSE2")

        D3W1 = tf.Variable(tf.random_normal([3, 3, 24, 24], stddev=0.1), name="D3W1")
        D3L1 = tf.nn.conv2d(D2L2  , D3W1, strides=[1, 1, 1, 1], padding='SAME', name="D3L1")
        D3W2 = tf.Variable(tf.random_normal([3, 3, 24, 24], stddev=0.1), name="D3W2")
        D3L2 = tf.nn.conv2d(D3L1  , D3W2, strides=[1, 1, 1, 1], padding='SAME', name="D3L2")
        D3L2 = tf.concat([D2L2, D3L2], axis=3, name="DENSE3")

        W1 = tf.Variable(tf.random_normal([3, 3, 48, 96], stddev=0.1), name="W1")
        L1 = tf.nn.conv2d(D3L2, W1, strides=[1, 1, 1, 1], padding='SAME', name="L1")
        #L1 = tf.nn.relu(L1, name="L1_Relu")
        L1 = tf.nn.dropout(L1, KP, name="L1_DROPOUT")

        W2 = tf.Variable(tf.random_normal([3, 3, 96, 48], stddev=0.1), name="W2")
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME', name="L2")
        L2 = tf.nn.relu(L2, name="L2_Relu")
        L2 = tf.nn.dropout(L2, KP, name="L2_DROPOUT")

        W3 = tf.Variable(tf.random_normal([3, 3, 48, 24], stddev=0.1), name="W3")
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME', name="L3")
        #L3 = tf.nn.relu(L3, name="L3_Relu")
        L3 = tf.nn.dropout(L3, KP, name="L3_DROPOUT")

        W4 = tf.Variable(tf.random_normal([3, 3, 24, 12], stddev=0.1), name="W4")
        L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME', name="L4")
        L4 = tf.nn.relu(L4, name="L4_Relu")
        L4 = tf.nn.dropout(L4, KP, name="L4_DROPOUT")

        W5 = tf.Variable(tf.random_normal([3, 3, 12, 3], stddev=0.1), name="W5")
        M = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME', name="M")
        #M = tf.nn.relu(M, name="M_Relu")

        #MaxValue = tf.reduce_max(M)
        #if tf.greater(MaxValue, tf.constant(1.0, tf.float32)) is not None:
        #    M = tf.divide(M, MaxValue, name="M_Normalike")
        M = tf.multiply(M, 255.0)
        #print("Model Shape =", M.shape)

        globalStep = tf.Variable(0, trainable=False, name="global_step")

        with tf.name_scope("COST_OPTIMIZER"):
            cost = tf.reduce_sum(tf.abs(tf.divide(tf.subtract(M, Y), 255.0)))
            #cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=M))
            optmz = tf.train.AdamOptimizer(0.0005).minimize(cost, global_step=globalStep)

            tf.summary.scalar("COST", cost)

        cls.__CLS_DIC_NNVARIABLES[NN_COST]       = cost
        cls.__CLS_DIC_NNVARIABLES[NN_OPTIMIZER]  = optmz
        cls.__CLS_DIC_NNVARIABLES[NN_MODEL]      = M
        cls.__CLS_DIC_NNVARIABLES[NN_X]          = X
        cls.__CLS_DIC_NNVARIABLES[NN_Y]          = Y
        cls.__CLS_DIC_NNVARIABLES[NN_KP]         = KP
        cls.__CLS_DIC_NNVARIABLES[NN_STEP]       = globalStep

    @classmethod
    def __restoreNeuralNetwork(cls):
        sess = cls.__CLS_DIC_NNVARIABLES[NN_SESSION]

        saver = tf.train.Saver(tf.global_variables())
        cls.__CLS_DIC_NNVARIABLES[NN_SAVER] = saver

        ckpt = tf.train.get_checkpoint_state("./Model")
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Restore Check-Point.")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Initialize Global Variables.")
            sess.run(tf.global_variables_initializer())
    
    @classmethod
    def __makeSummary(cls):
        sess = cls.__CLS_DIC_NNVARIABLES[NN_SESSION]

        cls.__CLS_DIC_NNVARIABLES[NN_MERGED] = tf.summary.merge_all()
        cls.__CLS_DIC_NNVARIABLES[NN_WRITER] = tf.summary.FileWriter("./Logs", sess.graph)
    
    @classmethod
    def __initializeNetwork(cls):
        cls.__makeNeuralNetwork()
        cls.__restoreNeuralNetwork()
        cls.__makeSummary()

    @classmethod
    def getNNVariables(cls):
        if len(cls.__CLS_DIC_NNVARIABLES) == 0:
            cls.__initializeNetwork()
        
        return cls.__CLS_DIC_NNVARIABLES
