# Use a trained I_encoder Net to generate feature maps

import tensorflow as tf
import numpy as np
from datetime import datetime
import h5py
import copy
import matplotlib.pyplot as plt
from scipy.misc import imread,imsave
from skimage import color

from dense_net import DenseFuseNet

patch_size = 84
CHANNELS = 1 # gray scale, default

LEARNING_RATE = 0.0002
EPSILON = 1e-5
DECAY_RATE = 0.9
eps = 1e-8


def generate_part1(source_imgs, model_path, BATCH_SIZE):
    _handler(source_imgs, model_path, BATCH_SIZE)


def _handler(source_imgs, model_path, BATCH_SIZE):
    num_imgs = source_imgs.shape[0]
    mod = num_imgs % BATCH_SIZE
    n_batches = int(num_imgs // BATCH_SIZE)
    print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        source_imgs = source_imgs[:-mod]

    with tf.Graph().as_default(), tf.Session() as sess:
        SOURCE_VIS = tf.placeholder(tf.float32, shape=(BATCH_SIZE, patch_size, patch_size, 1), name='SOURCE_VIS')
        SOURCE_IR = tf.placeholder(tf.float32, shape=(BATCH_SIZE, patch_size, patch_size, 1), name='SOURCE_IR')
        dfn = DenseFuseNet("DenseFuseNet")
        f11, f12, f13, f14, f21, f22, f23, f24 = dfn.transform_test_part1(SOURCE_VIS, SOURCE_IR)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
#
        f11 = sess.run(f11)
        f12 = sess.run(f12)
        f13 = sess.run(f13)
        f14 = sess.run(f14)
        f21 = sess.run(f21)
        f22 = sess.run(f22)
        f23 = sess.run(f23)
        f24 = sess.run(f24)

#######1111111111111111111111111111
        for batch in range(n_batches):
            # print('n_batches:',n_batches)
            VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
            IR_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
            VIS_batch = np.expand_dims(VIS_batch, -1)
            IR_batch = np.expand_dims(IR_batch, -1)
            FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch}

            t_f11 = sess.run(f11, feed_dict=FEED_DICT)
            t_f11=np.array([t_f11])

            if batch==0:
                dataset_f11 = t_f11
            elif batch<=100:
                dataset_f11=np.concatenate([dataset_f11,t_f11],axis=0)
            if batch % 10 == 0 and batch<=100:
                print('*******************')
                print('batch_for11: ', batch)
                print('dataset_f11:', dataset_f11.shape)

            if batch>=101:
                if batch==101:
                    dataset_f11_add1=t_f11
                elif batch<=200:
                    dataset_f11_add1 = np.concatenate([dataset_f11_add1, t_f11], axis=0)
                elif batch==201:
                    dataset_f11_add2 = t_f11
                else:
                    dataset_f11_add2 = np.concatenate([dataset_f11_add2, t_f11], axis=0)
                if (batch-100) % 10 == 0 and batch<=200:
                    print('*******************')
                    print('batch_for11: ', batch)
                    print('dataset_f11_add1:', dataset_f11_add1.shape)
                elif (batch-200) % 10 == 0:
                    print('*******************')
                    print('batch_for11: ', batch)
                    print('dataset_f11_add1:', dataset_f11_add1.shape)
                    print('dataset_f11_add2:', dataset_f11_add2.shape)

        f1 = h5py.File('feature/f11.h5', 'w')
        f1.create_dataset('f11', data=dataset_f11)
        print('dataset_f11 finish!')
        f1.close()

        f1 = h5py.File('feature/f11.h5', 'a')
        f1.create_dataset('f11_101_200', data=dataset_f11_add1)
        print('dataset_f11_add1 finish!')
        f1.close()

        f1 = h5py.File('feature/f11.h5', 'a')
        f1.create_dataset('f11_201_241', data=dataset_f11_add2)
        print('dataset_f11_add2 finish!')
        f1.close()

#######2222222222222222222222222222
        for batch in range(n_batches):
            VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
            IR_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
            VIS_batch = np.expand_dims(VIS_batch, -1)
            IR_batch = np.expand_dims(IR_batch, -1)
            FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch}

            t_f12 = sess.run(f12, feed_dict=FEED_DICT)
            t_f12=np.array([t_f12])
            if batch==0:
                dataset_f12 = t_f12
            elif batch<=100:
                dataset_f12=np.concatenate([dataset_f12,t_f12],axis=0)
            if batch % 10 == 0 and batch<=100:
                print('*******************')
                print('batch_for12: ', batch)
                print('dataset_f12:', dataset_f12.shape)

            if batch>=101:
                if batch==101:
                    dataset_f12_add1=t_f12
                elif batch<=200:
                    dataset_f12_add1 = np.concatenate([dataset_f12_add1, t_f12], axis=0)
                elif batch==201:
                    dataset_f12_add2 = t_f12
                else:
                    dataset_f12_add2 = np.concatenate([dataset_f12_add2, t_f12], axis=0)
                if (batch-100) % 10 == 0 and batch<=200:
                    print('*******************')
                    print('batch_for12: ', batch)
                    print('dataset_f12_add1:', dataset_f12_add1.shape)
                elif (batch-200) % 10 == 0:
                    print('*******************')
                    print('batch_for12: ', batch)
                    print('dataset_f12_add1:', dataset_f12_add1.shape)
                    print('dataset_f12_add2:', dataset_f12_add2.shape)

        f2 = h5py.File('feature/f12.h5', 'w')
        f2.create_dataset('f12', data=dataset_f12)
        print('dataset_f12 finish!')
        f2.close()

        f2 = h5py.File('feature/f12.h5', 'a')
        f2.create_dataset('f12_101_200', data=dataset_f12_add1)
        print('dataset_f12_add1 finish!')
        f2.close()

        f2 = h5py.File('feature/f12.h5', 'a')
        f2.create_dataset('f12_201_241', data=dataset_f12_add2)
        print('dataset_f12_add2 finish!')
        f2.close()

#######333333333333333333333333333333
        for batch in range(n_batches):
            VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
            IR_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
            VIS_batch = np.expand_dims(VIS_batch, -1)
            IR_batch = np.expand_dims(IR_batch, -1)
            FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch}

            t_f13 = sess.run(f13, feed_dict=FEED_DICT)
            t_f13 = np.array([t_f13])
            if batch == 0:
                dataset_f13 = t_f13
            elif batch <= 100:
                dataset_f13 = np.concatenate([dataset_f13, t_f13], axis=0)
            if batch % 10 == 0 and batch <= 100:
                print('*******************')
                print('batch_for13: ', batch)
                print('dataset_f13:', dataset_f13.shape)

            if batch >= 101:
                if batch == 101:
                    dataset_f13_add1 = t_f13
                elif batch <= 200:
                    dataset_f13_add1 = np.concatenate([dataset_f13_add1, t_f13], axis=0)
                elif batch == 201:
                    dataset_f13_add2 = t_f13
                else:
                    dataset_f13_add2 = np.concatenate([dataset_f13_add2, t_f13], axis=0)
                if (batch - 100) % 10 == 0 and batch <= 200:
                    print('*******************')
                    print('batch_for13: ', batch)
                    print('dataset_f13_add1:', dataset_f13_add1.shape)
                elif (batch - 200) % 10 == 0:
                    print('*******************')
                    print('batch_for13: ', batch)
                    print('dataset_f13_add1:', dataset_f13_add1.shape)
                    print('dataset_f13_add2:', dataset_f13_add2.shape)

        f3 = h5py.File('feature/f13.h5', 'w')
        f3.create_dataset('f13', data=dataset_f13)
        print('dataset_f13 finish!')
        f3.close()

        f3 = h5py.File('feature/f13.h5', 'a')
        f3.create_dataset('f13_101_200', data=dataset_f13_add1)
        print('dataset_f13_add1 finish!')
        f3.close()

        f3 = h5py.File('feature/f13.h5', 'a')
        f3.create_dataset('f13_201_241', data=dataset_f13_add2)
        print('dataset_f13_add2 finish!')
        f3.close()

#######4444444444444444444444444444
        for batch in range(n_batches):
            VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
            IR_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
            VIS_batch = np.expand_dims(VIS_batch, -1)
            IR_batch = np.expand_dims(IR_batch, -1)
            FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch}

            t_f14 = sess.run(f14, feed_dict=FEED_DICT)
            t_f14 = np.array([t_f14])
            if batch == 0:
                dataset_f14 = t_f14
            elif batch <= 100:
                dataset_f14 = np.concatenate([dataset_f14, t_f14], axis=0)
            if batch % 10 == 0 and batch <= 100:
                print('*******************')
                print('batch_for14: ', batch)
                print('dataset_f14:', dataset_f14.shape)

            if batch >= 101:
                if batch == 101:
                    dataset_f14_add1 = t_f14
                elif batch <= 200:
                    dataset_f14_add1 = np.concatenate([dataset_f14_add1, t_f14], axis=0)
                elif batch == 201:
                    dataset_f14_add2 = t_f14
                else:
                    dataset_f14_add2 = np.concatenate([dataset_f14_add2, t_f14], axis=0)
                if (batch - 100) % 10 == 0 and batch <= 200:
                    print('*******************')
                    print('batch_for14: ', batch)
                    print('dataset_f14_add1:', dataset_f14_add1.shape)
                elif (batch - 200) % 10 == 0:
                    print('*******************')
                    print('batch_for14: ', batch)
                    print('dataset_f14_add1:', dataset_f14_add1.shape)
                    print('dataset_f14_add2:', dataset_f14_add2.shape)

        f4 = h5py.File('feature/f14.h5', 'w')
        f4.create_dataset('f14', data=dataset_f14)
        print('dataset_f14 finish!')
        f4.close()

        f4 = h5py.File('feature/f14.h5', 'a')
        f4.create_dataset('f14_101_200', data=dataset_f14_add1)
        print('dataset_f14_add1 finish!')
        f4.close()

        f4 = h5py.File('feature/f14.h5', 'a')
        f4.create_dataset('f14_201_241', data=dataset_f14_add2)
        print('dataset_f14_add2 finish!')
        f4.close()

#####555555555555555555555555555555
        for batch in range(n_batches):
            VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
            IR_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
            VIS_batch = np.expand_dims(VIS_batch, -1)
            IR_batch = np.expand_dims(IR_batch, -1)
            FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch}

            t_f21 = sess.run(f21, feed_dict=FEED_DICT)
            t_f21 = np.array([t_f21])
            if batch == 0:
                dataset_f21 = t_f21
            elif batch <= 100:
                dataset_f21 = np.concatenate([dataset_f21, t_f21], axis=0)
            if batch % 10 == 0 and batch <= 100:
                print('*******************')
                print('batch_for21: ', batch)
                print('dataset_f21:', dataset_f21.shape)

            if batch >= 101:
                if batch == 101:
                    dataset_f21_add1 = t_f21
                elif batch <= 200:
                    dataset_f21_add1 = np.concatenate([dataset_f21_add1, t_f21], axis=0)
                elif batch == 201:
                    dataset_f21_add2 = t_f21
                else:
                    dataset_f21_add2 = np.concatenate([dataset_f21_add2, t_f21], axis=0)
                if (batch - 100) % 10 == 0 and batch <= 200:
                    print('*******************')
                    print('batch_for21: ', batch)
                    print('dataset_f21_add1:', dataset_f21_add1.shape)
                elif (batch - 200) % 10 == 0:
                    print('*******************')
                    print('batch_for21: ', batch)
                    print('dataset_f21_add1:', dataset_f21_add1.shape)
                    print('dataset_f21_add2:', dataset_f21_add2.shape)

        f5 = h5py.File('feature/f21.h5', 'w')
        f5.create_dataset('f21', data=dataset_f21)
        print('dataset_f21 finish!')
        f5.close()

        f5 = h5py.File('feature/f21.h5', 'a')
        f5.create_dataset('f21_101_200', data=dataset_f21_add1)
        print('dataset_f21_add1 finish!')
        f5.close()

        f5 = h5py.File('feature/f21.h5', 'a')
        f5.create_dataset('f21_201_241', data=dataset_f21_add2)
        print('dataset_f21_add2 finish!')
        f5.close()

######666666666666666666666666666666
        for batch in range(n_batches):
            # print('n_batches:',n_batches)
            VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
            IR_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
            VIS_batch = np.expand_dims(VIS_batch, -1)
            IR_batch = np.expand_dims(IR_batch, -1)
            FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch}

            t_f22 = sess.run(f22, feed_dict=FEED_DICT)
            t_f22 = np.array([t_f22])
            if batch == 0:
                dataset_f22 = t_f22
            elif batch <= 100:
                dataset_f22 = np.concatenate([dataset_f22, t_f22], axis=0)
            if batch % 10 == 0 and batch <= 100:
                print('*******************')
                print('batch_for22: ', batch)
                print('dataset_f22:', dataset_f22.shape)

            if batch >= 101:
                if batch == 101:
                    dataset_f22_add1 = t_f22
                elif batch <= 200:
                    dataset_f22_add1 = np.concatenate([dataset_f22_add1, t_f22], axis=0)
                elif batch == 201:
                    dataset_f22_add2 = t_f22
                else:
                    dataset_f22_add2 = np.concatenate([dataset_f22_add2, t_f22], axis=0)
                if (batch - 100) % 10 == 0 and batch <= 200:
                    print('*******************')
                    print('batch_for22: ', batch)
                    print('dataset_f22_add1:', dataset_f22_add1.shape)
                elif (batch - 200) % 10 == 0:
                    print('*******************')
                    print('batch_for22: ', batch)
                    print('dataset_f22_add1:', dataset_f22_add1.shape)
                    print('dataset_f22_add2:', dataset_f22_add2.shape)

        f6 = h5py.File('feature/f22.h5', 'w')
        f6.create_dataset('f22', data=dataset_f22)
        print('dataset_f22 finish!')
        f6.close()

        f6 = h5py.File('feature/f22.h5', 'a')
        f6.create_dataset('f22_101_200', data=dataset_f22_add1)
        print('dataset_f22_add1 finish!')
        f6.close()

        f6 = h5py.File('feature/f22.h5', 'a')
        f6.create_dataset('f22_201_241', data=dataset_f22_add2)
        print('dataset_f22_add2 finish!')
        f6.close()

#######777777777777777777777777777777
        for batch in range(n_batches):
            # print('n_batches:',n_batches)
            VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
            IR_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
            VIS_batch = np.expand_dims(VIS_batch, -1)
            IR_batch = np.expand_dims(IR_batch, -1)
            FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch}

            t_f23 = sess.run(f23, feed_dict=FEED_DICT)
            t_f23 = np.array([t_f23])
            if batch == 0:
                dataset_f23 = t_f23
            elif batch <= 100:
                dataset_f23 = np.concatenate([dataset_f23, t_f23], axis=0)
            if batch % 10 == 0 and batch <= 100:
                print('*******************')
                print('batch_for23: ', batch)
                print('dataset_f23:', dataset_f23.shape)

            if batch >= 101:
                if batch == 101:
                    dataset_f23_add1 = t_f23
                elif batch <= 200:
                    dataset_f23_add1 = np.concatenate([dataset_f23_add1, t_f23], axis=0)
                elif batch == 201:
                    dataset_f23_add2 = t_f23
                else:
                    dataset_f23_add2 = np.concatenate([dataset_f23_add2, t_f23], axis=0)
                if (batch - 100) % 10 == 0 and batch <= 200:
                    print('*******************')
                    print('batch_for23: ', batch)
                    print('dataset_f23_add1:', dataset_f23_add1.shape)
                elif (batch - 200) % 10 == 0:
                    print('*******************')
                    print('batch_for23: ', batch)
                    print('dataset_f23_add1:', dataset_f23_add1.shape)
                    print('dataset_f23_add2:', dataset_f23_add2.shape)

        f7 = h5py.File('feature/f23.h5', 'w')
        f7.create_dataset('f23', data=dataset_f23)
        print('dataset_f23 finish!')
        f7.close()

        f7 = h5py.File('feature/f23.h5', 'a')
        f7.create_dataset('f23_101_200', data=dataset_f23_add1)
        print('dataset_f23_add1 finish!')
        f7.close()

        f7 = h5py.File('feature/f23.h5', 'a')
        f7.create_dataset('f23_201_241', data=dataset_f23_add2)
        print('dataset_f23_add2 finish!')
        f7.close()

#######888888888888888888888888888888
        for batch in range(n_batches):
            # print('n_batches:',n_batches)
            VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
            IR_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
            VIS_batch = np.expand_dims(VIS_batch, -1)
            IR_batch = np.expand_dims(IR_batch, -1)
            FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch}

            t_f24 = sess.run(f24, feed_dict=FEED_DICT)
            t_f24 = np.array([t_f24])
            if batch == 0:
                dataset_f24 = t_f24
            elif batch <= 100:
                dataset_f24 = np.concatenate([dataset_f24, t_f24], axis=0)
            if batch % 10 == 0 and batch <= 100:
                print('*******************')
                print('batch_for24: ', batch)
                print('dataset_f24:', dataset_f24.shape)

            if batch >= 101:
                if batch == 101:
                    dataset_f24_add1 = t_f24
                elif batch <= 200:
                    dataset_f24_add1 = np.concatenate([dataset_f24_add1, t_f24], axis=0)
                elif batch == 201:
                    dataset_f24_add2 = t_f24
                else:
                    dataset_f24_add2 = np.concatenate([dataset_f24_add2, t_f24], axis=0)
                if (batch - 100) % 10 == 0 and batch <= 200:
                    print('*******************')
                    print('batch_for24: ', batch)
                    print('dataset_f24_add1:', dataset_f24_add1.shape)
                elif (batch - 200) % 10 == 0:
                    print('*******************')
                    print('batch_for24: ', batch)
                    print('dataset_f24_add1:', dataset_f24_add1.shape)
                    print('dataset_f24_add2:', dataset_f24_add2.shape)

        f8 = h5py.File('feature/f24.h5', 'w')
        f8.create_dataset('f24', data=dataset_f24)
        print('dataset_f24 finish!')
        f8.close()

        f8 = h5py.File('feature/f24.h5', 'a')
        f8.create_dataset('f24_101_200', data=dataset_f24_add1)
        print('dataset_f24_add1 finish!')
        f8.close()

        f8 = h5py.File('feature/f24.h5', 'a')
        f8.create_dataset('f24_201_241', data=dataset_f24_add2)
        print('dataset_f24_add2 finish!')
        f8.close()

        print('finish!')


