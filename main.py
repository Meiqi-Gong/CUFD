# Demo - train the DenseFuse network & use it to generate an image

from __future__ import print_function

from train_part1 import train_part1
from train_part2 import train_part2
from generate_param import generate_part1
from generate import generate
import h5py
import time
import numpy as np
import os

BATCH_SIZE = 84
EPOCHES = 5
LOGGING = 40
MODEL_SAVE_PATH = './model/'
IS_TRAINING = False
f = h5py.File('vis_ir_orig.h5', 'r')
sources = f['data'][:]
sources = np.transpose(sources, (0, 3, 2, 1))
model_path_1 = './model/1part1_model.ckpt'
model_path_2 = './model/feature/epoch3/part2_model.ckpt'
save_path='./result/'

def main():
    if IS_TRAINING:
        # print(('\nBegin to train the network ...\n'))
        # train_part1(sources, MODEL_SAVE_PATH, EPOCHES, BATCH_SIZE, logging_period=LOGGING)
        # generate_part1(sources, model_path_1, BATCH_SIZE)
        train_part2(sources, MODEL_SAVE_PATH, EPOCHES, BATCH_SIZE, logging_period=LOGGING)
    else:
        path_Road = '/data/gmq/dense/RoadScene'
        path_TNO40 = '/data/gmq/exe/test_imgs/TNO40/'
        Time=[]

        for i in range(10):
            index = i + 1
            infrared = path_TNO40 + 'IR/' + str(index) + '.bmp'
            visible = path_TNO40 + 'VIS/' + str(index) + '.bmp'
            # infrared = path_Road + '/IR/' + str(index) + '.jpg'
            # visible = path_Road + '/VIS/' + str(index) + '.jpg'
            savepath = './results/'
            begin = time.time()
            generate(infrared, visible, model_path_1, model_path_2, index, output_path=savepath)
            end = time.time()
            Time.append(end-begin)
            print("pic_num:%s" % index)
            print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))


if __name__ == '__main__':
    main()

