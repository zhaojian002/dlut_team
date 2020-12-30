import sys

import os

import time
import cv2
import random

import numpy as np

import tensorflow as tf

from PIL import Image

SIZE = 1280
count =0
WIDTH = 32

HEIGHT = 40

NUM_CLASSES = 41

iterations = 50

SAVER_DIR = "train-saver/carnumber/"

LETTERS_DIGITS = (
"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
"Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z","京", "闽", "粤", "苏", "沪", "浙","川")

license_num = ""

probability_max = "概率："

time_begin = time.time()
x = tf.placeholder(tf.float32, shape=[None, SIZE])

y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])


# 定义卷积函数

def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)

    L1_relu = tf.nn.relu(L1_conv + b)

    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')


# 定义全连接层函数

def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)





def distinguish_license_plate():
    global license_num
    global probability_max
    img = cv2.imread("test.jpg")  # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
    cv2.imshow('gray', img_gray)  # 显示图片

    # 2、将灰度图像二值化，设定阈值是100
    img_thre = img_gray
    cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV, img_thre)
    cv2.imshow('threshold', img_thre)

    # 3、保存黑白图片
    cv2.imwrite('test1.jpg', img_thre)

    # 4、分割字符
    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = img_thre.shape[0]
    width = img_thre.shape[1]
    white_max = 0
    black_max = 0

    for i in range(width):
        for j in range(height):
            if img_thre[j][i] == 255:
                img_thre[j][i] = 0
            else:
                img_thre[j][i] = 255
    cv2.imwrite('test2.jpg', img_thre)

    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if img_thre[j][i] == 255:
                s += 1
            if img_thre[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)


    arg = False  # False表示白底黑字；True表示黑底白字
    if black_max > white_max:
        arg = True


    # 分割图像




    n = 1
    start = 1
    end = 2
    end1 = 2
    start1 = 1
    global count
    img_thre = cv2.resize(img_thre, (328, 84), interpolation=cv2.INTER_CUBIC)
    while n<8:
        n += 1

        count += 1
        cj = img_thre[1:84, (n-2)*46+1:(n-1)*46+1]
        p0 = cv2.resize(cj, (32, 40), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('caijian', cj)
        cv2.imwrite('%s.bmp' % (count), p0)
        saver = tf.train.import_meta_graph("%smodel.ckpt.meta" % (SAVER_DIR))
        with tf.Session() as sess:

            model_file = tf.train.latest_checkpoint(SAVER_DIR)

            saver.restore(sess, model_file)

            # 第一个卷积层

            W_conv1 = sess.graph.get_tensor_by_name("W_conv1:0")

            b_conv1 = sess.graph.get_tensor_by_name("b_conv1:0")

            conv_strides = [1, 1, 1, 1]

            kernel_size = [1, 2, 2, 1]

            pool_strides = [1, 2, 2, 1]

            L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides,
                                 padding='SAME')

            # 第二个卷积层

            W_conv2 = sess.graph.get_tensor_by_name("W_conv2:0")

            b_conv2 = sess.graph.get_tensor_by_name("b_conv2:0")

            conv_strides = [1, 1, 1, 1]

            kernel_size = [1, 1, 1, 1]

            pool_strides = [1, 1, 1, 1]

            L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides,
                                 padding='SAME')

            # 全连接层

            W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")

            b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")

            h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])

            h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)

            # dropout

            keep_prob = tf.placeholder(tf.float32)

            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # readout层

            W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")

            b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")

            # 定义优化器和训练op

            conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

            path = '%s.bmp' % (count)

            img = Image.open(path)

            width2 = img.size[0]

            height2 = img.size[1]

            img_data = [[0] * SIZE for i in range(1)]

            for h in range(0, height2):

                for w in range(0, width2):

                    if img.getpixel((w, h)) <= 230:

                        img_data[0][w + h * width2] = 1

                    else:

                        img_data[0][w + h * width2] = 0

            result = sess.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})

            max1 = 0

            max2 = 0

            max3 = 0

            max1_index = 0

            max2_index = 0

            max3_index = 0

            for j in range(NUM_CLASSES):

                if result[0][j] > max1:
                    max1 = result[0][j]

                    max1_index = j

                    continue

                if (result[0][j] > max2) and (result[0][j] <= max1):
                    max2 = result[0][j]

                    max2_index = j

                    continue

                if (result[0][j] > max3) and (result[0][j] <= max2):
                    max3 = result[0][j]

                    max3_index = j

                    continue
            flag = 1
            if ord(LETTERS_DIGITS[max1_index]) <= 255 and count!=1:
                flag = 0
                license_num = license_num + LETTERS_DIGITS[max1_index]
                probability_max = probability_max + "  " + LETTERS_DIGITS[max1_index] + str(max1)
            elif ord(LETTERS_DIGITS[max2_index]) <= 255 and flag == 1 and count!=1:
                flag = 0
                license_num = license_num + LETTERS_DIGITS[max2_index]
                probability_max = probability_max + "  " + LETTERS_DIGITS[max2_index] + str(max2)
            elif ord(LETTERS_DIGITS[max3_index]) <= 255 and flag == 1 and count!=1:
                flag = 0
                license_num = license_num + LETTERS_DIGITS[max3_index]
                probability_max = probability_max + "  " + LETTERS_DIGITS[max3_index] + str(max3)
            else:
                license_num = license_num + LETTERS_DIGITS[max1_index]
                probability_max = probability_max + "  " + LETTERS_DIGITS[max1_index] + str(max1)
    print("车牌编号是: 【%s】" % license_num+"."+probability_max)
    return license_num+"."+probability_max
if __name__ == '__main__':
    distinguish_license_plate()