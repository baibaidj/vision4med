# coding=utf-8
# @Time	  : 2019-04-09 17:05
# @Author   : Monolith
# @FileName : BiSeNet.py


# coding=utf-8

import tensorflow as tf
from nets.xception_39_time_series import xception_39_series
from keras import models
from keras.layers import *
import keras.backend as K
from keras.regularizers import l2
import numpy as np
import os, sys


def bilinear_upsameple(tensor, size):
    y = tf.image.resize_bilinear(images=tensor, size=size)
    return y


def upsampling_series(input_tensor, scale):
    # dims = K.int_shape(input_tensor)
    # net = tf.keras.layers.UpSampling2D
    # net = layers.Lambda(lambda x: bilinear_upsameple(tensor=x, size=(scale, scale)))(input_tensor)
    net = TimeDistributed(UpSampling2D(size=(scale, scale), interpolation='bilinear'))(input_tensor) #, interpolation='bilinear'
    return net


def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successively Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = BatchNormalization()(inputs)
    net = Activation('relu')(net)
    net = Conv2DTranspose(n_filters,
                                 kernel_size=kernel_size,
                                 strides=[scale, scale],
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=l2(1e-4)
                                 )(net)
    return net


def conv_lstm_block(inputs, n_filters, kernel_size=(3, 3), strides=1, act = 'relu'):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = ConvLSTM2D(n_filters, kernel_size, strides=[strides, strides],
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-4),
                     return_sequences=True)(inputs)
    net = TimeDistributed(BatchNormalization())(net)
    net = Activation(act)(net)
    return net

def conv_block_series(inputs, n_filters, kernel_size=(3, 3), strides=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = TimeDistributed(Conv2D(n_filters, kernel_size, strides=[strides, strides],
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=l2(1e-4)))(inputs)
    net = TimeDistributed(BatchNormalization())(net)
    net = TimeDistributed(Activation('relu'))(net)
    return net


def multiply_with_broadcast(tensor_list):
    inputs, attention_weights = tensor_list
    b, t, h, w, c = K.int_shape(inputs)
    weight_dim = len(K.int_shape(attention_weights))
    print(weight_dim == 3)
    net = K.expand_dims(attention_weights, axis=2)
    if weight_dim == 3:
        net = K.expand_dims(net, axis=2)
    net = K.repeat_elements(net, rep=h, axis=2)
    net = K.repeat_elements(net, rep=w, axis=3)
    out = inputs * net
    return out






def AttentionRefinementModule(inputs, n_filters):
    # Global average pooling
    net = TimeDistributed(GlobalAveragePooling2D())(inputs)
    net = Lambda(lambda x : K.expand_dims(x, axis=2))(net)
    # net = K.mean(inputs, axis = [1,2], keepdims=False)
    # _, t, _, _, c = K.int_shape(inputs)
    # print(t, type(t), c, type(c))
    # net = Reshape((nb_times, 1, K.int_shape(inputs)[-1]))(net)
    # print(net.shape)
    net = TimeDistributed(Conv1D(n_filters, kernel_size=1,
                        kernel_initializer='he_normal',
                        ))(net)
    net = TimeDistributed(BatchNormalization())(net)
    # net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    # net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    # net = slim.batch_norm(net, fused=True)
    net = Activation('sigmoid')(net)#tf.sigmoid(net)
    # net = TimeDistributed(Multiply())([inputs, net])
    # net = Multiply()([inputs, net])
    net = Lambda(multiply_with_broadcast)([inputs, net])

    return net


def FeatureFusionModule(input_1, input_2, n_filters):
    inputs = Concatenate(axis=-1)([input_1, input_2])
    inputs = conv_lstm_block(inputs, n_filters=n_filters, kernel_size=(3, 3))

    # Global average pooling
    # _, t, _, _, c = K.int_shape(inputs)
    net = TimeDistributed(GlobalAveragePooling2D())(inputs)

    net = Lambda(lambda x : K.expand_dims(x, axis=2))(net)
    net = TimeDistributed(Conv1D(n_filters, kernel_size=1,
                        kernel_initializer='he_normal'
                        ))(net)
    net = Activation('relu')(net)
    net = TimeDistributed(Conv1D(n_filters, kernel_size=1,
                        kernel_initializer='he_normal',
                        ))(net)
    net = Activation('sigmoid')(net)
    # net = Multiply()([inputs, net])
    net = Lambda(multiply_with_broadcast)([inputs, net])

    out = Add()([inputs, net])

    return out


def bisenet_convlstm(num_classes, input_shape = (12,32*8, 32*9), size_factor = 1):
    """
    Builds the BiSeNet model.

    Arguments:
      num_classes: Number of classes

    Returns:
      BiSeNet model
    """

    if num_classes == 2:
        activation = 'sigmoid'
        nb_classes = 1
    else:
        activation = 'softmax'
        nb_classes = num_classes + 1
    # The spatial path
    # The number of feature maps for each convolution is not specified in the paper
    # It was chosen here to be equal to the number of feature maps of a classification
    # model at each corresponding stage


    with tf.device('/device:GPU:0'):
        # inputs = Input((None, None, None, 1))
        inputs = Input(input_shape + tuple([1]))
        spatial_net = conv_block_series(inputs, n_filters=16 * size_factor, kernel_size=[3, 3], strides=2)
        print('spatial path 0 %s' % (list(K.get_variable_shape(spatial_net))))
        spatial_net = conv_block_series(spatial_net, n_filters=32 * size_factor, kernel_size=[3, 3], strides=2)
        print('spatial path 1 %s' % (list(K.get_variable_shape(spatial_net))))
        spatial_net = conv_block_series(spatial_net, n_filters=64 * size_factor, kernel_size=[3, 3], strides=2)
        print('spatial path 2 %s' %(list(K.get_variable_shape(spatial_net))))
    ### Context path
    # logits, end_points, frontend_scope, init_fn = frontend_builder.build_frontend(inputs, frontend,
    #                                                                               pretrained_dir=pretrained_dir,
    #                                                                               is_training=is_training)
    with tf.device('/device:GPU:1'):
        end_points = xception_39_series(inputs, nb_filter_factor= size_factor)
        print('output shape of context path %s' % (list(K.get_variable_shape(end_points['pool4']))))

        net_4 = AttentionRefinementModule(end_points['pool3'], n_filters=32* size_factor)
        net_5 = AttentionRefinementModule(end_points['pool4'], n_filters=64* size_factor)

        global_channels = TimeDistributed(GlobalAveragePooling2D())(net_5)
        #tf.reduce_mean(net_5, [1, 2], keep_dims=True)
        # net_5_scaled = Multiply()([global_channels, net_5])
        net_5_scaled = Lambda(multiply_with_broadcast)([net_5, global_channels])
        ### Combining the paths
        net_4 = upsampling_series(net_4, scale=2)
        net_5_scaled = upsampling_series(net_5_scaled, scale=4)

        context_net = Concatenate(axis=-1)([net_4, net_5_scaled])

        net = FeatureFusionModule(input_1=spatial_net, input_2=context_net, n_filters=nb_classes)

        ### Final upscaling and finish
        net = upsampling_series(net, scale=4)

        net = ConvLSTM2D(nb_classes,
                         kernel_size = (3, 3),
                         padding='same',
                         activation=activation,
                         return_sequences=True)(net)
        net = upsampling_series(net, scale=2)
        # net = enet_upsample(net, 16, scale=4, name='unsample4')
        # net = enet_upsample(net, 8, scale=2, name='unsample2')
        # n_filters, kernel_size, strides = [strides, strides]

        # net = ConvLSTM2D(nb_classes,
        #                  kernel_size = 1,
        #                  padding='same',
        #                  activation=activation,
        #                  return_sequences=True)(net)
        net = TimeDistributed(Conv2D(nb_classes, kernel_size=1, padding='same', activation = activation))(net)

        model = models.Model(inputs, net, name='BiSeNet_ConvLSTM')
    return model



if __name__ == '__main__':
    from keras.utils.vis_utils import plot_model
    from time import time
    import os

    result_path = r'/home/dejun/Pictures'
    tic = time()

    model = bisenet_convlstm(10, size_factor=2)

    toc = time()
    print(round(toc - tic, 4))
    model.summary()
    with open(os.path.join(result_path, 'bisenet_lstm.json'), 'w') as files:
        files.write(model.to_json())
    # 存储模型图
    plot_model(model, to_file=os.path.join(result_path, 'bisenet_lstm.png'), show_shapes=True)
