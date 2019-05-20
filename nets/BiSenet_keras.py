# coding=utf-8

import tensorflow as tf
from nets.xception39 import xception_39, separable_block
from keras import layers, models
import keras.backend as K
from keras.regularizers import l2
import numpy as np
import os, sys


def bilinear_upsameple(tensor, size):
    y = tf.image.resize_bilinear(images=tensor, size=size)
    return y


def up_sampling(input_tensor, scale):
    # dims = K.int_shape(input_tensor)
    # net = tf.keras.layers.UpSampling2D
    # net = layers.Lambda(lambda x: bilinear_upsameple(tensor=x, size=(scale, scale)))(input_tensor)
    net = layers.UpSampling2D(size=(scale, scale), interpolation='bilinear')(input_tensor) #, interpolation='bilinear'
    return net


def ConvBlock(inputs, n_filters, kernel_size=[3, 3], strides=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = layers.Conv2D(n_filters, kernel_size, strides=[strides, strides],
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(inputs)

    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)
    return net


def AttentionRefinementModule(inputs):
    # Global average pooling
    nb_channels = K.get_variable_shape(inputs)[-1]
    net = layers.GlobalAveragePooling2D()(inputs)
    # net = K.mean(inputs, axis = [1,2], keepdims=False)

    net = layers.Reshape((1, nb_channels))(net)
    # print(net.shape)
    net = layers.Conv1D(nb_channels, kernel_size=1,
                        kernel_initializer='he_normal',
                        )(net)
    net = layers.BatchNormalization()(net)
    # net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    # net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    # net = slim.batch_norm(net, fused=True)
    net = layers.Activation('sigmoid')(net)#tf.sigmoid(net)

    net = layers.Multiply()([inputs, net])

    return net


def FeatureFusionModule(input_1, input_2, n_filters):
    inputs = layers.concatenate([input_1, input_2], axis=-1)
    inputs = ConvBlock(inputs, n_filters=n_filters, kernel_size=[3, 3])

    # Global average pooling
    # net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    # net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    # net = tf.nn.relu(net)
    # net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = layers.GlobalAveragePooling2D()(inputs)
    net = layers.Reshape((1, K.get_variable_shape(net)[-1]))(net)
    net = layers.Conv1D(n_filters, kernel_size=1,
                        kernel_initializer='he_normal'
                        )(net)
    net = layers.Activation('relu')(net)
    net = layers.Conv1D(n_filters, kernel_size=1,
                        kernel_initializer='he_normal',
                        )(net)
    net = layers.Activation('sigmoid')(net)
    net = layers.Multiply()([inputs, net])

    net = layers.Add()([inputs, net])

    return net


def enet_upsample(tensor, nfilters, scale = 8, name=''):
    y = tensor
    skip = tensor


    skip = layers.Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', strides=(1, 1),
                  padding='same', use_bias=False, name=f'1x1_conv_skip_{name}')(skip)
    skip = layers.UpSampling2D(size=(scale, scale), interpolation='bilinear', name=f'upsample_skip_{name}')(skip)

    # 1*1 dimensionality reduction
    y = layers.Conv2D(filters=nfilters // 4, kernel_size=(1, 1), kernel_initializer='he_normal', strides=(1, 1),
               padding='same', use_bias=False, name=f'1x1_conv_{name}')(y)
    y = layers.BatchNormalization(momentum=0.1, name=f'bn_1x1_{name}')(y)
    y = layers.PReLU(shared_axes=[1, 2], name=f'prelu_1x1_{name}')(y)

    # main conv

    y = layers.Conv2DTranspose(filters=nfilters // 4, kernel_size=(3, 3),
                               kernel_initializer='he_normal', strides=(scale, scale),
                               padding='same', name=f'3x3_deconv_{name}')(y)


    y = layers.BatchNormalization(momentum=0.1, name=f'bn_main_{name}')(y)
    y = layers.PReLU(shared_axes=[1, 2], name=f'prelu_{name}')(y)

    y = layers.Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', use_bias=False,
               name=f'final_1x1_{name}')(y)
    y = layers.BatchNormalization(momentum=0.1, name=f'bn_final_{name}')(y)
    print(f'add_{name}')
    y = layers.Add(name=f'add_{name}')([y, skip])
    y = layers.ReLU(name=f'relu_out_{name}')(y)

    return y

def BiSenet(num_classes, size_factor=2):
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


    with tf.device('/device:GPU:0'):
        # The spatial path
        # The number of feature maps for each convolution is not specified in the paper
        # It was chosen here to be equal to the number of feature maps of a classification
        # model at each corresponding stage
        inputs = layers.Input((None, None, 1))
        spatial_net = ConvBlock(inputs, n_filters= int(16 * size_factor), kernel_size=[3, 3], strides=2)
        print('spatial path 0 %s' % (list(K.get_variable_shape(spatial_net))))
        spatial_net = ConvBlock(spatial_net, n_filters=int(32 * size_factor), kernel_size=[3, 3], strides=2)
        print('spatial path 1 %s' % (list(K.get_variable_shape(spatial_net))))
        spatial_net = ConvBlock(spatial_net, n_filters=int(64 * size_factor), kernel_size=[3, 3], strides=2)
        print('spatial path 2 %s' %(list(K.get_variable_shape(spatial_net))))


    with tf.device('/device:GPU:1'):
        ### Context path

        end_points = xception_39(inputs, nb_filter_factor=size_factor)
        print('output shape of context path %s' % (list(K.get_variable_shape(end_points['pool4']))))

        net_4 = AttentionRefinementModule(end_points['pool3'])

        net_5 = AttentionRefinementModule(end_points['pool4'])

        global_channels = layers.GlobalAveragePooling2D()(net_5)
        #tf.reduce_mean(net_5, [1, 2], keep_dims=True)
        net_5_scaled = layers.Multiply()([global_channels, net_5])

        ### Combining the paths
        net_4 = up_sampling(net_4, scale=2)
        net_5_scaled = up_sampling(net_5_scaled, scale=4)

        context_net = layers.concatenate([net_4, net_5_scaled], axis=-1)

        net = FeatureFusionModule(input_1=spatial_net, input_2=context_net, n_filters= int(nb_classes*size_factor))

        ### Final upscaling and finish
        net = up_sampling(net, scale=4)
        # net = ConvBlock(net, n_filters=nb_classes, kernel_size=[3, 3])
        net = separable_block(net, kernel_size=(3,3))

        net = up_sampling(net, scale=2)
        net = layers.Conv2D(nb_classes, (1, 1), activation=activation)(net)

        model = models.Model(inputs, net, name='BiSeNet')
    return model

if __name__ == '__main__':
    from keras.utils.vis_utils import plot_model
    from time import time
    import os

    result_path = r'/home/dejun/Pictures'
    tic = time()

    model = BiSenet(4)

    toc = time()
    print(round(toc - tic, 4))
    model.summary()
    with open(os.path.join(result_path, 'bisenet.json'), 'w') as files:
        files.write(model.to_json())
    # 存储模型图
    plot_model(model, to_file=os.path.join(result_path, 'bisenet.png'), show_shapes=True)
