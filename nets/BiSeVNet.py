# coding=utf-8
# @Time	  : 2019-04-09 17:05
# @Author   : Monolith
# @FileName : BiSeNet.py


# coding=utf-8

from nets.net_basis import *
import tensorflow as tf
from keras import models
from keras.layers import *
import keras.backend as K
from nets.ResNet3D import ResNet50V2_3D
from keras.regularizers import l2



def Upsampling(input_tensor, scale, pseudo_bilinear_3d = False):
    dims = K.int_shape(input_tensor)
    # net = tf.keras.UpSampling2D
    # net = Lambda(lambda x: bilinear_upsameple(tensor=x, size=(scale, scale)))(input_tensor)
    if len(dims) == 4:
        net = UpSampling2D(size=(scale, scale), interpolation='bilinear')(input_tensor)  # , interpolation='bilinear'
    else:
        if pseudo_bilinear_3d:
            net = TimeDistributed(UpSampling2D(size=(scale, scale), interpolation='bilinear'))(input_tensor)
            net = Permute((2, 1, 3, 4))(net)  # (B, z, H, W, C) -> (B, H, z, w, c)
            net = TimeDistributed(UpSampling2D(size=(scale, 1), interpolation='bilinear'))(net)
            net = Permute((2, 1, 3, 4))(net)  # (B, z, H, W, C) -> (B, H, z, w, c)
        else:
            net = UpSampling3D(size = scale)(input_tensor)

    return net


def conv_bt_act_3d(inputs, nb_filters, kernel_size=3, strides = 1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = Conv3D(nb_filters, kernel_size, strides=strides,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(inputs)

    net = BatchNormalization()(net)
    net = Activation('relu')(net)


    return net



def AttentionRefinementModule3D(inputs):
    # Global average pooling

    nb_channels = K.get_variable_shape(inputs)[-1]

    net = GlobalAveragePooling3D()(inputs)
    net = Dense(nb_channels, kernel_initializer='he_normal')(net)
    # net = Reshape((1, 1, 1, nb_channels))(net) # B, 1, C
    # net = Conv3D(nb_channels, kernel_size=1,kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('sigmoid')(net)
    net = Reshape((1, 1, 1, nb_channels))(net)
    # Permute((1, 2, 4, 3), name=f'permute_1_{name}')(x_copy)  # (B, T, H, W, C) -> (B, T, H, C, W)
    net = Multiply()([inputs, net])

    return net


def FeatureFusionModule3D(input_1, input_2, n_filters):
    inputs = concatenate([input_1, input_2], axis=-1)

    # bottle neck
    inputs = conv_bt_act_3d(inputs, nb_filters= int(n_filters//2), kernel_size=1)
    inputs = conv_bt_act_3d(inputs, nb_filters=int(n_filters //2), kernel_size=3)
    inputs = conv_bt_act_3d(inputs, nb_filters=n_filters, kernel_size=1)

    # Global average pooling
    net = se3d_fc(inputs)

    net = Add()([inputs, net])

    return net


def bisenet_3d(num_classes, size_factor=1):
    """
    Builds the BiSeNet model.

    Arguments:
      num_classes: Number of classes

    Returns:
      BiSeNet model
    """

    # The spatial path
    # The number of feature maps for each convolution is not specified in the paper
    # It was chosen here to be equal to the number of feature maps of a classification
    # model at each corresponding stage
    if num_classes == 1:
        activation = 'sigmoid'
        nb_classes = 1
    else:
        activation = 'softmax'
        nb_classes = num_classes + 1

    with tf.device('/device:GPU:0'):
        inputs = Input((None, None, None, 1))  #ip, nb_filter, kernel_size=3, strides = 1, dilation_rate=1
        spatial_net = conv_bt_act_3d(inputs, nb_filters=16 * size_factor, kernel_size=3, strides=2)
        print('spatial path 0 %s' % (list(K.get_variable_shape(spatial_net))))
        spatial_net = conv_bt_act_3d(spatial_net, nb_filters=32 * size_factor, kernel_size=3, strides=2)
        print('spatial path 1 %s' % (list(K.get_variable_shape(spatial_net))))
        spatial_net = conv_bt_act_3d(spatial_net, nb_filters=64 * size_factor, kernel_size=3, strides=2)
        print('spatial path 2 %s' % (list(K.get_variable_shape(spatial_net))))

    # Context path
    with tf.device('/device:GPU:1'):
        end_points = ResNet50V2_3D(inputs, nb_filter_factor=size_factor)
        print('output shape of context path %s' % (list(K.get_variable_shape(end_points['out32']))))

        context_16 = AttentionRefinementModule3D(end_points['out16'])

        context_32 = AttentionRefinementModule3D(end_points['out32'])

        global_channels = GlobalAveragePooling3D()(context_32)
        global_channels = Reshape((1, 1, 1, K.get_variable_shape(global_channels)[-1]))(global_channels)
        # tf.reduce_mean(net_5, [1, 2], keep_dims=True)
        context_32_calibrate = Multiply()([global_channels, context_32])


        ### Combining the paths
        context_16 = Upsampling(context_16, scale=2)
        context_32_calibrate = Upsampling(context_32_calibrate, scale=4)

        context_16_supervise = Conv3D(nb_classes, 1, activation= activation)(context_16)
        context_32_supervise = Conv3D(nb_classes, 1, activation= activation)(context_32_calibrate)
        context_16_supervise = UpSampling3D(8, name = 'ct_16')(context_16_supervise)
        context_32_supervise = UpSampling3D(8, name = 'ct_32')(context_32_supervise)


        context_net = concatenate([context_16, context_32_calibrate], axis=-1)

        combine_out = FeatureFusionModule3D(input_1=spatial_net, input_2=context_net, n_filters=nb_classes * 2)
    with tf.device('/device:GPU:0'):
        ### Final upscaling and finish
        combine_out = Upsampling(combine_out, scale=4)
        combine_out = conv_bt_act_3d(combine_out, nb_filters=nb_classes * 2, kernel_size=3)
        # net = separable_block(net, nb_filters= nb_classes, kernel_size=[3, 3])
        combine_out = Upsampling(combine_out, scale=2)
        combine_out = Conv3D(nb_classes, 1, activation=activation, name= 'f_out')(combine_out)
        print(f'output shape is {combine_out.shape}')

        final_out = [combine_out, context_16_supervise, context_32_supervise]
        model = models.Model(inputs = inputs, outputs= final_out, name='BiSeNet3D')
    return model


if __name__ == '__main__':
    from keras.utils.vis_utils import plot_model
    from time import time
    import os

    result_path = r'/home/dejun/Pictures'
    tic = time()

    model = bisenet_3d(10, size_factor=1)

    toc = time()
    print(round(toc - tic, 4))
    model.summary()
    with open(os.path.join(result_path, 'BiSeNet3D.json'), 'w') as files:
        files.write(model.to_json())
    # 存储模型图
    plot_model(model, to_file=os.path.join(result_path, 'BiSeNet3D.png'), show_shapes=True)
