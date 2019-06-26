'''DenseNet models for Keras.
# Reference
- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf)
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf


def up_sampling(input_tensor, scale):
    dims = K.int_shape(input_tensor)
    # net = tf.keras.UpSampling2D
    # net = Lambda(lambda x: bilinear_upsameple(tensor=x, size=(scale, scale)))(input_tensor)
    if len(dims) == 4:
        net = UpSampling2D(size=(scale, scale), interpolation='bilinear')(input_tensor)  # , interpolation='bilinear'
    else:
        net = TimeDistributed(UpSampling2D(size=(scale, scale), interpolation='bilinear'))(input_tensor)
        net = Permute((2, 1, 3, 4))(net)  # (B, z, H, W, C) -> (B, H, z, w, c)
        net = TimeDistributed(UpSampling2D(size=(scale, 1), interpolation='bilinear'))(net)
        net = Permute((2, 1, 3, 4))(net)  # (B, z, H, W, C) -> (B, H, z, w, c)

    return net


# from DenseNet.subpixel import SubPixelUpscaling

def __bottleneck(x, nb_filter, increase_factor=4., weight_decay=1e-4):
    inter_channel = int(
        nb_filter * increase_factor)  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua
    x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    return x


def __conv_block(ip, nb_filter, kernal_size=(3, 3, 3), dilation_rate=1,
                 bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer='he_normal',
                   padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = Conv3D(nb_filter, kernal_size, dilation_rate=dilation_rate,
               kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    if dropout_rate:
        x = SpatialDropout3D(dropout_rate)(x)
    return x


def __dense_block(x, nb_layers, growth_rate, kernal_size=(3, 3, 3),
                  dilation_list=None,
                  bottleneck=True, dropout_rate=None, weight_decay=1e-4,
                  return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    if dilation_list is None:
        dilation_list = [1] * nb_layers
    elif type(dilation_list) is int:
        dilation_list = [dilation_list] * nb_layers
    else:
        if len(dilation_list) != nb_layers:
            raise ('the length of dilation_list should be equal to nb_layers %d' % nb_layers)

    x_list = [x]

    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, kernal_size, dilation_list[i],
                          bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)
        if i == 0:
            x = cb
        else:
            x = concatenate([x, cb], axis=concat_axis)

    if return_concat_list:
        return x, x_list
    else:
        return x


def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4,
                       pool_kernal=(3, 3, 3), pool_strides=(2, 2, 2)):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv3D(int(nb_filter * compression), (1, 1, 1),
               kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling3D(pool_kernal, strides=pool_strides)(x)

    return x


def __transition_up_block(ip, nb_filters, compression=1.0,
                          kernal_size=(3, 3, 3), pool_strides=(2, 2, 2),
                          type='deconv', weight_decay=1E-4):
    ''' SubpixelConvolutional Upscaling (factor = 2)
    Args:
        ip: keras tensor
        nb_filters: number of layers
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines type of upsampling performed
        weight_decay: weight decay factor
    Returns: keras tensor, after applying upsampling operation.
    '''

    if type == 'upsampling':
        x = UpSampling3D(size=kernal_size, interpolation='bilinear')(ip)
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = Conv3D(int(nb_filters * compression), (1, 1, 1), kernel_initializer='he_normal', padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
    # elif type == 'subpixel':
    #     x = Conv3D(nb_filters, kernal_size, activation='relu', padding='same', kernel_regularizer=l2(weight_decay),
    #                use_bias=False, kernel_initializer='he_normal')(ip)
    #     x = SubPixelUpscaling(scale_factor=2)(x)
    #     x = Conv3D(nb_filters, kernal_size, activation='relu', padding='same', kernel_regularizer=l2(weight_decay),
    #                use_bias=False, kernel_initializer='he_normal')(x)
    else:
        x = Conv3DTranspose(int(nb_filters * compression), kernal_size, strides=pool_strides, activation='relu',
                            padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(ip)

    return x


# def up_sample_3d_bilinear(input_tensor):


def dense_v_net(nb_classes=3,
                encoder_nb_layers=(5, 8, 8),
                growth_rate=(4, 8, 12),
                dilation_list=(5, 3, 1),
                dropout_rate=0.25,
                weight_decay=1e-4,
                init_conv_filters=24):
    """ Build the DenseNet model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        encoder_nb_layers: number of layers in each dense block.
            Can be a positive integer or a list.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1)
        nb_upsampling_conv: number of convolutional layers in upsampling via subpixel convolution
        upsampling_type: Can be one of 'upsampling', 'deconv' and 'subpixel'. Defines
            type of upsampling algorithm used.
        input_shape: Only used for shape inference in fully convolutional networks.
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                    Note that if sigmoid is used, classes must be 1.
    Returns: keras tensor with nb_layers of conv_block appended
    """

    img_input = Input(shape=(None, None, None, 1))

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.get_variable_shape(img_input)
    print('input shape is', input_shape)
    if concat_axis == 1:  # channels_first dim ordering
        _, _, z, rows, cols = input_shape
    else:
        _, z, rows, cols, _ = input_shape

    # if reduction != 0.0:
    #     assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # layers in each dense block
    nb_dense_block = len(encoder_nb_layers)  # Convert tuple to list

    print('# dense layers', nb_dense_block)
    # compute compression factor
    with tf.device('/device:GPU:0'):
        # Initial convolution
        x = Conv3D(init_conv_filters, (5, 5, 5), strides=2, kernel_initializer='he_normal', padding='same',
                   name='initial_conv3D',
                   use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        print('initial convolution output shape', K.get_variable_shape(x))

        skip_list = []

        # Add dense blocks and transition down block
        for block_idx in range(nb_dense_block):
            # x, nb_layers, growth_rate, kernal_size = (3, 3, 3),
            # dilation_list = None,
            # bottleneck = True, dropout_rate = None, weight_decay = 1e-4,
            # return_concat_list = False
            x = __dense_block(x, encoder_nb_layers[block_idx],
                              growth_rate[block_idx],
                              kernal_size=(3, 3, 3),
                              dilation_list=dilation_list[block_idx],
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay,
                              )
            print('encoding dense block', str(block_idx),
                  'output shape ', K.get_variable_shape(x))
            # Skip connection
            skip_list.append(x)
            # add transition_block  ##先降维后pooling好，还是先pooling后降维好？
            # if block_idx
            x = AveragePooling3D((2, 2, 2))(x)
            # x = __transition_block(x, nb_filter,
            #                        compression=compression,
            #                        weight_decay=weight_decay,
            #                        pool_kernal=(3, 3, 3),
            #                        pool_strides=(2, 2, 2))
            print('down sample', str(block_idx),
                  'output shape ', K.get_variable_shape(x))

        # skip_list = skip_list[::-1]  # reverse the skip list
    with tf.device('/device:GPU:1'):
        x_level3 = __conv_block(skip_list[-1], 32, bottleneck=True, dropout_rate=dropout_rate)
        print('decoding level3 shape ', K.get_variable_shape(x_level3))
        x_level3 = up_sampling(x_level3, scale=4)
        # x_level3 = UpSampling3D(size = (4,4,4))(x_level3)

        x_level2 = __conv_block(skip_list[-2], 32, bottleneck=True, dropout_rate=dropout_rate)
        print('decoding level2 shape ', K.get_variable_shape(x_level2))
        x_level2 = up_sampling(x_level2, scale=2)
        # x_level2 = UpSampling3D(size=(2, 2, 2))(x_level2)

        x_level1 = __conv_block(skip_list[-3], 16, bottleneck=True, dropout_rate=dropout_rate)
        print('decoding level1 shape ', K.get_variable_shape(x_level1))
        x = Concatenate()([x_level3, x_level2, x_level1])
        print('decoding concatenation shape ', K.get_variable_shape(x))

        x = __conv_block(x, 24, bottleneck=False, dropout_rate=dropout_rate)
        print('decoding level1 shape ', K.get_variable_shape(x))

        # x = UpSampling3D(size=(2, 2, 2))(x)
        x = up_sampling(x, scale=2)
        print('decoding segmentation shape ', K.get_variable_shape(x))

        print('last upsample output shape is ', K.get_variable_shape(x))

        # x = Dropout(0.3)(x)
        if nb_classes == 1:
            x = Conv3D(nb_classes, 1, activation='sigmoid', padding='same', use_bias=False)(x)
        elif nb_classes > 1:
            x = Conv3D(nb_classes + 1, 1, activation='softmax', padding='same', use_bias=False)(x)
        print('top layer output shape ', K.get_variable_shape(x))

        # Create model.
        model = Model(img_input, x, name='dense_v_net')
    return model


if __name__ == '__main__':
    from keras.utils.vis_utils import plot_model
    from time import time
    import os

    result_path = r'/home/dejun/Pictures'
    tic = time()
    model = dense_v_net(nb_classes=10,
                        encoder_nb_layers=(5, 8, 8),
                        dilation_list=(5, 3, 1),
                        growth_rate=(4, 8, 16),
                        dropout_rate=0.2,
                        weight_decay=1e-4,
                        init_conv_filters=24)

    toc = time()
    print(round(toc - tic, 4))
    # model.summary()
    with open(os.path.join(result_path, 'densevnet.json'), 'w') as files:
        files.write(model.to_json())
    # 存储模型图
    plot_model(model, to_file=os.path.join(result_path, 'densevnet.png'), show_shapes=True)

    # from keras.callbacks import ModelCheckpoint, TensorBoard
    # plot_model(model, 'test.png', show_shapes=True)

    # # Add dense blocks and transition up block
    # for block_idx in range(nb_dense_block):
    #     n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx+1]
    #     print('n_filters_keep', n_filters_keep)
    #     # upsampling block must upsample only the feature maps (concat_list[1:]),
    #     # not the concatenation of the input with the feature maps (concat_list[0].
    #     l = concatenate(concat_list[1:], axis=concat_axis)
    #     # nb_filter_temp = K.get_variable_shape(l)[-1]
    #     #print('the feature maps concat_list[1:] shape', K.get_variable_shape(l))
    #     t = __transition_up_block(l, nb_filters=n_filters_keep,
    #                               kernal_size=(2, 2, 2), pool_strides=(2, 2, 2),
    #                               type=upsampling_type, weight_decay=weight_decay)
    #     print('up sample', str(nb_dense_block-block_idx), 'output shape ', K.get_variable_shape(t))
    #     # concatenate the skip connection with the transition block
    #     x = concatenate([t, skip_list[block_idx]], axis=concat_axis)
    #     # x = __bottleneck(x, n_filters_keep, increase_factor= 0.5)
    #     print('skip connection', str(nb_dense_block-block_idx), 'output shape ', K.get_variable_shape(x))
    #     # Dont allow the feature map size to grow in upsampling dense blocks
    #     x_up, nb_filter, concat_list = __dense_block(x, nb_layers[nb_dense_block + block_idx + 1], nb_filter=growth_rate,
    #                                                  growth_rate=growth_rate, dropout_rate=dropout_rate,
    #                                                  weight_decay=weight_decay, return_concat_list=True,
    #                                                  grow_nb_filters=False)
    #     print('decoding dense block', str(nb_dense_block-block_idx), 'output shape ', K.get_variable_shape(x_up))
