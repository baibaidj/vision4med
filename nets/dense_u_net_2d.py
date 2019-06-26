# coding=utf-8
# @Time	  : 2019-03-27 18:04
# @Author   : Monolith
# @FileName : dense_u_net_2d.py

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# from keras.utils.vis_utils import plot_model
import time
tic = time.time()
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
# from keras.utils.layer_utils import convert_all_kernels_in_model, convert_dense_weights_data_format
# from keras.utils.data_utils import get_file
# from keras.engine.topology import get_source_inputs
# from keras_applications.imagenet_utils import _obtain_input_shape
# from keras_applications.imagenet_utils import decode_predictions
import keras.backend as K
import tensorflow as tf
toc = time.time()
print('import takes %.3f' %(toc-tic))

def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
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
        inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  grow_nb_filters=True, return_concat_list=False):
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

    x_list = [x]

    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)

        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):
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
    x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x_copy = x
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    x_copy = Conv2D(int(nb_filter * compression), (3, 3),
                    strides=2, kernel_initializer='he_normal',
                    padding='same', use_bias=False,
                    kernel_regularizer=l2(weight_decay))(x_copy)
    x = concatenate([x, x_copy], axis=-1)

    return x


def __transition_up_block(ip, nb_filters, compression = 0.5, weight_decay=1E-4):
    ''' SubpixelConvolutional Upscaling (factor = 2)
    Args:
        ip: keras tensor
        nb_filters: number of layers
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines type of upsampling performed
        weight_decay: weight decay factor
    Returns: keras tensor, after applying upsampling operation.
    '''

    # if type == 'upsampling':
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)

    x = Conv2D(int(nb_filters * compression), (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x_copy = x
    x_up = UpSampling2D()(x)
    # elif type == 'subpixel':
    #     x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay),
    #                use_bias=False, kernel_initializer='he_normal')(ip)
    #     x = SubPixelUpscaling(scale_factor=2)(x)
    #     x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay),
    #                use_bias=False, kernel_initializer='he_normal')(x)
    # else:
    x_copy = Conv2DTranspose(int(nb_filters * compression), kernel_size = (5, 5),
                             strides=2, activation='relu', padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x_copy)
    x = concatenate([x_up, x_copy], axis=-1)
    return x


def dense_u_net(nb_classes=1,
                nb_dense_block=3,
                growth_rate=12,
                nb_layers_per_block=4,
                reduction=0.0,
                dropout_rate=None,
                weight_decay=1e-4,
                init_conv_filters=48):
    ''' Build the DenseNet model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        nb_layers_per_block: number of layers in each dense block.
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
    '''

    img_input = Input(shape=(None, None, 1), name='image_input')

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block + 1), 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be (nb_dense_block + 1)'

        bottleneck_nb_layers = nb_layers[-1]
        rev_layers = nb_layers[::-1]
        nb_layers.extend(rev_layers[1:])
    else:
        bottleneck_nb_layers = nb_layers_per_block
        nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

    print('number of layers in all the blocks %s' %str(nb_layers))
    # compute compression factor
    compression = 1.0 - reduction

    with tf.device('/device:GPU:1'):
        # Initial convolution
        x = Conv2D(init_conv_filters, (7, 7),
                   strides=2,
                   kernel_initializer='he_normal',
                   padding='same',
                   name='initial_conv2D',
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay))(img_input)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

        print('encode initial output shape ', K.get_variable_shape(x))
        nb_filter = init_conv_filters

        # skip_list = []
        # Add dense blocks and transition down block
        for block_idx in range(nb_dense_block):
            x, nb_filter = __dense_block(x, nb_layers[block_idx],
                                         nb_filter,
                                         growth_rate,
                                         dropout_rate=dropout_rate,
                                         weight_decay=weight_decay)
            print('encode', str(block_idx+1), 'output shape ', K.get_variable_shape(x))
            # Skip connection
            # skip_list.append(x)

            # add transition_block
            x = __transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)
            print('pool  ', str(block_idx+1), 'output shape ', K.get_variable_shape(x))
            nb_filter = int(nb_filter * compression)  # this is calculated inside transition_down_block

        # The last dense_block does not have a transition_down_block
        # return the concatenated feature maps without the concatenation of the input
        print('bottom num feature maps %d ' %bottleneck_nb_layers)
        _, nb_filter, x_list = __dense_block(x, bottleneck_nb_layers, nb_filter, growth_rate,  # , concat_list
                                        dropout_rate=dropout_rate, weight_decay=weight_decay,
                                        return_concat_list=True
                                        )
        x_up = concatenate(x_list[1:], axis=-1)
        print('encoding last output shape ', K.get_variable_shape(x_up))
        # skip_list = skip_list[::-1]  # reverse the skip list

    with tf.device('/device:GPU:0'):
        # Add dense blocks and transition up block
        for block_idx in range(nb_dense_block):
            n_filters_keep = nb_layers[nb_dense_block + block_idx +1] * growth_rate
            print(n_filters_keep)
            # upsampling block must upsample only the feature maps (concat_list[1:]),
            # not the concatenation of the input with the feature maps (concat_list[0].
            # l = concatenate(concat_list[1:], axis=concat_axis)

            x = __transition_up_block(x_up, nb_filters= int(n_filters_keep),
                                      compression=reduction, weight_decay=weight_decay)
            print('unpool', str(nb_dense_block-block_idx), 'output shape ', K.get_variable_shape(x))
            # concatenate the skip connection with the transition block
            # x = concatenate([t, skip_list[block_idx]], axis=concat_axis)

            # Dont allow the feature map size to grow in upsampling dense blocks
            _, nb_filter, x_list = __dense_block(x, nb_layers[nb_dense_block + block_idx + 1]//2,  # , concat_list
                                            nb_filter=nb_filter,
                                            growth_rate=growth_rate*2,
                                            dropout_rate=dropout_rate,
                                            weight_decay=weight_decay,
                                            return_concat_list=True,
                                            grow_nb_filters=False)
            x_up = concatenate(x_list[1:], axis=-1)
            print('decode', str(nb_dense_block - block_idx), 'output shape ', K.get_variable_shape(x_up))
        x = __transition_up_block(x_up, nb_filters=nb_classes * 4, weight_decay=weight_decay)
        print('encoding last output shape ', K.get_variable_shape(x))
        # x_up = Conv2DTranspose(x_up, kernel_size=(2, 2), strides=(2, 2), kernel_initializer='he_normal',
        #                              padding='same', name='image_output')(x)
        # x = __conv_block(x_up, nb_classes*4, dropout_rate = dropout_rate)

        if nb_classes == 1:
            x = Conv2D(nb_classes, (1, 1), activation='sigmoid', padding='same', use_bias=False)(x)
        elif nb_classes > 1:
            x = Conv2D(nb_classes + 1, (1, 1), activation='softmax', padding='same', use_bias=False)(x)
        else:
            raise ValueError('nb_classes must be 1 or larger')
        print('output shape ', K.get_variable_shape(x))
    model = Model(img_input, x, name='fcn-densenet')

    return model


if __name__ == '__main__':


    model_path = r'/media/dejun/holder/lith/body_class/models'

    tic = time.time()
    model = dense_u_net(nb_classes=3,
                        nb_dense_block=3,
                        growth_rate=8,
                        nb_layers_per_block=[4, 6, 8, 10],
                        reduction=0.5,
                        dropout_rate=0.3,
                        weight_decay=1e-4,
                        init_conv_filters=32)

    # model = DenseNet((32, 32, 3), depth=100, nb_dense_block=3,
    #                  growth_rate=12, bottleneck=True, reduction=0.5, weights=None)
    # model = DenseNetClassify(classes=11, input_shape=input_shape, nb_dense_block = 4,
    #                  growth_rate = 12, nb_filter = 24, nb_layers_per_block = [4, 8, 16, 32], bottleneck=False,
    #                  reduction=0.5, dropout_rate=0.3, weight_decay=1e-4,
    #                  subsample_initial_block=True,
    #                  activation='softmax')
    # model.summary()
    toc = time.time()
    print('initialize model takes %.3f' % (toc - tic))
    # import os
    # from keras.callbacks import ModelCheckpoint, TensorBoard
    # plot_model(model, os.path.join(model_path, 'test_nobn.png'), show_shapes=True)
