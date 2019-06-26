# -*- coding: utf-8 -*-
# @Time    : 2018/7/25 15:53
# @Author  : Monolith
# @FileName: ResNet3D.py

from nets.net_basis import *

bn_axis = -1


def res_block2_3d(x, filters, kernel_size=3, stride=1,
           conv_shortcut=False, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """

    preact = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                name=name + '_preact_bn')(x)
    preact = Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = Conv3D(2 * filters, 1, strides=stride,
                          name=name + '_0_conv')(preact)
    else:
        shortcut = MaxPooling3D(1, strides=stride)(x) if stride > 1 else x

    x = Conv3D(filters, 1, strides=1, use_bias=False,
               name=name + '_1_conv')(preact)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    x = ZeroPadding3D(padding=((1, 1), (1, 1), (1,1)), name=name + '_2_pad')(x)
    x = Conv3D(filters, kernel_size, strides=stride,
               use_bias=False, name=name + '_2_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)

    x = Conv3D(2 * filters, 1, name=name + '_3_conv')(x)
    x = Add(name=name + '_out')([shortcut, x])
    return x


def res_stack2_3d(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = res_block2_3d(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = res_block2_3d(x, filters, name=name + '_block' + str(i))

    x_pre_downsample = x
    x = res_block2_3d(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x_pre_downsample, x


def ResNet50V2_3D(img_input, nb_filter_factor=1, preact=True, use_bias=False):
    print('Initializing ResNet50V2_3D')
    end_points = {}
    x = ZeroPadding3D(padding=((3, 3), (3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = Conv3D(int(16 * nb_filter_factor), 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if preact is False:
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                               name='conv1_bn')(x)
        x = Activation('relu', name='conv1_relu')(x)

    x = ZeroPadding3D(padding=((1, 1), (1, 1), (1, 1)), name='pool1_pad')(x)
    x = MaxPooling3D(3, strides=2, name='pool1_pool')(x)

    out4, x = res_stack2_3d(x, int(8 * nb_filter_factor), 3, name='conv2')
    end_points['out4'] = out4
    print('out4', out4.shape)

    out8, x = res_stack2_3d(x, int(16 * nb_filter_factor), 4, name='conv3')

    end_points['out8'] = out8
    print('out8', out8.shape)

    out16, x = res_stack2_3d(x, int(32 * nb_filter_factor), 6, name='conv4')
    end_points['out16'] = out16
    print('out16', out16.shape)

    _, x = res_stack2_3d(x, int(64 * nb_filter_factor), 3, stride1=1, name='conv5')
    if preact is True:
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                               name='post_bn')(x)
        x = Activation('relu', name='post_relu')(x)

    end_points['out32'] = x
    print('out32', x.shape)

    return end_points




if __name__=='__main__':
    from keras.utils.vis_utils import plot_model
    from time import time
    import os
    result_path = r'/home/dejun/Pictures'
    tic = time()

    img_input = Input((None, None, None, 1))

    end_points = ResNet50V2_3D(img_input)

    net = GlobalAveragePooling3D()(end_points['pool4'])
    nb_classes = 2
    if nb_classes == 2:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    net = Dense(nb_classes, activation=activation)(net)
    model = models.Model(img_input, net, name='resnetv2_3d')

    toc = time()
    print(round(toc - tic, 4))
    model.summary()
    # 存储模型图
    plot_model(model, to_file=os.path.join(result_path, 'resnetv2_3d.png'), show_shapes=True)