# coding=utf-8
# @Time	  : 2019-06-25 17:01
# @Author   : Monolith
# @FileName : PSPNet.py

from __future__ import print_function
from nets.ResNet import *
import tensorflow as tf
# Weight decay not implemented


def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)



def resnet_8(inp, nb_filter_factor = 1, depth = 50):
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    # Short branch(only start of network)

    cnv1 = Conv2D(16* nb_filter_factor, (3, 3), strides=(2, 2), padding='same', name=names[0],
                  use_bias=False)(inp)  # "conv1_1_3x3_s2"
    bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_1_3x3_s2/relu"

    cnv1 = Conv2D(16* nb_filter_factor, (3, 3), strides=(1, 1), padding='same', name=names[2],
                  use_bias=False)(relu1)  # "conv1_2_3x3"
    bn1 = BN(name=names[3])(cnv1)  # "conv1_2_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_2_3x3/relu"

    cnv1 = Conv2D(32* nb_filter_factor, (3, 3), strides=(1, 1), padding='same', name=names[4],
                  use_bias=False)(relu1)  # "conv1_3_3x3"
    bn1 = BN(name=names[5])(cnv1)  # "conv1_3_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_3_3x3/relu"

    res = MaxPooling2D(pool_size=(3, 3), padding='same',
                       strides=(2, 2))(relu1)  # "pool1_3x3_s2"

    # ---Residual layers(body of network)

    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    # 2_1- 2_3
    out4, x = res_stack2(res, int(64 * nb_filter_factor), 4, stride1=2, name='conv2_down2')

    # res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
    # for i in range(2):
    #     res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)

    # 3_1 - 3_3
    out8, x = res_stack2(x, int(128 * nb_filter_factor), 5, stride1=1, name='conv3')
    # res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
    # for i in range(3):
    #     res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)


    if depth is 50:
        # 4_1 - 4_6
        _, x = res_stack2(x, int(256 * nb_filter_factor), 5, stride1=1, name='conv4')
        # res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        # for i in range(5):
        #     res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    elif depth is 101:
        # 4_1 - 4_23
        _, x = res_stack2(x, int(256 * nb_filter_factor), 22, stride1=1, name='conv4')
        # res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        # for i in range(22):
        #     res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")

    # 5_1 - 5_3
    _, x = res_stack2(x, int(512 * nb_filter_factor), 3, stride1=1, name='conv5')
    # res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
    # for i in range(2):
    #     res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)

    res = Activation('relu')(res)
    return res


def pool_block(input_tensor, pool_factor, target_nb_filter = 32):

    h = K.int_shape(input_tensor)[1]
    w = K.int_shape(input_tensor)[2]
    pool_size = strides = [int(np.round(float(h) / pool_factor)), int(np.round(float(w) / pool_factor))]

    x = AveragePooling2D(pool_size, strides=strides, padding='same')(input_tensor)
    x = Conv2D(target_nb_filter, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D( size = strides, interpolation='bilinear')(x)

    return x


def _pspnet(n_classes, nb_filter_factor = 1, depth = 50, nb_filter_pool=32):

    inp = Input((None, None, 1))
    res = resnet_8(nb_filter_factor, depth)
    net = res

    pool_factors = [1, 2, 3, 6]
    pool_outs = [net]


    # Build the Pyramid Pooling Module
    for p in pool_factors:
        pooled = pool_block(net, p, target_nb_filter=nb_filter_pool)
        pool_outs.append(pooled)

    net = Concatenate(axis=-1)(pool_outs)


    net = Conv2D(256, (1, 1), use_bias=False)(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)

    net = Conv2D(n_classes, (3, 3),  padding='same')(net)
    net = UpSampling2D(size=(8,8), interpolation='bilinear')(net)

    model = models.Model(inp, net)
    return model


