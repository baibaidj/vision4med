# coding=utf-8
# @Time	  : 2019-04-09 18:33
# @Author   : Monolith
# @FileName : xception39.py

from nets.net_basis import *


def separable_block(inputs, kernel_size, dilation = 1):
    nb_filter = K.get_variable_shape(inputs)[-1]

    residual = inputs

    x = Activation('relu')(residual)
    x = SeparableConv2D(nb_filter, kernel_size,
                        dilation_rate=dilation,
                        padding='same',
                        use_bias=False,
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4)
                        )(x)
    x = BatchNormalization()(x)

    x = add([x, residual])

    return x


def separable_block_down(inputs, nb_filter, ix=10, filter_size=(3, 3), dilation=1):
    prefix = 'downsample_' + str(ix)
    residual = Conv2D(nb_filter, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(1e-4)
                             )(inputs)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(nb_filter, filter_size,
                               padding='same',
                               use_bias=False,
                               name=prefix + '_sepconv1',
                               dilation_rate=dilation,
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(1e-4)
                               )(inputs)
    x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
    x = Activation('relu', name=prefix + '_sepconv2_act')(x)

    x = SeparableConv2D(nb_filter, filter_size,
                               padding='same',
                               use_bias=False,
                               name=prefix + '_sepconv2',
                               dilation_rate=dilation,
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(1e-4)
                               )(x)
    x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name=prefix + '_pool'
                            )(x)
    x = add([x, residual])
    return x


def separable_res_block_deep(inputs, ix=0, filter_size=(3, 3), dilation=1):
    nb_filter = K.get_variable_shape(inputs)[-1]

    residual = inputs

    prefix = 'repeat' + str(ix)

    x = Activation('relu', name=prefix + '_sepconv1_act')(residual)
    x = SeparableConv2D(nb_filter, filter_size,
                               dilation_rate=dilation,
                               padding='same',
                               use_bias=False,
                               name=prefix + '_sepconv1',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(1e-4)
                               )(x)
    x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)

    x = Activation('relu', name=prefix + '_sepconv2_act')(x)
    x = SeparableConv2D(nb_filter, filter_size,
                               dilation_rate=dilation,
                               padding='same',
                               use_bias=False,
                               name=prefix + '_sepconv2',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(1e-4)
                               )(x)
    x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)

    x = Activation('relu', name=prefix + '_sepconv3_act')(x)
    x = SeparableConv2D(nb_filter, filter_size,
                               dilation_rate=dilation,
                               padding='same',
                               use_bias=False,
                               name=prefix + '_sepconv3')(x)
    x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

    x = add([x, residual])
    return x


def xception_39(img_input, nb_filter_factor = 1):
    end_points = {}
    x = Conv2D(int(8*nb_filter_factor), (7, 7),
                      strides=(2, 2),
                      use_bias=False,
                      padding='same',
                      name='block1_conv1',
                      )(img_input)

    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)

    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
    end_points['out4'] = x
    print('out4', x.shape)

    x = separable_block_down(x, int(16*nb_filter_factor), ix=0, filter_size=(3, 3))
    for i in range(3):
        x = separable_res_block_deep(x, ix=10 + i, filter_size=(3, 3), dilation=max(3, i + 1))
    end_points['out8'] = x
    print('out8', x.shape)

    x = separable_block_down(x, int(32*nb_filter_factor), ix=1, filter_size=(3, 3))
    for i in range(7):
        x = separable_res_block_deep(x, ix=20 + i, filter_size=(3, 3), dilation=max(3, i + 1))
    end_points['out16'] = x
    print('out16', x.shape)

    x = separable_block_down(x, int(64*nb_filter_factor), ix=2, filter_size=(3, 3))
    for i in range(3):
        x = separable_res_block_deep(x, ix=30 + i, filter_size=(3, 3), dilation=max(3, i + 1))
    end_points['out32'] = x
    print('out32', x.shape)

    return end_points



if __name__ == '__main__':
    from keras.utils.vis_utils import plot_model
    from time import time
    import os

    result_path = r'/home/dejun/Pictures'
    tic = time()
    img_input = Input((None, None, 1))

    end_points = xception_39(img_input)

    net = GlobalAveragePooling2D()(end_points['out32'])
    nb_classes = 2
    if nb_classes == 2:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    net = Dense(nb_classes, activation=activation)(net)
    model = models.Model(img_input, net, name='xception')

    toc = time()
    print(round(toc - tic, 4))
    model.summary()
    # 存储模型图
    plot_model(model, to_file=os.path.join(result_path, 'xception.png'), show_shapes=True)
