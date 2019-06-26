# coding=utf-8
# @Time	  : 2019-03-18 11:57
# @Author   : Monolith
# @FileName : Enet.py

import os
from keras.models import Model
from keras.layers import Input, concatenate, Conv2DTranspose, Permute, SpatialDropout2D, Add, \
    Conv3D, PReLU, ReLU, ZeroPadding2D, MaxPooling3D, UpSampling3D, Conv2D, MaxPooling2D, UpSampling2D, \
    Dropout, Cropping2D, advanced_activations, Activation, BatchNormalization, \
    ConvLSTM2D, TimeDistributed, Bidirectional
from keras import backend as K
import tensorflow as tf

os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu, floatX=float32, optimizer=fast_compile,nvcc.flags=-D_FORCE_INLINES'
os.environ['TENSORFLOW_FLAGS'] = 'mode=FAST_RUN, device=gpu, floatX=float32, \
                                    optimizer=fast_compile,nvcc.flags=-D_FORCE_INLINES'

def initial_block(tensor, n_filters = 13):
    conv = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same', name='initial_block_conv',
                  kernel_initializer='he_normal')(tensor)
    pool = MaxPooling2D(pool_size=(2, 2), name='initial_block_pool')(tensor)
    concat = concatenate([conv, pool], axis=-1, name='initial_block_concat')
    concat = PReLU(shared_axes=[1, 2], name=f'prelu_initial_block')(concat)
    return concat

def bottleneck_encoder(tensor, nfilters, downsampling=False, dilated= None, asymmetric=False, normal=False,
                       drate=0.1, name=''):
    y = tensor
    skip = tensor
    stride = 1
    ksize = 1
    if downsampling:
        stride = 2
        ksize = 2
        skip = MaxPooling2D(pool_size=(2, 2), name=f'max_pool_{name}')(skip)
        skip = Permute((1, 3, 2), name=f'permute_1_{name}')(skip)  # (B, H, W, C) -> (B, H, C, W)
        ch_pad = nfilters - K.int_shape(tensor)[-1]
        skip = ZeroPadding2D(padding=((0, 0), (0, ch_pad)), name=f'zeropadding_{name}')(skip)
        skip = Permute((1, 3, 2), name=f'permute_2_{name}')(skip)  # (B, H, C, W) -> (B, H, W, C)

    # 1*1 projection dimensionality reduction
    y = Conv2D(filters=nfilters // 4, kernel_size=(ksize, ksize), kernel_initializer='he_normal',
               strides=(stride, stride), padding='same', use_bias=False, name=f'1x1_conv_{name}')(y)
    y = BatchNormalization(momentum=0.1, name=f'bn_1x1_{name}')(y)
    y = PReLU(shared_axes=[1, 2], name=f'prelu_1x1_{name}')(y)

    # main conv
    if normal:
        y = Conv2D(filters=nfilters // 4, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same',
                   name=f'3x3_conv_{name}')(y)
    elif asymmetric:
        y = Conv2D(filters=nfilters // 4, kernel_size=(5, 1), kernel_initializer='he_normal', padding='same',
                   use_bias=False, name=f'5x1_conv_{name}')(y)
        y = Conv2D(filters=nfilters // 4, kernel_size=(1, 5), kernel_initializer='he_normal', padding='same',
                   name=f'1x5_conv_{name}')(y)
    elif dilated:
        y = Conv2D(filters=nfilters // 4, kernel_size=(3, 3), kernel_initializer='he_normal',
                   dilation_rate=(dilated, dilated), padding='same', name=f'dilated_conv_{name}')(y)
    y = BatchNormalization(momentum=0.1, name=f'bn_main_{name}')(y)
    y = PReLU(shared_axes=[1, 2], name=f'prelu_{name}')(y)

    # 1*1 dimensionality restored
    y = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', use_bias=False,
               name=f'final_1x1_{name}')(y)
    y = BatchNormalization(momentum=0.1, name=f'bn_final_{name}')(y)

    # regularization
    y = SpatialDropout2D(rate=drate, name=f'spatial_dropout_final_{name}')(y)

    y = Add(name=f'add_{name}')([y, skip])
    y = PReLU(shared_axes=[1, 2], name=f'prelu_out_{name}')(y)

    return y


def bottleneck_decoder(tensor, nfilters, upsampling=False, normal=False, name=''):
    y = tensor
    skip = tensor

    if upsampling:
        skip = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', strides=(1, 1),
                      padding='same', use_bias=False, name=f'1x1_conv_skip_{name}')(skip)
        skip = UpSampling2D(size=(2, 2), name=f'upsample_skip_{name}')(skip)

    # 1*1 dimensionality reduction
    y = Conv2D(filters=nfilters // 4, kernel_size=(1, 1), kernel_initializer='he_normal', strides=(1, 1),
               padding='same', use_bias=False, name=f'1x1_conv_{name}')(y)
    y = BatchNormalization(momentum=0.1, name=f'bn_1x1_{name}')(y)
    y = PReLU(shared_axes=[1, 2], name=f'prelu_1x1_{name}')(y)

    # main conv
    if upsampling:
        y = Conv2DTranspose(filters=nfilters // 4, kernel_size=(3, 3), kernel_initializer='he_normal', strides=(2, 2),
                            padding='same', name=f'3x3_deconv_{name}')(y)
    elif normal:
        Conv2D(filters=nfilters // 4, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal',
               padding='same', name=f'3x3_conv_{name}')(y)
        
        
    y = BatchNormalization(momentum=0.1, name=f'bn_main_{name}')(y)
    y = PReLU(shared_axes=[1, 2], name=f'prelu_{name}')(y)

    y = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', use_bias=False,
               name=f'final_1x1_{name}')(y)
    y = BatchNormalization(momentum=0.1, name=f'bn_final_{name}')(y)

    y = Add(name=f'add_{name}')([y, skip])
    y = ReLU(name=f'relu_out_{name}')(y)

    return y


def ENET(nclasses):
    print('. . . . .Building ENet. . . . .')
    img_input = Input(shape=(None, None, 1), name='image_input')
    with tf.device('/device:GPU:0'):
        x = initial_block(img_input)

        x = bottleneck_encoder(x, 64, downsampling=True, normal=True, name='1.0', drate=0.01)
        for _ in range(1, 5):
            x = bottleneck_encoder(x, 64, normal=True, name=f'1.{_}', drate=0.01)

        x = bottleneck_encoder(x, 128, downsampling=True, normal=True, name=f'2.0')
        x = bottleneck_encoder(x, 128, normal=True, name=f'2.1')
        x = bottleneck_encoder(x, 128, dilated=2, name=f'2.2')
        x = bottleneck_encoder(x, 128, asymmetric=True, name=f'2.3')
        x = bottleneck_encoder(x, 128, dilated=4, name=f'2.4')
        x = bottleneck_encoder(x, 128, normal=True, name=f'2.5')
        x = bottleneck_encoder(x, 128, dilated=6, name=f'2.6')
        x = bottleneck_encoder(x, 128, asymmetric=True, name=f'2.7')
        x = bottleneck_encoder(x, 128, dilated=8, name=f'2.8')

    with tf.device('/device:GPU:1'):
        x = bottleneck_encoder(x, 128, normal=True, name=f'3.0')
        x = bottleneck_encoder(x, 128, dilated=2, name=f'3.1')
        x = bottleneck_encoder(x, 128, asymmetric=True, name=f'3.2')
        x = bottleneck_encoder(x, 128, dilated=4, name=f'3.3')
        x = bottleneck_encoder(x, 128, normal=True, name=f'3.4')
        x = bottleneck_encoder(x, 128, dilated=8, name=f'3.5')
        x = bottleneck_encoder(x, 128, asymmetric=True, name=f'3.6')
        x = bottleneck_encoder(x, 128, dilated=16, name=f'3.7')

        x = bottleneck_decoder(x, 64, upsampling=True, name='4.0')
        x = bottleneck_decoder(x, 64, normal=True, name='4.1')
        x = bottleneck_decoder(x, 64, normal=True, name='4.2')

        x = bottleneck_decoder(x, 16, upsampling=True, name='5.0')
        x = bottleneck_decoder(x, 16, normal=True, name='5.1')

        img_output = Conv2DTranspose(nclasses, kernel_size=(2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                     padding='same', name='image_output')(x)
        img_output = Activation('softmax')(img_output)

    model = Model(inputs=img_input, outputs=img_output, name='enet')
    print('. . . . .Build Compeleted. . . . .')
    return model