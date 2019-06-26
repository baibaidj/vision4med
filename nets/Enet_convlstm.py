# coding=utf-8
# @Time	  : 2019-03-18 14:04
# @Author   : Monolith
# @FileName : Enet_convlstm.py


import os
from keras.models import Model
from keras.layers import Input, Concatenate, Conv2DTranspose, Permute, SpatialDropout2D, Add, \
    Conv3D, PReLU, ReLU, ZeroPadding3D, MaxPooling3D, UpSampling3D, Conv2D, MaxPooling2D, UpSampling2D, \
    Dropout, Cropping2D, advanced_activations, Activation, BatchNormalization, \
    ConvLSTM2D, TimeDistributed, Bidirectional
from keras import backend as K
import tensorflow as tf

os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu, floatX=float32, optimizer=fast_compile,nvcc.flags=-D_FORCE_INLINES'
os.environ['TENSORFLOW_FLAGS'] = 'mode=FAST_RUN, device=gpu, floatX=float32, \
                                    optimizer=fast_compile,nvcc.flags=-D_FORCE_INLINES'



def initial_block_lstm(tensor, n_filters = 13):
    """

    :param tensor: 5 dim
    :param n_filters:
    :return:
    """
    conv = TimeDistributed(Conv2D(filters=n_filters,
                      kernel_size=(3, 3),
                      strides=(2, 2),
                      padding='same',
                      name='initial_block_conv',
                      kernel_initializer='he_normal'))(tensor)
    pool = TimeDistributed(MaxPooling2D(pool_size=(2, 2),
                                        name='initial_block_pool'))(tensor)
    concat = Concatenate()([conv, pool])
    # print(concat.shape)
    concat = PReLU(shared_axes=[1, 2, 3], name=f'prelu_initial_block')(concat)
    return concat


def bottleneck_encoder_lstm(tensor, nfilters, downsampling=False,
                            normal=False, dilated=None, asymmetric=False,
                            dropout_rate=0.1, name=''):
    print('incoming %s shape %s' %(name, str(K.get_variable_shape(tensor))))
    x = tensor
    x_copy = tensor
    stride = 1
    ksize = 1

    # main branch
    if downsampling:
        stride = 2
        ksize = 2
        # max pool
        x_copy = TimeDistributed(MaxPooling2D(pool_size=(2, 2), name=f'max_pool_{name}'))(x_copy)
        # padding
        b, t, h, w, c = K.get_variable_shape(tensor)
        # var_chpad = K.zeros(shape=(b, t, h, w, nfilters-c))
        # x_copy = Concatenate()([x_copy, var_chpad])
        x_copy = Permute((1, 2, 4, 3), name=f'permute_1_{name}')(x_copy)  # (B, T, H, W, C) -> (B, T, H, C, W)
        x_copy = ZeroPadding3D(padding=((0, 0), (0, 0), (0, nfilters-c)), name=f'zeropadding_{name}')(x_copy)
        x_copy = Permute((1, 2, 4, 3), name=f'permute_2_{name}')(x_copy)  # (B, H, C, W) -> (B, H, W, C)

    # side branch
    # 1*1 projection dimensionality reduction
    x = TimeDistributed(Conv2D(filters=nfilters // 4,
                               kernel_size=(ksize, ksize),
                               kernel_initializer='he_normal',
                               strides=(stride, stride),
                               padding='same',
                               use_bias=False,
                               name=f'1x1_conv_{name}'))(x)
    x = TimeDistributed(BatchNormalization(momentum=0.1, name=f'bn_1x1_{name}'))(x)
    x = PReLU(shared_axes=[1, 2, 3], name=f'prelu_1x1_{name}')(x)

    # main conv
    if normal:
        x = TimeDistributed(Conv2D(filters=nfilters // 4, kernel_size=(3, 3),
                                   kernel_initializer='he_normal', padding='same',
                                   name=f'3x3_conv_{name}'))(x)
    elif asymmetric:
        x = TimeDistributed(Conv2D(filters=nfilters // 4, kernel_size=(5, 1),
                                   kernel_initializer='he_normal', padding='same',
                                   use_bias=False, name=f'5x1_conv_{name}'))(x)
        x = TimeDistributed(Conv2D(filters=nfilters // 4, kernel_size=(1, 5),
                                   kernel_initializer='he_normal', padding='same',
                                   name=f'1x5_conv_{name}'))(x)
    elif dilated:
        x = TimeDistributed(Conv2D(filters=nfilters // 4, kernel_size=(3, 3),
                                   kernel_initializer='he_normal',
                                   dilation_rate=(dilated, dilated), padding='same',
                                   name=f'dilated_conv_{name}'))(x)
    x = TimeDistributed(BatchNormalization(momentum=0.1, name=f'bn_main_{name}'))(x)
    x = PReLU(shared_axes=[1, 2, 3], name=f'prelu_{name}')(x)

    # 1*1 dimensionality restored
    x = TimeDistributed(Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', use_bias=False,
               name=f'final_1x1_{name}'))(x)
    x = TimeDistributed(BatchNormalization(momentum=0.1, name=f'bn_final_{name}'))(x)

    # regularization
    x = TimeDistributed(SpatialDropout2D(rate=dropout_rate, name=f'spatial_dropout_final_{name}'))(x)

    # summing the main and side branches
    x = Add(name=f'add_{name}')([x, x_copy])
    x = PReLU(shared_axes=[1, 2, 3], name=f'prelu_out_{name}')(x)

    print('outcome %s shape %s' %(name, str(K.get_variable_shape(x))))
    return x


def bottleneck_decoder_lstm(tensor, nfilters, up_flag = False,
                        projection_ratio = 4, dropout_rate = 0.1,
                        name='', if_relu = False):

    print('incoming %s shape %s' %(name, str(K.get_variable_shape(tensor))))

    x = tensor
    x_copy = tensor

    if K.get_variable_shape(x_copy)[-1] != nfilters:
        x_copy = TimeDistributed(Conv2D(filters=nfilters,
                            kernel_size=(1, 1),
                            kernel_initializer='he_normal',
                            strides=(1, 1),
                            padding='same',
                            use_bias=False,
                            name=f'1x1_conv_skip_{name}'))(x_copy)
    # main branch
    if up_flag:
        x_copy = TimeDistributed(UpSampling2D(size=(2, 2),interpolation='bilinear',
                                            name=f'upsample_skip_{name}'))(x_copy)

    # side branch
    # 1*1 dimensionality reduction
    x = TimeDistributed(Conv2D(filters=nfilters // projection_ratio,
                   kernel_size=(1, 1),
                   kernel_initializer='he_normal',
                   strides=(1, 1),
                   padding='same',
                   use_bias=False,
                   name=f'1x1_conv_{name}'))(x)
    x = TimeDistributed(BatchNormalization(momentum=0.1, name=f'bn_1x1_{name}'))(x)
    x = PReLU(shared_axes=[1, 2, 3], name=f'prelu_1x1_{name}')(x)

    # upsampling using deconv
    if up_flag:
        x = TimeDistributed(Conv2DTranspose(filters=nfilters // projection_ratio,
                                            kernel_size=(3, 3),
                                            strides=(2, 2),
                                            kernel_initializer='he_normal',
                                            padding='same',
                                            name=f'3x3_deconv_{name}'))(x)
    else:
        x = ConvLSTM2D(filters=nfilters // projection_ratio,
                            kernel_size=(3, 3),
                            kernel_initializer='he_normal',
                            strides=(1, 1),
                            padding='same',
                            use_bias=False,
                            name=f'3x3_convlstm_{name}',
                            return_sequences=True)(x)

    x = TimeDistributed(BatchNormalization(momentum=0.1, name=f'bn_main_{name}'))(x)
    x = PReLU(shared_axes=[1, 2, 3], name=f'prelu_{name}')(x)

    # 1*1 ConvLSTM to restore the feature depth to target number
    x = ConvLSTM2D(filters=nfilters,
                   kernel_size=(1, 1),
                   kernel_initializer='he_normal',
                   use_bias=False,
                   name=f'final_3x3_{name}',
                   return_sequences=True)(x)
    x = TimeDistributed(BatchNormalization(momentum=0.1, name=f'bn_final_{name}'))(x)

    # regularization
    x = TimeDistributed(SpatialDropout2D(rate=dropout_rate, name=f'spatial_dropout_final_{name}'))(x)

    # summing the main and side branches
    x = Add(name=f'add_{name}')([x, x_copy])
    if if_relu:
        x = ReLU(name=f'relu_out_{name}')(x)
    else:
        x = PReLU(shared_axes=[1, 2, 3], name=f'relu_out_{name}')(x)

    print('outcome %s shape %s' %(name, str(K.get_variable_shape(x))))
    return x




def enet_convlstm(nb_classes = 5):
    """
        
    1. 先用conv降低分辨率至1/2
    2. 然后进行下采样和特征提取
    3. 升采样总体很小
    4. 在下采样和升采样过程中，采样和卷积并行，最后相加
    
    :param nb_classes:
    :return: 
    """
    print('. . . . .Building ENet. . . . .')

    if nb_classes == 2:
        activation = 'sigmoid'
        nb_classes = 1
    else:
        activation = 'softmax'
        nb_classes = nb_classes + 1

    with tf.device('/device:GPU:0'):
        img_input = Input(shape=(None, None, None, 1), name='image_input')
        x = initial_block_lstm(img_input, n_filters=24)
        # encode1 = x # 1/2 shape

        x = bottleneck_encoder_lstm(x, 64, downsampling=True, normal=True, name='1.0', dropout_rate=0.01)
        for _ in range(1, 4):
            x = bottleneck_encoder_lstm(x, 64, normal=True, name=f'1.{_}', dropout_rate=0.01)

        # encode2 = x # 1/4 shape
        x = bottleneck_encoder_lstm(x, 128, downsampling=True, normal=True, name=f'2.0')
        x = bottleneck_encoder_lstm(x, 128, normal=True, name=f'2.1')
        x = bottleneck_encoder_lstm(x, 128, dilated=2, name=f'2.2')
        x = bottleneck_encoder_lstm(x, 128, asymmetric=True, name=f'2.3')
        x = bottleneck_encoder_lstm(x, 128, dilated=4, name=f'2.4')
        # x = bottleneck_encoder_lstm(x, 128, normal=True, name=f'2.5')
        x = bottleneck_encoder_lstm(x, 128, dilated=6, name=f'2.6')
        # x = bottleneck_encoder_lstm(x, 128, asymmetric=True, name=f'2.7')
        # x = bottleneck_encoder_lstm(x, 128, dilated=8, name=f'2.8')
        # encode3 = x # 1/8 shape
    with tf.device('/device:GPU:1'):
        x = bottleneck_decoder_lstm(x, 64, up_flag= True, name='4.0')
        # x = concatenate([x,encode2], axis=-1)
        x = bottleneck_decoder_lstm(x, 64, name='4.1')
        x = bottleneck_decoder_lstm(x, 64, name='4.2')

        x = bottleneck_decoder_lstm(x, 16, up_flag=True, name='5.0')
        # x = concatenate([x, encode1])
        x = bottleneck_decoder_lstm(x, 16, name='5.1')

        # x = bottleneck_decoder_lstm(x, 16, up_flag=True, name='6.0')
        # img_output = TimeDistributed(Conv2D(nb_classes, 1,
        #                                     kernel_initializer='he_normal',
        #                                     padding='same', name='image_output',
        #                                     activation = activation))(x)
        img_output = TimeDistributed(Conv2DTranspose(nb_classes, kernel_size=(2, 2),
                                                     strides=(2, 2), kernel_initializer='he_normal',
                                                     padding='same', name='image_output',
                                                     activation = activation))(x)

        # img_output = Activation('softmax')(img_output)

    model = Model(inputs=img_input, outputs=img_output, name='ENET_lstm')
    print('. . . . .Build Compeleted. . . . .')
    return model

if __name__=='__main__':
    from keras.utils.vis_utils import plot_model
    from time import time
    import os
    result_path = r'/home/dejun/Pictures'
    tic = time()

    test_net = enet_convlstm(nb_classes=3)
    toc = time()
    print(round(toc-tic, 4))
    test_net.summary()
    # 存储模型图
    plot_model(test_net, to_file=os.path.join(result_path, 'enet_lstm.png'), show_shapes=True)