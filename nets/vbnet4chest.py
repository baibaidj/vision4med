# coding=utf-8
# @Time	  : 2019-05-13 14:34
# @Author   : Monolith
# @FileName : vbnet4chest.py


from nets.net_basis import *


def se3d_fc(inputs):
    # Global average pooling
    nb_channels = K.get_variable_shape(inputs)[-1]

    net = GlobalAveragePooling3D()(inputs)
    net = Dense(nb_channels, kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Dense(nb_channels, kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('sigmoid')(net)
    net = Reshape((1, 1, 1, nb_channels))(net)
    net = Multiply()([inputs, net])

    return net


def bottle_neck_residual_3d(inputs, nb_filters,
                            kernel_size = 3,
                            reduction = 0.5,
                            weight_decay = 1e-4,
                            dropout_rate = None,
                            dilation_rate = 1):
    nb_channels = K.get_variable_shape(inputs)[-1]

    if nb_channels != nb_filters:
        inputs = bt_act_conv_3d(inputs, nb_filters, kernel_size=3)

    residual = bt_act_conv_3d(inputs,
                              nb_filters * reduction,
                              kernel_size=1,
                              weight_decay = weight_decay)

    residual = bt_act_conv_3d(residual,
                              nb_filters * reduction,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              dilation_rate=dilation_rate
                              )
    residual = bt_act_conv_3d(residual,
                              nb_filters,
                              kernel_size=1,
                              weight_decay=weight_decay,
                              dropout_rate=dropout_rate
                              )
    # residual = se3d_fc(residual)
    out = Add()([inputs, residual])
    return out

def down_block(inputs, nb_filters, kernel_size,
               dropout_rate = 0.25, nb_resnet = 1, enc_ix = 0):

    # x = BatchNormalization(axis=-1, epsilon=1.1e-5)(inputs)
    # x = Activation('relu')(x)
    x = AveragePooling3D(pool_size=2, padding='same')(inputs)

    for i in range(nb_resnet):
        x = bottle_neck_residual_3d(x, nb_filters,
                                    kernel_size=kernel_size,
                                    dropout_rate = dropout_rate)

    print(f'encoder {enc_ix} input {inputs.shape}  output {x.shape}')
    return x


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
        shortcut = Conv3D(filters, 1, strides=stride,
                          name=name + '_0_conv')(preact)
    else:
        shortcut = MaxPooling3D(1, strides=stride)(x) if stride > 1 else x

    x = Conv3D(filters//2, 1, strides=1, use_bias=False,
               name=name + '_1_conv')(preact)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    x = ZeroPadding3D(padding=((1, 1), (1, 1), (1,1)), name=name + '_2_pad')(x)
    x = Conv3D(filters//2, kernel_size, strides=stride,
               use_bias=False, name=name + '_2_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)

    x = Conv3D(filters, 1, name=name + '_3_conv')(x)
    x = Add(name=name + '_out')([shortcut, x])
    return x

def res_block2_3d_up(x, filters, kernel_size=3, stride=1,
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

    if stride > 1:
        preact = UpSampling3D(size=stride)(preact) if stride > 1 else x

    if conv_shortcut is True:
        shortcut = Conv3D(filters, 1, name=name + '_0_conv')(preact)
    else:
        shortcut = preact

    x = Conv3D(filters//2, 1, strides=1, use_bias=False,
               name=name + '_1_conv')(preact)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    # x = ZeroPadding3D(padding=((1, 1), (1, 1), (1,1)), name=name + '_2_pad')(x)
    x = Conv3D(filters//2, kernel_size,
                        padding='same',
                        use_bias=False,
                        name=name + '_2_conv')(x)
    x = BatchNormalization(axis=bn_axis,
                           epsilon=1.001e-5,
                           name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)

    x = Conv3D(filters, 1, name=name + '_3_conv')(x)
    # print(f'    res block up shortcut {K.get_variable_shape(shortcut)} residual {K.get_variable_shape(x)}')
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
    input_shape = x.shape

    x = res_block2_3d(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = res_block2_3d(x, filters, name=name + '_block' + str(i))

    # x_pre_downsample = x
    x = res_block2_3d(x, filters, stride=stride1, name=name + '_block' + 'down')
    print(f'encoder {name} input {input_shape}  output {x.shape}')
    return x



def FeatureFusionModule3D(input_1, input_2, n_filters):

    # print(f'        fusion input1:{K.get_variable_shape(input_1)} '
    #       f'input2:{K.get_variable_shape(input_2)}')
    inputs = concatenate([input_1, input_2], axis=-1)
    # inputs = bt_act_conv_3d(inputs, n_filters, kernel_size=3)

    inputs = bottle_neck_residual_3d(inputs, n_filters,
                                     kernel_size=3)
    # Global average pooling
    net = GlobalAveragePooling3D()(inputs)
    net = Reshape((1, 1,1, K.get_variable_shape(net)[-1]))(net)
    net = Conv3D(n_filters, kernel_size=1,
                 kernel_initializer='he_normal'
                 )(net)
    net = Activation('relu')(net)
    net = Conv3D(n_filters, kernel_size=1,
                 kernel_initializer='he_normal',
                 )(net)
    net = Activation('sigmoid')(net)
    net = Multiply()([inputs, net])

    net = Add()([inputs, net])

    return net


def up_block(inputs, skip_input, nb_filters, kernel_size,
             dropout_rate = 0.25, nb_resnet=1, upsample_factor = 2):

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(inputs)
    x = Activation('relu')(x)
    x = UpSampling3D(size = upsample_factor)(x)
    # x = bt_act_conv_3d(up,
    #                    nb_filters,
    #                    kernel_size=1,
    #                    )
    # print(f'x shape {x.shape} skip_input shape {skip_input.shape}')
    for i in range(nb_resnet):
        x = bottle_neck_residual_3d(x, nb_filters,
                                    kernel_size=kernel_size,
                                    dropout_rate = dropout_rate)

    # skip_input =  BatchNormalization(axis=-1, epsilon=1.1e-5)(skip_input)
    # skip_input = Activation('relu')(skip_input)
    # if dec_ix == 1:
    print(f' decoder {upsample_factor} skip {skip_input.shape} x {x.shape}')
    x = FeatureFusionModule3D(skip_input, x, nb_filters)
    # else:
    #     x = Add()([skip_input, x])

    return x

def res_stack2_3d_up(x, skip_input, filters, blocks, stride1=2, name=None):
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
    input_shape = K.get_variable_shape(x)
    skip_input_shape = K.get_variable_shape(skip_input)
    x = res_block2_3d_up(x, filters, conv_shortcut=True,
                         stride=stride1, name=name + '_block' + 'up')

    x = FeatureFusionModule3D(x, skip_input, filters)
    # print(f'    shape after fusion {x.shape}')
    # x = res_block2_3d_up(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = res_block2_3d_up(x, filters, name=name + '_block' + str(i))
    # x_pre_downsample = x
    print(f'decoder {name} input {input_shape}  skip_shape {skip_input_shape} output {x.shape}')
    return x


def vb_net_chest(nb_classes,
                 size_factor = 2,
                 output_act = None):

    if nb_classes == 2:
        activation = 'sigmoid'
        nb_classes = 1
    else:
        activation = 'softmax'
        nb_classes = nb_classes + 1

    if output_act is not None:
        activation = output_act

    with tf.device('/device:GPU:0'):
        inputs = Input((None, None, None, 1))
        print(f'initial input {inputs.shape}')
        inital_conv = Conv3D(int(8 * size_factor),
                   strides=2,
                   kernel_size= 5,
                   dilation_rate=1,
                   kernel_initializer='he_normal',
                   padding='same',
                   kernel_regularizer=l2(1e-4),
                   use_bias=False)(inputs)

        # x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
        # inital_conv = Activation('relu')(x)

        # enc1 = MaxPooling3D(pool_size=2)(inital_conv)
        enc1 = res_stack2_3d(inital_conv, 8 * size_factor, 3, name='enc_1')
        # print(f'inital conv and max pool {enc1.shape}')
        enc2 = res_stack2_3d(enc1, 16 * size_factor, 3,name='enc_2')
        enc3 = res_stack2_3d(enc2, 32 * size_factor, 4, name='enc_3')
        enc4 = res_stack2_3d(enc3, 64 * size_factor, 6, name='enc_4')
    with tf.device('/device:GPU:1'):

        dec3 = res_stack2_3d_up(enc4, enc3, 32 * size_factor, 5, name = 'dec3')
        dec2 = res_stack2_3d_up(dec3, enc2, 16 * size_factor, 4, name= 'dec2')
        dec1 = res_stack2_3d_up(dec2, enc1, 8 * size_factor, 4, name= 'dec1')
        out = res_stack2_3d_up(dec1, inital_conv, 4 * size_factor, 4, name='dec0')
        out = BatchNormalization(axis=-1, epsilon=1.1e-5)(out)
        out = Activation('relu')(out)
        out = UpSampling3D(size=2)(out)



        out = Conv3D(nb_classes,
                   kernel_size= 1,
                   kernel_initializer='he_normal',
                   padding='same',
                   activation = activation)(out)
        print(f'final output {out.shape}')
        model = models.Model(inputs, out, name='vb_net_chest')

    return model


# enc2 = down_block(enc1, 32 * size_factor, 3, nb_resnet=nb_resnet, dropout_rate = dropout_rate, enc_ix = 2)
# enc3 = down_block(enc2, 64 * size_factor, 3, nb_resnet=nb_resnet, dropout_rate = dropout_rate, enc_ix = 3)
# enc4 = down_block(enc3, 128 * size_factor, 3, nb_resnet=nb_resnet, dropout_rate = dropout_rate, enc_ix = 4)
# enc5 = down_block(enc4, 128 * size_factor, 3, nb_resnet=nb_resnet, dropout_rate = dropout_rate)
# dec4 = up_block(enc5, enc4, 64 * size_factor, 3, nb_resnet=nb_resnet, dropout_rate = dropout_rate, upsample_factor= 2)
# dec3 = up_block(enc4, enc3, 64 * size_factor, 3, nb_resnet=nb_resnet, dropout_rate = dropout_rate, upsample_factor= 2)
# dec2 = up_block(dec3, enc2, 32 * size_factor, 3, nb_resnet=nb_resnet, dropout_rate = dropout_rate, upsample_factor= 2)
# dec1 = up_block(dec2, inital_conv, 16 * size_factor, 3, nb_resnet=nb_resnet, dropout_rate = dropout_rate, upsample_factor= 4)

if __name__ == '__main__':
    from keras.utils.vis_utils import plot_model
    from time import time
    import os

    result_path = r'/home/dejun/Pictures'
    tic = time()

    model = vb_net_chest(4)

    toc = time()
    print(round(toc - tic, 4))
    model.summary()
    with open(os.path.join(result_path, 'vb_net_chest.json'), 'w') as files:
        files.write(model.to_json())
    # 存储模型图
    plot_model(model, to_file=os.path.join(result_path, 'vb_net_chest.png'), show_shapes=True)

    # tic = time()
    # model_root = r'/media/dejun/holder/models/model_1.1.8.0'
    # model_name = r'VBnet_chest4_segthor'
    #
    # model_json_name = model_name + '.json'
    # model_h5_name = model_name + '.h5'
    # model_json_path = os.path.join(model_root, model_json_name)
    # model_h5_path = os.path.join(model_root, model_h5_name)
    # with open(model_json_path) as file:
    #     model = models.model_from_json(file.read())
    # toc = time()
    # print(f'load model architecture takes {toc-tic}')
    # model.load_weights(model_h5_path)
    #
    # tic = time()
    # print(f'load model weightes takes {tic - toc}')