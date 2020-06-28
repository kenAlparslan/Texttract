import os
from utils import *
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, ELU


def get_architecture(model_name="InceptionResNetV2", input_dim_width=224, input_dim_length=224,num_dense_layers=0,num_dense_nodes=0,num_class=16,dropout_pct=0.2, weights=None):

    # priors to use for base architecture
    def create_normal_residual_block(inputs, ch, N):
        # Conv with skip connections
        x = inputs
        for i in range(N):
            # adjust channels
            if i == 0:
                skip = Conv2D(ch, 1)(x)
                skip = BatchNormalization()(skip)
                skip = Activation("relu")(skip)
            else:
                skip = x
            x = Conv2D(ch, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(ch, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Add()([x, skip])
        return x



    def wide_resnet(N=1, k=1,input_dim_width=input_dim_width, input_dim_length=input_dim_length,num_class=num_class):
        """
        Create vanilla conv Wide ResNet (N=4, k=10)
        """
        # input
        input =Input((input_dim_width,input_dim_length,3))
        # 16 channels block
        x = Conv2D(16, 3, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # 1st block
        x = create_normal_residual_block(x, 16*k, N)
        # The original wide resnet is stride=2 conv for downsampling,
        # but replace them to average pooling because centers are shifted when octconv
        # 2nd block
        x = AveragePooling2D(2)(x)
        x = create_normal_residual_block(x, 32*k, N)
        # 3rd block
        x = AveragePooling2D(2)(x)
        x = create_normal_residual_block(x, 64*k, N)
        # FC
        x = GlobalAveragePooling2D()(x)
        x = Dense(num_class, activation="softmax")(x)

        model = Model(input, x)
        return model



    def very_wide_resnet(N=1, k=1,input_dim_width=input_dim_width, input_dim_length=input_dim_length,num_class=num_class):
        """
        Create vanilla conv Wide ResNet (N=4, k=10)
        """
        # input
        input =Input((input_dim_width,input_dim_length,3))
        # 16 channels block
        x = Conv2D(64, 3, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # 1st block
        x = create_normal_residual_block(x, 64*k, N)
        # The original wide resnet is stride=2 conv for downsampling,
        # but replace them to average pooling because centers are shifted when octconv
        # 2nd block
        x = AveragePooling2D(2)(x)
        x = create_normal_residual_block(x, 128*k, N)
        # 3rd block
        x = AveragePooling2D(2)(x)
        x = create_normal_residual_block(x, 256*k, N)
        # FC
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout_pct)(x)
        x = Dense(num_class, activation="softmax")(x)

        model = Model(input, x)
        return model




    print(str(model_name)) # InceptionResNetV2
    base_model = InceptionResNetV2(input_shape= (input_dim_width, input_dim_length, 3), weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = ELU(alpha=1.0)(x)

    if num_dense_layers == 2:
        x = Dense(num_dense_nodes)(x)
        x = Dropout(dropout_pct)(x)
        x = Dense(num_dense_nodes)(x)
        x = Dropout(dropout_pct)(x)

    elif num_dense_layers == 1:
        x = Dense(num_dense_nodes)(x)
        x = Dropout(dropout_pct)(x)

    elif dropout_pct > 0:
        x = Dropout(dropout_pct)(x)
        #x = keras.layers.AlphaDropout(dropout_pct, noise_shape=None, seed=None)(x)

    predictions = Dense(num_class, activation='softmax',name='predictions',kernel_initializer='glorot_normal', 
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
            kernel_constraint=None, bias_constraint=None)(x)
    model = Model(inputs=base_model.input, outputs=predictions, name = str(model_name))

    return model
