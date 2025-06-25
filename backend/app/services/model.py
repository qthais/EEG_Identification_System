
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
def build_cnn2d_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # CNN 2D layers
    x = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.03))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.03))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.03))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.03))(x)  # Thêm L2 regularization
    x= Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model