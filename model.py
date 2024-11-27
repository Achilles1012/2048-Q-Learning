import tensorflow as tf
from keras.models import Model # type: ignore
from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate # type: ignore
import os

ACTION_SPACE = 4
INPUT_SHAPE = (4, 4, 16)
DEPTH_1 = 128
DEPTH_2 = 256
HIDDEN_UNITS = 512
INITIAL_LR = 1e-4

class Model2048:

    def __init__(self, INPUT_SHAPE=(4, 4, 16), ACTION_SPACE=4, pretrained=False):
        self.INPUT_SHAPE = INPUT_SHAPE
        self.ACTION_SPACE = ACTION_SPACE
        self.model = self.get_model()
        """
        if pretrained:
            models = (file for file in os.listdir("DQN-2048-main\\models") if os.path.isfile(os.path.join("DQN-2048-main\\models", file)))
            last_model = sorted(models, key=lambda file_name: int(file_name[10:-3]))[-1]
            saved_model_path = os.path.join("models", last_model)
            self.model.load_weights(saved_model_path)
        """
        if pretrained:
            # Ensure the correct path to the models directory
            models_dir = os.path.join("DQN-2048-main", "models")
            
            # Check if the directory exists before listing files
            if os.path.isdir(models_dir):
                models = (file for file in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, file)))
                last_model = sorted(models, key=lambda file_name: int(file_name[10:-3]))[-1]
                saved_model_path = os.path.join(models_dir, last_model)
                self.model.load_weights(saved_model_path)
            else:
                print(f"Error: The directory '{models_dir}' does not exist.")

    def get_model(self):
        input_layer = Input(shape=self.INPUT_SHAPE)
        conv1 = Conv2D(filters=DEPTH_1, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
        conv1_1 = Conv2D(filters=DEPTH_1, kernel_size=(3, 2), padding='same', activation='relu')(conv1)
        conv1_2 = Conv2D(filters=DEPTH_1, kernel_size=(2, 3), padding='same', activation='relu')(conv1)

        conv2 = Conv2D(filters=DEPTH_2, kernel_size=(2, 2), padding='same', activation='relu')(input_layer)
        conv2_1 = Conv2D(filters=DEPTH_2, kernel_size=(2, 1), padding='same', activation='relu')(conv2)
        conv2_2 = Conv2D(filters=DEPTH_2, kernel_size=(1, 2), padding='same', activation='relu')(conv2)

        conv3 = Conv2D(filters=DEPTH_1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

        flatten = [Flatten()(x) for x in [conv1_1, conv1_2, conv2_1, conv2_2, conv3]]
        concat = Concatenate()(flatten)

        fc_layer1 = Dense(DEPTH_1, activation='relu')(concat)
        fc_layer2 = Dense(DEPTH_2, activation='relu')(fc_layer1)

        fc_layer3 = Dense(self.ACTION_SPACE, activation='linear')(fc_layer2)

        model = Model(inputs=input_layer, outputs=fc_layer3)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(float(INITIAL_LR), 50, 0.90, staircase=True)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])

        return model
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

