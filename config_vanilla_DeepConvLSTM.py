# coding=utf-8

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NUM_SENSOR_CHANNELS = 113

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 12

# Batch Size
BATCH_SIZE = 100

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

# Dropout rate
DROPOUT_RATE = .5

# Number of epoches
NUM_EPOCHES = 1000

# Number for model
TEST_MODEL_NUMBER = 1494275214


from keras import backend as K
K.set_image_dim_ordering('tf')
