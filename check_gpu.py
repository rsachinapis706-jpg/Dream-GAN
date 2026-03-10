import tensorflow as tf
print('\n==============================')
print('TensorFlow version:', tf.__version__)
print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))
for gpu in tf.config.list_physical_devices('GPU'):
    print('GPU Details:', gpu)
print('==============================\n')
