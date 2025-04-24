import tensorflow as tf

"""
print("Cuda available: ", tf.test.is_built_with_cuda())

print("GPU available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus=tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    print("Device name:", gpu.name)
"""
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
