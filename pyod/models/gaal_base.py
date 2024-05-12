# -*- coding: utf-8 -*-
"""Base file for Generative Adversarial Active Learning.
Part of the codes are adapted from
https://github.com/leibinghe/GAAL-based-outlier-detection
"""
# Author: Winston Li <jk_zhengli@hotmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function
import numpy as np
import math
import pathlib
from .base_dl import _get_tensorflow_version
import tensorflow as tf

# if tensorflow 2, import from tf directly
if _get_tensorflow_version() <= 200:
    import keras
    from keras.layers import Input, Dense
    from keras.models import Sequential, Model
else:
    import tensorflow.keras as keras
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Sequential, Model


# TODO: create a base class for so_gaal and mo_gaal
def create_discriminator(latent_size, data_size):  # pragma: no cover
    """Create the discriminator of the GAN for a given latent size.

    Parameters
    ----------
    latent_size : int
        The size of the latent space of the generator.

    data_size : int
        Size of the input data.

    Returns
    -------
    D : Keras model() object
        Returns a model() object.
    """

    dis = Sequential()
    dis.add(Dense(int(math.ceil(math.sqrt(data_size))),
                  input_dim=latent_size, activation='relu',
                  kernel_initializer=keras.initializers.VarianceScaling(
                      scale=1.0, mode='fan_in', distribution='normal',
                      seed=None)))
    dis.add(Dense(1, activation='sigmoid',
                  kernel_initializer=keras.initializers.VarianceScaling(
                      scale=1.0, mode='fan_in', distribution='normal',
                      seed=None)))
    data = Input(shape=(latent_size,))
    fake = dis(data)
    return Model(data, fake)


def create_generator(latent_size):  # pragma: no cover
    """Create the generator of the GAN for a given latent size.

    Parameters
    ----------
    latent_size : int
        The size of the latent space of the generator

    Returns
    -------
    D : Keras model() object
        Returns a model() object.
    """

    gen = Sequential()
    gen.add(Dense(latent_size, input_dim=latent_size, activation='relu',
                  kernel_initializer=keras.initializers.Identity(
                      gain=1.0)))
    gen.add(Dense(latent_size, activation='relu',
                  kernel_initializer=keras.initializers.Identity(
                      gain=1.0)))
    latent = Input(shape=(latent_size,))
    fake_data = gen(latent)
    return Model(latent, fake_data)

@staticmethod
def representative_dataset(self):
    for _ in range(100):
        data = np.random.rand(1, 244, 244, 3)
        yield [data.astype(np.float32)]


def convert_generator_tflite(self, latent_size, model_path= 'anomaly_model_quant'):
    # Build the model using the _build_model method of the class
    model = self.create_generator(latent_size)
    
    # Create a TFLite converter instance and initialize it with the Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization options for the converter
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set a representative dataset for quantization
    # Note: The `self.representative_dataset` should be defined elsewhere in the class
    converter.representative_dataset = self.representative_dataset
    
    # Set the supported operations for the target device
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Set the input and output types for inference
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    
    # Convert the model to a quantized TensorFlow Lite model
    tflite_quant_model = converter.convert()

    tflite_models_dir = pathlib.Path("/tmp/tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    
    tflite_model_quant_file = tflite_models_dir/f"{model_path}.tflite"
    tflite_model_quant_file.write_bytes(tflite_quant_model)

    return tflite_model_quant_file

def convert_discriminator_tflite(self, latent_size, model_path= 'anomaly_model_quant'):
    # Build the model using the _build_model method of the class
    model = self.create_discriminator(latent_size)
    
    # Create a TFLite converter instance and initialize it with the Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization options for the converter
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set a representative dataset for quantization
    # Note: The `self.representative_dataset` should be defined elsewhere in the class
    converter.representative_dataset = self.representative_dataset
    
    # Set the supported operations for the target device
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Set the input and output types for inference
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    
    # Convert the model to a quantized TensorFlow Lite model
    tflite_quant_model = converter.convert()

    tflite_models_dir = pathlib.Path("/tmp/tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    
    tflite_model_quant_file = tflite_models_dir/f"{model_path}.tflite"
    tflite_model_quant_file.write_bytes(tflite_quant_model)

    return tflite_model_quant_file
    
def representative_dataset(self):
    for _ in range(100):
        data = np.random.rand(1, 244, 244, 3)
        yield [data.astype(np.float32)]


def convert_to_tflite(self, model, model_path= 'anomaly_model_quant'):
        # Create a TFLite converter instance and initialize it with the Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
        # Set optimization options for the converter
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
        # Set a representative dataset for quantization
        # Note: The `self.representative_dataset` should be defined elsewhere in the class
    converter.representative_dataset = self.representative_dataset
    
        # Set the supported operations for the target device
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
        # Set the input and output types for inference
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    
        # Convert the model to a quantized TensorFlow Lite model
    tflite_quant_model = converter.convert()

    tflite_models_dir = pathlib.Path("/tmp/tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    
    tflite_model_quant_file = tflite_models_dir/f"{model_path}.tflite"
    tflite_model_quant_file.write_bytes(tflite_quant_model)

    return tflite_model_quant_file
    
def run_tflite_model(tflite_file, X_test, index):
        
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    for i, ind in enumerate(index):
        test_X = X_test[ind]

    # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_X / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

            
    return output
