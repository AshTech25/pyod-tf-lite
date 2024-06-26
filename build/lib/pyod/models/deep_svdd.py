# -*- coding: utf-8 -*-
"""Deep One-Class Classification for outlier detection
"""
# Author: Rafal Bodziony <bodziony.rafal@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import pathlib
from .base import BaseDetector
from .base_dl import _get_tensorflow_version
from ..utils.utility import check_parameter

# if tensorflow 2, import from tf directly
if _get_tensorflow_version() >= 200:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras import Model, Input
else:
    raise ModuleNotFoundError('DeepSVDD runs only with TensorFlow 2.0+')


class DeepSVDD(BaseDetector):
    """Deep One-Class Classifier with AutoEncoder (AE) is a type of neural
    networks for learning useful data representations in an unsupervised way.
    DeepSVDD trains a neural network while minimizing the volume of a
    hypersphere that encloses the network representations of the data,
    forcing the network to extract the common factors of variation.
    Similar to PCA, DeepSVDD could be used to detect outlying objects in the
    data by calculating the distance from center
    See :cite:`ruff2018deepsvdd` for details.

    Parameters
    ----------
    c: float, optional (default='forwad_nn_pass')
        Deep SVDD center, the default will be calculated based on network
        initialization first forward pass. To get repeated results set
        random_state if c is set to None.

    use_ae: bool, optional (default=False)
        The AutoEncoder type of DeepSVDD it reverse neurons from hidden_neurons
        if set to True.

    hidden_neurons : list, optional (default=[64, 32])
        The number of neurons per hidden layers. if use_ae is True, neurons
        will be reversed eg. [64, 32] -> [64, 32, 32, 64, n_features]

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://keras.io/activations/

    output_activation : str, optional (default='sigmoid')
        Activation function to use for output layer.
        See https://keras.io/activations/

    optimizer : str, optional (default='adam')
        String (name of optimizer) or optimizer instance.
        See https://keras.io/optimizers/

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    l2_regularizer : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer. By default, l2 regularizer is used. See
        https://keras.io/regularizers/

    validation_size : float in (0., 1), optional (default=0.1)
        The percentage of data to be used for validation.

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    verbose : int, optional (default=1)
        Verbosity mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

        For verbose >= 1, model summary may be printed.

    random_state : random_state: int, RandomState instance or None, optional
        (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is used
        to define the threshold on the decision function.

    Attributes
    ----------
    model_ : Keras Object
        The underlying DeppSVDD in Keras.

    history_: Keras Object
        The AutoEncoder training history.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, c=None,
                 use_ae=False,
                 hidden_neurons=None,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 optimizer='adam',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None, contamination=0.1):
        super(DeepSVDD, self).__init__(contamination=contamination)
        self.c = c
        self.use_ae = use_ae
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state

        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
        # default values
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32]

        self.hidden_neurons_ = self.hidden_neurons

        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)

    def _init_c(self, X_norm, eps=0.1):
        # create true Center value from model predict of intermediate layers
        model_center = Model(self.model_.inputs,
                             self.model_.get_layer('net_output').output)

        out_ = model_center.predict(X_norm)
        nf_predict = out_.shape[0]
        out_ = np.sum(out_, axis=0)
        out_ /= nf_predict
        self.c = out_
        self.c[(abs(self.c) < eps) & (self.c < 0)] = -eps
        self.c[(abs(self.c) < eps) & (self.c > 0)] = eps

        return self

    def _build_model(self, training=True):

        inputs = Input(shape=(self.n_features_,))
        x = Dense(self.hidden_neurons_[0], use_bias=False, activation=self.hidden_activation,
                  activity_regularizer=l2(self.l2_regularizer))(inputs)
        for hidden_neurons in self.hidden_neurons_[1:-1]:
            x = Dense(hidden_neurons, use_bias=False, activation=self.hidden_activation,
                      activity_regularizer=l2(self.l2_regularizer))(x)
            x = Dropout(self.dropout_rate)(x)

        # add name to last hidden layer
        x = Dense(self.hidden_neurons_[-1], use_bias=False, activation=self.hidden_activation,
                  activity_regularizer=l2(self.l2_regularizer),
                  name='net_output')(x)

        # build distance loss
        dist = tf.math.reduce_sum((x - self.c) ** 2, axis=-1)
        outputs = dist
        loss = tf.math.reduce_mean(dist)

        # Instantiate Deep SVDD
        dsvd = Model(inputs, outputs)

        # Weight decay
        w_d = 1e-6 * sum([np.linalg.norm(w) for w in dsvd.get_weights()])

        # Use AutoEncoder version of DeepSVDD
        if self.use_ae:
            for reversed_neurons in self.hidden_neurons_[::-1]:
                x = Dense(reversed_neurons, use_bias=False, activation=self.hidden_activation,
                          activity_regularizer=l2(self.l2_regularizer))(x)
                x = Dropout(self.dropout_rate)(x)
            x = Dense(self.n_features_, use_bias=False, activation=self.output_activation,
                      activity_regularizer=l2(self.l2_regularizer))(x)
            dsvd.add_loss(
                loss + tf.math.reduce_mean(tf.math.square(x - inputs)) + w_d)
        else:
            dsvd.add_loss(loss + w_d)

        dsvd.compile(optimizer=self.optimizer)

        if self.verbose >= 1 and training:
            print(dsvd.summary())
        return dsvd
    
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

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        # Verify and construct the hidden units
        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

        # Standardize data for better performance
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)

        # Shuffle the data for validation as Keras do not shuffling for
        # Validation Split
        np.random.shuffle(X_norm)

        # Validate and complete the number of hidden neurons
        if np.min(self.hidden_neurons) > self.n_features_ and self.use_ae:
            raise ValueError("The number of neurons should not exceed "
                             "the number of features")
        if self.c is None:
            self.c = 0.0
            self.model_ = self._build_model(training=False)
            self._init_c(X_norm)

        # Build DeepSVDD model & fit with X
        self.model_ = self._build_model()
        self.history_ = self.model_.fit(X_norm, X_norm,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        validation_split=self.validation_size,
                                        verbose=self.verbose).history
        # Predict on X itself and calculate the reconstruction error as
        # the outlier scores. Noted X_norm was shuffled has to recreate
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        self.decision_scores_ = self.model_.predict(X_norm)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model_', 'history_'])
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        # Predict on X and return the reconstruction errors
        pred_scores = self.model_.predict(X_norm)
        return pred_scores
