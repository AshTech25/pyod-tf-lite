# -*- coding: utf-8 -*-
"""Cook's distance outlier detection (CD)
"""

# Author: D Kulik
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy as np


def _Cooks_dist(X, y, model):
    """Calculated the Cook's distance

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The training dataset.

    y : numpy array of shape (n_samples)
        The training datset

    model : object
        Regression model used to calculate the Cook's distance

    Returns
    -------
    distances_ : numpy array of shape (n_samples)
        Cook's distance
    """

    # Leverage is computed as the diagonal of the projection matrix of X
    leverage = (X * np.linalg.pinv(X).T).sum(1)

    # Compute the rank and the degrees of freedom of the model
    rank = np.linalg.matrix_rank(X)
    df = X.shape[0] - rank

    # Compute the MSE from the residuals
    residuals = y - model.predict(X)
    mse = np.dot(residuals, residuals) / df

    # Compute Cook's distance
    if (mse != 0) or (mse != np.nan):
        residuals_studentized = residuals / np.sqrt(mse) / np.sqrt(
            1 - leverage)
        distance_ = residuals_studentized ** 2 / X.shape[1]
        distance_ *= leverage / (1 - leverage)
        distance_ = ((distance_ - distance_.min())
                     / (distance_.max() - distance_.min()))

    else:
        distance_ = np.ones(len(y)) * np.nan

    return distance_


def _process_distances(X, model):
    """Calculated the mean Cook's distances for
    each feature

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The training dataset.

    model : object
        Regression model used to calculate the Cook's distance

    Returns
    -------
    distances_ : numpy array of shape (n_samples)
        mean Cook's distance
    """

    distances_ = []
    for i in range(X.shape[1]):
        mod = model

        # Extract new X and y inputs
        exp = np.delete(X.copy(), i, axis=1)
        resp = X[:, i]

        exp = exp.reshape(-1, 1) if exp.ndim == 1 else exp

        # Fit the model
        mod.fit(exp, resp)

        # Get Cook's Distance
        distance_ = _Cooks_dist(exp, resp, mod)

        distances_.append(distance_)

    distances_ = np.nanmean(distances_, axis=0)

    return distances_


class CD(BaseDetector):
    """Cook's distance can be used to identify points that negatively
       affect a regression model. A combination of each observation’s
       leverage and residual values are used in the measurement. Higher
       leverage and residuals relate to  higher Cook’s distances. Note
       that this method is unsupervised and requires at least two 
       features for X with which to calculate the mean Cook's distance
       for each datapoint. Read more in the :cite:`cook1977detection`.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
        
    model : object, optional (default=LinearRegression())
        Regression model used to calculate the Cook's distance
          

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
       The modified z-score to use as a threshold. Observations with
       a modified z-score (based on the median absolute deviation) greater
       than this value will be classified as outliers.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
        """

    def __init__(self, contamination=0.1, model=LinearRegression()):
        super(CD, self).__init__(contamination=contamination)
        self.model = model

    def fit(self, X, y=None):
        """"Fit detector. y is ignored in unsupervised methods.

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

        # Validate inputs X and y
        X = check_array(X)

        self._set_n_classes(y)

        # Get Cook's distance
        distances_ = _process_distances(X, self.model)

        self.decision_scores_ = distances_

        self._process_decision_scores()

        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        For consistency, outliers are assigned with larger anomaly scores.

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

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        # Validate input X
        X = check_array(X)

        # Get Cook's distance
        distances_ = _process_distances(X, self.model)

        return distances_

    def convert_to_onnx(self,model,file_name):
        initial_type = [('float_input', FloatTensorType([None, 4]))]
        onx = convert_sklearn(model, initial_types=initial_type)
        with open(file_name, "wb") as f:
            f.write(onx.SerializeToString())

    def predict_with_onnx(self, file_name, X_test):
        sess = rt.InferenceSession(file_name)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
        return pred_onx
    
    def quantize_onnx(self, model, file_name):
        initial_type = [('float_input', FloatTensorType([None, 4]))]
        onx = convert_sklearn(model, initial_types=initial_type)
        quantized_model = rt.quantization.quantize_static(onx, quantization_mode=rt.quantization.QuantizationMode.IntegerOps, force_fusions=True)
        with open(file_name, "wb") as f:
            f.write(quantized_model.SerializeToString())