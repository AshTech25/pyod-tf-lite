# -*- coding: utf-8 -*-
"""Deep Isolation Forest for Anomaly Detection (DIF)
"""
# Author: Hongzuo Xu <hongzuoxu@126.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader

from .base import BaseDetector
from ..utils.torch_utility import get_activation_by_name


class DIF(BaseDetector):
	"""Deep Isolation Forest (DIF) is an extension of iForest. It uses deep
	representation ensemble to achieve non-linear isolation on original data
	space. See :cite:`xu2023dif` for details.

	Parameters
	----------
	batch_size : int, optional (default=1000)
		Number of samples per gradient update.

	representation_dim, int, optional (default=20)
		Dimensionality of the representation space.

	hidden_neurons, list, optional (default=[64, 32])
		The number of neurons per hidden layers. So the network has the
		structure as [n_features, hidden_neurons[0], hidden_neurons[1], ..., representation_dim]

	hidden_activation, str, optional (default='tanh')
		Activation function to use for hidden layers.
		All hidden layers are forced to use the same type of activation.
		See https://pytorch.org/docs/stable/nn.html for details.
		Currently only
		'relu': nn.ReLU()
		'sigmoid': nn.Sigmoid()
		'tanh': nn.Tanh()
		are supported. See pyod/utils/torch_utility.py for details.

	skip_connection, boolean, optional (default=False)
		If True, apply skip-connection in the neural network structure.

	n_ensemble, int, optional (default=50)
		The number of deep representation ensemble members.

	n_estimators, int, optional (default=6)
		The number of isolation forest of each representation.

	max_samples, int, optional (default=256)
		The number of samples to draw from X to train each base isolation tree.

	contamination : float in (0., 0.5), optional (default=0.1)
		The amount of contamination of the data set,
		i.e. the proportion of outliers in the data set. Used when fitting to
		define the threshold on the decision function.

	random_state : int or None, optional (default=None)
		If int, random_state is the seed used by the random
		number generator;
		If None, the random number generator is the
		RandomState instance used by `np.random`.

	device, 'cuda', 'cpu', or None, optional (default=None)
		if 'cuda', use GPU acceleration in torch
		if 'cpu', use cpu in torch
		if None, automatically determine whether GPU is available


	Attributes
	----------
	net_lst : list of torch.Module
		The list of representation neural networks.

	iForest_lst : list of iForest
		The list of instantiated iForest model.

	x_reduced_lst: list of numpy array
		The list of training data representations

	decision_scores_ : numpy array of shape (n_samples,)
		The outlier scores of the training data.
		The higher, the more abnormal. Outliers tend to have higher
		scores. This value is available once the detector is fitted.

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

	def __init__(self,
				 batch_size=1000,
				 representation_dim=20,
				 hidden_neurons=None,
				 hidden_activation='tanh',
				 skip_connection=False,
				 n_ensemble=50,
				 n_estimators=6,
				 max_samples=256,
				 contamination=0.1,
				 random_state=None,
				 device=None):
		super(DIF, self).__init__(contamination=contamination)
		self.batch_size = batch_size
		self.representation_dim = representation_dim
		self.hidden_activation = hidden_activation
		self.skip_connection = skip_connection
		self.hidden_neurons = hidden_neurons

		self.n_ensemble = n_ensemble
		self.n_estimators = n_estimators
		self.max_samples = max_samples

		self.random_state = random_state
		self.device = device

		self.minmax_scaler = None

		# create default calculation device (support GPU if available)
		if self.device is None:
			self.device = torch.device(
				"cuda:0" if torch.cuda.is_available() else "cpu")

		# set random seed
		if self.random_state is not None:
			torch.manual_seed(self.random_state)
			torch.cuda.manual_seed(self.random_state)
			torch.cuda.manual_seed_all(self.random_state)
			np.random.seed(self.random_state)

		# default values for the amount of hidden neurons
		if self.hidden_neurons is None:
			self.hidden_neurons = [500, 100]

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

		n_samples, n_features = X.shape[0], X.shape[1]

		# conduct min-max normalization before feeding into neural networks
		self.minmax_scaler = MinMaxScaler()
		self.minmax_scaler.fit(X)
		X = self.minmax_scaler.transform(X)

		# prepare neural network parameters
		network_params = {
			'n_features': n_features,
			'n_hidden': self.hidden_neurons,
			'n_output': self.representation_dim,
			'activation': self.hidden_activation,
			'skip_connection': self.skip_connection
		}

		# iteration
		self.net_lst = []
		self.iForest_lst = []
		self.x_reduced_lst = []
		ensemble_seeds = np.random.randint(0, 100000, self.n_ensemble)
		for i in range(self.n_ensemble):
			# instantiate network class and seed random seed
			net = MLPnet(**network_params).to(self.device)
			torch.manual_seed(ensemble_seeds[i])

			# initialize network parameters
			for name, param in net.named_parameters():
				if name.endswith('weight'):
					torch.nn.init.normal_(param, mean=0., std=1.)

			x_reduced = self._deep_representation(net, X)

			# save network and representations
			self.x_reduced_lst.append(x_reduced)
			self.net_lst.append(net)

			# perform iForest upon representations
			self.iForest_lst.append(
				IsolationForest(n_estimators=self.n_estimators,
								max_samples=self.max_samples,
								random_state=ensemble_seeds[i])
			)
			self.iForest_lst[i].fit(x_reduced)

		self.decision_scores_ = self.decision_function(X)
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
		check_is_fitted(self, ['net_lst', 'iForest_lst', 'x_reduced_lst'])
		X = check_array(X)

		# conduct min-max normalization before feeding into neural networks
		X = self.minmax_scaler.transform(X)

		testing_n_samples = X.shape[0]
		score_lst = np.zeros([self.n_ensemble, testing_n_samples])

		# iteration
		for i in range(self.n_ensemble):
			# transform testing data to representation
			x_reduced = self._deep_representation(self.net_lst[i], X)

			# calculate outlier scores
			scores = _cal_score(x_reduced, self.iForest_lst[i])
			score_lst[i] = scores

		final_scores = np.average(score_lst, axis=0)
		return final_scores

	def _deep_representation(self, net, X):
		x_reduced = []

		with torch.no_grad():
			loader = DataLoader(X, batch_size=self.batch_size,
								drop_last=False, pin_memory=True,
								shuffle=False)
			for batch_x in loader:
				batch_x = batch_x.float().to(self.device)
				batch_x_reduced = net(batch_x)
				x_reduced.append(batch_x_reduced)

		x_reduced = torch.cat(x_reduced).data.cpu().numpy()
		x_reduced = StandardScaler().fit_transform(x_reduced)
		x_reduced = np.tanh(x_reduced)
		return x_reduced


class MLPnet(torch.nn.Module):
	def __init__(self, n_features, n_hidden=[500, 100], n_output=20,
				 activation='ReLU', bias=False, batch_norm=False,
				 skip_connection=False):
		super(MLPnet, self).__init__()
		self.skip_connection = skip_connection
		self.n_output = n_output

		num_layers = len(n_hidden)

		if type(activation) == str:
			activation = [activation] * num_layers
			activation.append(None)

		assert len(activation) == len(
			n_hidden) + 1, 'activation and n_hidden are not matched'

		self.layers = []
		for i in range(num_layers + 1):
			in_channels, out_channels = \
				self.get_in_out_channels(i, num_layers, n_features,
										 n_hidden, n_output, skip_connection)
			self.layers += [
				LinearBlock(in_channels, out_channels,
							bias=bias, batch_norm=batch_norm,
							activation=activation[i],
							skip_connection=skip_connection if i != num_layers else False)
			]
		self.network = torch.nn.Sequential(*self.layers)

	def forward(self, x):
		x = self.network(x)
		return x

	@staticmethod
	def get_in_out_channels(i, num_layers, n_features, n_hidden, n_output,
							skip_connection):
		if skip_connection is False:
			in_channels = n_features if i == 0 else n_hidden[i - 1]
			out_channels = n_output if i == num_layers else n_hidden[i]
		else:
			in_channels = n_features if i == 0 else np.sum(
				n_hidden[:i]) + n_features
			out_channels = n_output if i == num_layers else n_hidden[i]
		return in_channels, out_channels


class LinearBlock(torch.nn.Module):
	def __init__(self, in_channels, out_channels,
				 activation='Tanh', bias=False, batch_norm=False,
				 skip_connection=False):
		super(LinearBlock, self).__init__()

		self.skip_connection = skip_connection

		self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)

		if activation is not None:
			# self.act_layer = _instantiate_class("torch.nn.modules.activation", activation)
			self.act_layer = get_activation_by_name(activation)
		else:
			self.act_layer = torch.nn.Identity()

		self.batch_norm = batch_norm
		if batch_norm is True:
			dim = out_channels
			self.bn_layer = torch.nn.BatchNorm1d(dim, affine=bias)

	def forward(self, x):
		x1 = self.linear(x)
		x1 = self.act_layer(x1)

		if self.batch_norm is True:
			x1 = self.bn_layer(x1)

		if self.skip_connection:
			x1 = torch.cat([x, x1], axis=1)

		return x1


def _cal_score(xx, clf):
	depths = np.zeros((xx.shape[0], len(clf.estimators_)))
	depth_sum = np.zeros(xx.shape[0])
	deviations = np.zeros((xx.shape[0], len(clf.estimators_)))
	leaf_samples = np.zeros((xx.shape[0], len(clf.estimators_)))

	for ii, estimator_tree in enumerate(clf.estimators_):
		tree = estimator_tree.tree_
		n_node = tree.node_count

		if n_node == 1:
			continue

		# get feature and threshold of each node in the iTree
		# in feature_lst, -2 indicates the leaf node
		feature_lst, threshold_lst = tree.feature.copy(), tree.threshold.copy()

		# compute depth and score
		leaves_index = estimator_tree.apply(xx)
		node_indicator = estimator_tree.decision_path(xx)

		# The number of training samples in each test sample leaf
		n_node_samples = estimator_tree.tree_.n_node_samples

		# node_indicator is a sparse matrix with shape (n_samples, n_nodes),
		# indicating the path of input data samples
		# each layer would result in a non-zero element in this matrix,
		# and then the row-wise summation is the depth of data sample
		n_samples_leaf = estimator_tree.tree_.n_node_samples[leaves_index]
		d = (np.ravel(node_indicator.sum(axis=1)) + _average_path_length(
			n_samples_leaf) - 1.0)
		depths[:, ii] = d
		depth_sum += d

		# decision path of data matrix XX
		node_indicator = np.array(node_indicator.todense())

		# set a matrix with shape [n_sample, n_node],
		# representing the feature value of each sample on each node
		# set the leaf node as -2
		value_mat = np.array([xx[i][feature_lst] for i in range(xx.shape[0])])
		value_mat[:, np.where(feature_lst == -2)[0]] = -2
		th_mat = np.array([threshold_lst for _ in range(xx.shape[0])])

		mat = np.abs(value_mat - th_mat) * node_indicator

		exist = (mat != 0)
		dev = mat.sum(axis=1) / (exist.sum(axis=1) + 1e-6)
		deviations[:, ii] = dev

	scores = 2 ** (-depth_sum / (len(clf.estimators_) * _average_path_length(
		[clf.max_samples_])))
	deviation = np.mean(deviations, axis=1)
	leaf_sample = (clf.max_samples_ - np.mean(leaf_samples,
											  axis=1)) / clf.max_samples_

	scores = scores * deviation
	# scores = scores * deviation * leaf_sample
	return scores


def _average_path_length(n_samples_leaf):
	"""
	The average path length in a n_samples iTree, which is equal to
	the average path length of an unsuccessful BST search since the
	latter has the same structure as an isolation tree.
	Parameters
	----------
	n_samples_leaf : array-like of shape (n_samples,)
		The number of training samples in each test sample leaf, for
		each estimators.

	Returns
	-------
	average_path_length : ndarray of shape (n_samples,)
	"""

	n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

	n_samples_leaf_shape = n_samples_leaf.shape
	n_samples_leaf = n_samples_leaf.reshape((1, -1))
	average_path_length = np.zeros(n_samples_leaf.shape)

	mask_1 = n_samples_leaf <= 1
	mask_2 = n_samples_leaf == 2
	not_mask = ~np.logical_or(mask_1, mask_2)

	average_path_length[mask_1] = 0.
	average_path_length[mask_2] = 1.
	average_path_length[not_mask] = (
			2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
			- 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
	)

	return average_path_length.reshape(n_samples_leaf_shape)

