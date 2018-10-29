# Copyright 2018 Srikanth Gadicherla @imsrgadich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
from gpflow.decors import params_as_tensors
from gpflow.params import Parameter
from gpflow.kernels import Kernel,RBF,Cosine
from gpflow import transforms, autoflow,settings

class SpectralMixture(Kernel):
    def __init__(self, num_mixtures=1, mixture_weights=None,\
                 mixture_scales=None,mixture_means=None,\
                 input_dim=1,active_dims=None,name=None):
        """
        - num_mixtures is the number of mixtures; denoted as Q in
        Wilson 2013.
        - mixture_weights
        - mixture_variance is
        - mixture_means is the list (or array) of means of the
        mixtures.
        - input_dim is the dimension of the input to the kernel.
        - active_dims is the dimension of the X which needs to be used.
        References:
        http://hips.seas.harvard.edu/files/wilson-extrapolation-icml-2013_0.pdf
        http://www.cs.cmu.edu/~andrewgw/typo.pdf
        """
        super().__init__(input_dim,active_dims,name=name)
        # Q(num_of_mixtures)=1 then SM kernel is SE Kernel.
        if num_mixtures == 1:
            print("Using default mixture = 1")

        # number of mixtures is non trainable.
        self.num_mixtures = Parameter(num_mixtures,trainable=False)
        self.mixture_weights = Parameter(mixture_weights,transform=transforms.positive)
        self.mixture_scales = Parameter(mixture_scales,transform=transforms.positive)
        self.mixture_means = Parameter(mixture_means,transform=transforms.positive)

    @params_as_tensors
    def K(self, X1, X2=None):
        if self.mixture_weights == None or self.mixture_means == None \
                                      or self.mixture_scales == None:
                raise RuntimeError('Parameters of spectral mixture kernel not initialized.\
                                    Run `sm_kern_object.initialize_(train_x,train_y)`.')
                # initialization can only be done by user as it needs target data as well.
        if X2 is None:
            X2 = X1

        # get absolute distances
        X1 = tf.transpose(tf.expand_dims(X1, -1), perm=[1, 0, 2])  # D x N1 x 1
        X2 = tf.expand_dims(tf.transpose(X2, perm=[1, 0]), -2)  # D x 1 x N2

        r = tf.abs(tf.subtract(X1, X2)) # D x N1 x N2


        cos_term = tf.multiply(tf.tensordot(self.mixture_means, r, axes=((1),(0))), 2. * np.pi)
        # num_mixtures x N1 x N2

        scales_expand = tf.expand_dims(tf.expand_dims(self.mixture_scales, -2), -2)
                                                                # D x 1 x 1 x num_mixtures
        r_tile = tf.tile(tf.expand_dims(r,-1),(1,1,1,self.num_mixtures))
                                                               # D x N1 x N2 x num_mixtures
        exp_term = tf.multiply(tf.transpose(tf.reduce_sum(tf.square(tf.multiply(r_tile, scales_expand)), 0)\
                                            ,perm=[2, 0, 1]), -2. * np.pi ** 2)
                                                            # num_mixtures x N1 x N2

        weights = tf.expand_dims(tf.expand_dims(self.mixture_weights,-1),-1)
        weights = tf.tile(weights,(1,tf.shape(X1)[1],tf.shape(X2)[2]))
        return tf.reduce_sum(tf.multiply(weights,tf.multiply(tf.exp(exp_term),tf.cos(cos_term))),0)


    @params_as_tensors
    def Kdiag(self, X):

        # just the sum of weights. Weights represent the signal
        # variance.
        return tf.fill(tf.stack([tf.shape(X)[0]]),tf.reduce_sum(self.mixture_weights,0))


def sm_init(train_x, train_y, num_mixtures):
    """
    For initialization of the parameters for the Spectral Mixture
    Kernel.
    :param train_x: input data
    :param train_y: target data
    :param num_mixtures: number of mixtures
    :return: param_name       dimensions
             ----------       ----------
             mixture weights| num_mixtures x 1
             mixture means  | num_mixtures x input_dim
             mixture scales | input_dim x num_mixtures
    """
    assert isinstance(num_mixtures, int)
    assert train_x.shape[0] == train_y.shape[0]

    input_dim = np.shape(train_x)[1]  # type: int

    if np.size(train_x.shape) == 1:
        train_x = np.expand_dims(train_x ,-1)

    if np.size(train_x.shape) == 2:
        train_x = np.expand_dims(train_x ,0)

    train_x_sort = np.copy(train_x)
    train_x_sort.sort(axis=1)

    max_dist = np.squeeze(train_x_sort[: ,-1, :] - train_x_sort[: ,0, :])

    min_dist_sort = np.squeeze(np.abs(train_x_sort[: ,1:, :] - train_x_sort[: ,:-1, :]))
    min_dist = np.zeros([input_dim] ,dtype=float)

    # min of each data column could be zero. Hence, picking minimum which is not zero
    for ind in np.arange(input_dim):
        try:
            min_dist[ind] = min_dist_sort[np.amin(np.where(min_dist_sort[:,ind] > 0), axis=1), ind]
        except:
            min_dist[ind] = min_dist_sort[np.amin(np.where(min_dist_sort > 0), axis=1)]


    # for random restarts during batch processing. We need to initialize at every
    # batch. Lock the seed here.
    seed= np.random.randint(low=1 ,high=2**31)
    np.random.seed(seed)

    # Inverse of lengthscales should be drawn from truncated Gaussian |N(0, max_dist^2)|
    # dim: Q x D
    # mixture_scales = tf.multiply(,tf.cast(max_dist,dtype=tf.float32)**(-1)

    mixture_scales = (np.multiply(np.abs(np.random.randn(num_mixtures,input_dim)),
                                         np.expand_dims(max_dist ,axis=0)))**(-1)

    # Draw means from Unif(0, 0.5 / minimum distance between two points), dim: Q x D
    # the nyquist is half of maximum frequency. TODO
    nyquist = np.divide(0.5,min_dist)
    mixture_means = np.multiply(np.random.rand(num_mixtures,input_dim),\
                                                       np.expand_dims(nyquist,0))
    mixture_means[0,:] = 0

    # Mixture weights should be roughly the std  of the y values divided by
    # the number of mixtures
    # dim: 1 x Q
    mixture_weights= np.divide(np.std(train_y,axis=0),num_mixtures)*np.ones(num_mixtures)

    return mixture_weights, mixture_means, mixture_scales.T
    
    
def initialize_from_data(train_x, train_y, **kwargs):
    if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
        raise RuntimeError("train_x and train_y should be tensors")
    if train_x.ndimension() == 1:
        train_x = train_x.unsqueeze(-1)
    if train_x.ndimension() == 2:
        train_x = train_x.unsqueeze(0)

    train_x_sort = train_x.sort(1)[0]
    max_dist = train_x_sort[:, -1, :] - train_x_sort[:, 0, :]
    min_dist_sort = (train_x_sort[:, 1:, :] - train_x_sort[:, :-1, :]).squeeze(0)
    min_dist = torch.zeros(1, self.n_dims)
    for ind in range(self.n_dims):
        min_dist[:, ind] = min_dist_sort[(torch.nonzero(min_dist_sort[:, ind]))[0], ind]

    # Inverse of lengthscales should be drawn from truncated Gaussian | N(0, max_dist^2) |
    self.log_mixture_scales.data.normal_().mul_(max_dist).abs_().pow_(-1).log_()
    # Draw means from Unif(0, 0.5 / minimum distance between two points)
    self.log_mixture_means.data.uniform_().mul_(0.5).div_(min_dist).log_()
    # Mixture weights should be roughly the stdv of the y values divided by the number of mixtures
    self.log_mixture_weights.data.fill_(train_y.std() / self.n_mixtures).log_()

