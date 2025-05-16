"""
Gated Recurrent Unit
"""

import brainunit as u
import jax
import jax.numpy as jnp
from typing import Callable, Union

import brainscale
import brainstate
import brainstate as bst

__all__ = [
    'FiringRateState',
    'SpikeFunction',
    'EGRU',
    'MergeEvents',
    'Network',
]


class FiringRateState(brainstate.ShortTermState):
    pass


class SpikeFunction(bst.surrogate.Surrogate):
    """
    A surrogate gradient function for spiking neural networks.

    This class implements a surrogate gradient function that approximates
    the gradient of a step function (typically used in spiking neural networks)
    to enable backpropagation through discrete spike events.

    Parameters:
    -----------
    dampening_factor : float
        A scaling factor applied to the surrogate gradient to control its magnitude.
    pseudo_derivative_support : float
        Determines the width of the surrogate gradient function around zero.

    """

    def __init__(
        self,
        dampening_factor: float,
        pseudo_derivative_support: float,
    ):
        super().__init__()
        self.dampening_factor = dampening_factor
        self.pseudo_derivative_support = pseudo_derivative_support

    def surrogate_grad(self, x) -> jax.Array:
        """
        Compute the surrogate gradient for the input.

        This method calculates a piece-wise linear approximation of the gradient
        of a step function, which is used during backpropagation in spiking neural networks.

        Parameters:
        -----------
        x : jax.Array
            The input array for which to compute the surrogate gradient.

        Returns:
        --------
        jax.Array
            The computed surrogate gradient.
        """
        dz_du = (
            self.dampening_factor *
            jnp.maximum(1 - self.pseudo_derivative_support * jnp.abs(x), 0.)
        )
        return dz_du


class EGRU(bst.nn.Module):
    r"""
    Event based Gated Recurrent Unit layer [1]_.

    Code adapted from: https://github.com/Efficient-Scalable-Machine-Learning/EvNN

    $$
    \begin{aligned}
    \mathbf{u}^{\langle t\rangle} &= \sigma\left(\mathbf{W}_u\left[\mathbf{x}^{\langle t\rangle}, \mathbf{y}^{\langle t-1\rangle}\right]+\mathbf{b}_u\right), \\
    \mathbf{r}^{\langle t\rangle} &= \sigma\left(\mathbf{W}_r\left[\mathbf{x}^{\langle t\rangle}, \mathbf{y}^{\langle t-1\rangle}\right]+\mathbf{b}_r\right), \\
    \mathbf{z}^{\langle t\rangle} &= g\left(\mathbf{W}_z\left[\mathbf{x}^{\langle t\rangle}, \mathbf{r}^{\langle t\rangle} \odot \mathbf{y}^{\langle t-1\rangle}\right]+\mathbf{b}_z\right), \\
    \mathbf{y}^{\langle t\rangle} &= \mathbf{u}^{\langle t\rangle} \odot \mathbf{z}^{\langle t\rangle}+\left(1-\mathbf{u}^{\langle t\rangle}\right) \odot \mathbf{y}^{\langle t-1\rangle},
    \end{aligned}
    $$

    where $\mathbf{W}_{u / r / z}, \mathbf{b}_{u / r / z}$ denote network weights and biases,
    $\odot$ denotes the element-wise (Hadamard) product, and $\sigma(\cdot)$ is the vectorized sigmoid function.
    The notation $\left[\mathbf{x}^{\langle t\rangle}, \mathbf{y}^{\langle t-1\rangle}\right]$ denotes vector
    concatenation. The function $g(\cdot)$ is an element-wise nonlinearity, typically the hyperbolic tangent.

    We introduce an event generating mechanisms by augmenting the GRU with a rectifier (a thresholding function).
    With this addition the internal state variable $y_i^{\langle t\rangle}$ is nonzero only when the internal
    dynamics reach a threshold $\vartheta_i$ and is cleared immediately afterwards, thus making
    $y_i^{\langle t\rangle}$ event-based. Formally, we add an auxiliary internal state $c_i^{\langle t\rangle}$
    to the model, and replace
    $\mathbf{y}^{\langle t\rangle}=\left(y_1^{\langle t\rangle}, y_2^{\langle t\rangle}, \ldots\right)$
    with the event-based form

    $$
    \begin{aligned}
    y_i^{\langle t\rangle} &=c_i^{\langle t\rangle} H\left(c_i^{\langle t\rangle}-\vartheta_i\right) \\
    c_i^{\langle t\rangle} &=u_i^{\langle t\rangle} z_i^{\langle t\rangle}+\left(1-u_i^{\langle t\rangle}\right) c_i^{\langle t-1\rangle}-y_i^{\langle t-1\rangle},
    \end{aligned}
    $$

    where $H(\cdot)$ is the Heaviside step function and $\vartheta_i>0$ is a trainable threshold parameter.
    $H(\cdot)$ is the threshold gating mechanism here, generating a single non-zero output when
    $c_i^{\langle t\rangle}$ crosses the threshold $\vartheta_i$. That is, at all time steps $t$
    with $c_i^{\langle t\rangle}\left\langle\vartheta_i, \forall i\right.$, we have $y_i^{\langle t\rangle}=0$.
    The $-y_i^{\langle t-1\rangle}$ term makes emission of multiple consecutive events by the
    same unit unlikely, hence favoring overall sparse activity. With this formulation, each unit only needs
    to be updated when an input is received either externally or from another unit in the network.
    This is because, if both $x_i^{\langle t\rangle}=y_i^{\langle t-1\rangle}=0$ for the $i$-th unit,
    then $u_i^{\langle t\rangle}, r_i^{\langle t\rangle}, z_i^{\langle t\rangle}$ are essentially constants,
    and hence the update for $y_i^{\langle t\rangle}$ can be retroactively calculated efficiently on the
    next incoming event.


    References
    ----------
    .. [1] Subramoney A, Nazeer K K, Schöne M, et al. Efficient recurrent architectures through activity sparsity and sparse back-propagation through time[J]. arXiv preprint arXiv:2206.06178, 2022.

    """

    def __init__(
        self,
        input_size: bst.typing.Size,
        hidden_size: bst.typing.Size,
        dropout_prob: float = 0.0,
        zoneout_prob: float = 0.0,
        thr_mean: float = 0.3,
        w_init: Union[bst.typing.ArrayLike, Callable] = bst.init.XavierNormal(),
        b_init: Union[bst.typing.ArrayLike, Callable] = bst.init.ZeroInit(),
        state_init: Union[bst.typing.ArrayLike, Callable] = bst.init.ZeroInit(),
        param_type: Callable = brainscale.ETraceParam,
        spk_fun: Callable = SpikeFunction(dampening_factor=0.7, pseudo_derivative_support=1.0),
    ):
        """
        Initialize the parameters of the GRU layer.

        Arguments:
          input_size: int, the feature dimension of the input.
          hidden_size: int, the feature dimension of the output.
          dropout_prob: (optional) float, sets the dropout rate for DropConnect
            regularization on the recurrent matrix.
          zoneout_prob: (optional) float, sets the zoneout rate for Zoneout
            regularization.
          grad_clip: (optional) float, sets the gradient clipping value.

        """
        super().__init__()

        self.in_size = input_size
        self.out_size = hidden_size

        assert callable(spk_fun), 'GRU: spike_fun must be a callable'
        self.spk_fun = spk_fun

        assert 0 <= dropout_prob <= 1, 'GRU: dropout_prob must be in [0.0, 1.0]'
        assert 0 <= zoneout_prob <= 1, 'GRU: zoneout_prob must be in [0.0, 1.0]'
        self.zoneout_prob = zoneout_prob
        self.dropout_prob = dropout_prob
        self.zoneout = bst.nn.Dropout(1. - self.zoneout_prob)
        self.dropout = bst.nn.Dropout(1. - self.dropout_prob)

        self.alpha = 0.9
        self.state_init = state_init

        # parameters
        params = dict(w_init=w_init, b_init=b_init, param_type=param_type)
        self.Wz = brainscale.nn.Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.Wr = brainscale.nn.Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.Whx = brainscale.nn.Linear(self.in_size[-1], self.out_size[-1], **params)
        self.Whr = brainscale.nn.Linear(self.out_size[-1], self.out_size[-1], **params)

        # initialize thresholds according to the beta distribution with mean 'thr_mean'
        assert 0 < thr_mean < 1, f"thr_mean must be between 0 and 1, but {thr_mean} was given"
        beta = 3
        alpha = beta * thr_mean / (1 - thr_mean)
        thr = bst.random.beta(alpha, beta, self.out_size[-1])
        self.thr = brainscale.ElemWiseParam(thr)

    def init_state(self, batch_size: int = None, **kwargs):
        # bst.random.random(1)

        # hidden state for all sequences.
        self.h = brainscale.ETraceState(bst.init.param(self.state_init, self.out_size, batch_size))

        # the output gate for all sequences (values: 0 or 1).
        self.o = brainstate.HiddenState(bst.init.param(self.state_init, self.out_size, batch_size))
        # self.o = brainscale.ETraceState(bst.init.param(self.state_init, self.out_size, batch_size))

        # internal state variable
        self.y = brainscale.ETraceState(bst.init.param(self.state_init, self.out_size, batch_size))

        # smoothed output values, can be beneficial for training.
        self.tr = brainscale.ETraceState(bst.init.param(self.state_init, self.out_size, batch_size))

        # # firing rate
        self.fr = FiringRateState(bst.init.param(bst.init.ZeroInit(), self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = bst.init.param(self.state_init, self.out_size, batch_size)
        self.o.value = bst.init.param(self.state_init, self.out_size, batch_size)
        self.y.value = bst.init.param(self.state_init, self.out_size, batch_size)
        self.tr.value = bst.init.param(self.state_init, self.out_size, batch_size)
        self.fr.value = bst.init.param(bst.init.ZeroInit(), self.out_size, batch_size)

    def update(self, x):
        """
        Runs a forward pass of the EGRU layer.

        Arguments:
          inputs: Tensor, a batch of input sequences to pass through the GRU.
            Dimensions (seq_len, batch_size, input_size) if `batch_first` is
            `False`, otherwise (batch_size, seq_len, input_size).

        Returns:
          output: Tensor, the output of the EGRU layer. Dimensions
            (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
            or (batch_size, seq_len, hidden_size) if `batch_first` is `True`. Note
            that if `lengths` was specified, the `output` tensor will not be
            masked. It's the caller's responsibility to either not use the invalid
            entries or to mask them out before using them.
          h: the hidden state for all sequences. Dimensions
            (seq_len, batch_size, hidden_size).
          o: the output gate for all sequences (values: 0 or 1).
          trace: smoothed output values, can be beneficial for training.

        """
        fit = bst.environ.get('fit')

        old_h = self.h.value
        xh = u.math.concatenate([x, old_h], axis=-1)
        z = u.math.sigmoid(self.Wz(xh))
        r = u.math.sigmoid(self.Wr(xh))
        g = u.math.tanh(self.Whx(x) + self.Whr(old_h) * r)
        cur_h = z * old_h + (1 - z) * g
        if self.zoneout_prob > 0.0:
            if fit:
                cur_h = self.zoneout(cur_h - old_h) + old_h
            else:
                cur_h = self.zoneout_prob * cur_h + (1 - self.zoneout_prob) * old_h

        thr = self.thr.execute()
        event = self.spk_fun(cur_h - thr)

        self.o.value = event
        self.h.value = cur_h - event * thr
        self.y.value = event * cur_h
        self.tr.value = self.alpha * self.tr.value + (1 - self.alpha) * self.y.value
        self.fr.value = self.fr.value + event
        return self.tr.value


class MergeEvents(bst.nn.Module):
    """
    A module for merging event data in neural networks.

    This class provides functionality to merge event data by either taking the mean
    across channels or passing the data through unchanged. It can also optionally
    flatten the output.

    Parameters:
    -----------
    in_size : bst.typing.Size
        The size of the input data.
    method : str, optional
        The method used for merging events. Can be either 'mean' or 'none'.
        Default is 'mean'.
    flatten : bool, optional
        Whether to flatten the output data. Default is True.

    Attributes:
    -----------
    method : str
        The chosen method for merging events.
    flatten : bool
        Whether the output should be flattened.
    in_size : bst.typing.Size
        The size of the input data.
    out_size : tuple
        The shape of the output data.
    """

    def __init__(
        self,
        in_size: bst.typing.Size,
        method: str = 'mean',
        flatten: bool = True,
    ):
        super().__init__()
        assert method in ['mean', 'none'], 'Unknown Method'
        self.method = method
        self.flatten = flatten
        self.in_size = in_size
        self.out_size = jax.eval_shape(self.update, jax.ShapeDtypeStruct(self.in_size, jnp.float32)).shape

    def update(self, data):
        """
        Processes the input data according to the specified method and flattening option.

        Parameters:
        -----------
        data : jax.Array
            The input data to be processed. Expected to be a 3D array (H x W x channels).

        Returns:
        --------
        jax.Array
            The processed data. If flatten is True, the output will be a 1D array.
            Otherwise, it will maintain its original dimensions or be reduced to
            H x W x 1 if the method is 'mean'.

        Raises:
        -------
        AssertionError
            If the input data is not 3-dimensional.
        """
        # print('MergeEvents = ', data.shape)
        assert data.ndim == 3, f'Expected 3D input,  H x W x channels, but got {data.shape}'
        if self.method == 'mean':
            data = jnp.mean(data, axis=-1, keepdims=True)
        if self.flatten:
            return data.flatten()
        else:
            return data


class Network(bst.nn.Module):
    """
    A neural network module that combines convolutional and recurrent layers for processing event-based data.

    This network can be configured to use CNN layers, different types of RNN layers, and various regularization techniques.

    Parameters:
    -----------
    input_size : bst.typing.Size
        The size of the input data.
    frame_size : int
        The size of each frame in the input data.
    num_layers : int
        The number of recurrent layers in the network.
    hidden_dim : int
        The dimension of the hidden state in the recurrent layers.
    num_class : int
        The number of output classes.
    zoneout : float, optional
        The zoneout probability for regularization in recurrent layers. Default is 0.0.
    dropconnect : float, optional
        The dropconnect probability for regularization in recurrent layers. Default is 0.0.
    layer_dropout : float, optional
        The dropout probability between layers. Default is 0.0.
    pseudo_derivative_width : float, optional
        The width of the pseudo-derivative for the spike function. Default is 1.7.
    threshold_mean : float, optional
        The mean threshold for the event-based GRU. Default is 0.2465.
    rnn_type : str, optional
        The type of recurrent layer to use. Can be 'lstm', 'gru', or 'event-gru'. Default is 'event-gru'.
    event_agg_method : str, optional
        The method for aggregating events. Can be 'mean' or 'none'. Default is 'mean'.
    use_cnn : bool, optional
        Whether to use convolutional layers before the recurrent layers. Default is True.
    """

    def __init__(
        self,
        input_size: bst.typing.Size,
        frame_size: int,
        n_rnn_layer: int,
        n_rnn_hidden: int,
        n_class: int,
        zoneout: float = 0.0,
        dropconnect: float = 0.0,
        layer_dropout: float = 0.0,
        pseudo_derivative_width: float = 1.7,
        threshold_mean: float = 0.2465,
        rnn_type: str = 'event-gru',
        event_agg_method: str = 'mean',
        use_cnn: bool = True,
    ):
        super().__init__()

        # parameters
        self.input_size = input_size
        self.frame_size = frame_size
        self.num_layers = n_rnn_layer
        self.use_cnn = use_cnn
        self.rnn_type = rnn_type
        self.event_agg_method = event_agg_method
        self.hidden_dim = n_rnn_hidden
        self.dropconnect = dropconnect
        self.zoneout = zoneout
        self.layer_dropout = layer_dropout

        # Max pooling
        self.pool = bst.nn.Sequential(
            brainscale.nn.MaxPool2d(in_size=input_size, kernel_size=128 // frame_size),
            MergeEvents.desc(method=event_agg_method, flatten=not use_cnn),
        )
        # print('pool outsize = ', self.pool.out_size)

        # cnn layers
        if self.use_cnn:
            self.cnn = bst.nn.Sequential(
                brainscale.nn.Conv2d(self.pool.out_size, out_channels=64, kernel_size=11, stride=4, padding=2),
                brainscale.nn.ReLU(),
                brainscale.nn.MaxPool2d.desc(kernel_size=3, stride=2),
                brainscale.nn.Conv2d.desc(out_channels=192, kernel_size=5, padding=2),
                brainscale.nn.ReLU(),
                brainscale.nn.MaxPool2d.desc(kernel_size=3, stride=2),
                brainscale.nn.Conv2d.desc(out_channels=384, kernel_size=3, padding=1),
                brainscale.nn.ReLU(),
                (
                    brainscale.nn.MaxPool2d.desc(kernel_size=3, stride=2)
                    if frame_size >= 64 else
                    brainscale.nn.Identity()
                ),
                brainscale.nn.Conv2d.desc(out_channels=256, kernel_size=3, padding=1),
                brainscale.nn.ReLU(),
                (
                    brainscale.nn.MaxPool2d.desc(kernel_size=3, stride=2)
                    if frame_size >= 128 else
                    brainscale.nn.Identity()
                ),
                brainscale.nn.Conv2d.desc(out_channels=256, kernel_size=3, padding=1),
                brainscale.nn.ReLU(),
                brainscale.nn.Flatten.desc(),
                brainscale.nn.Linear.desc(out_size=512),
                (
                    brainscale.nn.Dropout(1 - self.layer_dropout)
                    if self.layer_dropout > 0. else
                    brainscale.nn.Identity()
                ),
            )
        else:
            self.cnn = brainscale.nn.Flatten()

        # rnn layers
        rnn_layers = []
        input_size = self.cnn.out_size if self.use_cnn else self.pool.out_size
        for layer_idx in range(n_rnn_layer):
            if self.rnn_type == 'lstm':
                rnn = brainscale.nn.LSTMCell(input_size, self.hidden_dim)

            elif self.rnn_type == 'gru':
                rnn = brainscale.nn.GRUCell(input_size, self.hidden_dim)

            elif self.rnn_type == 'event-gru':
                rnn = EGRU(
                    input_size,
                    self.hidden_dim,
                    dropout_prob=dropconnect,
                    zoneout_prob=zoneout,
                    spk_fun=SpikeFunction(
                        dampening_factor=0.7,
                        pseudo_derivative_support=pseudo_derivative_width
                    ),
                    thr_mean=threshold_mean,
                )

            else:
                raise RuntimeError("Unknown lstm type: %s" % self.rnn_type)
            input_size = rnn.out_size
            rnn_layers.append(rnn)
            if self.layer_dropout > 0.:
                rnn_layers.append(brainscale.nn.Dropout(1 - self.layer_dropout))
        self.rnn = bst.nn.Sequential(*rnn_layers)

        # fully connected readout layer
        self.fc = brainscale.nn.Linear(self.hidden_dim, n_class)

    def update(self, x):
        """
        Perform a forward pass through the network.

        Parameters:
        -----------
        x : tensor
            The input tensor to the network.

        Returns:
        --------
        tensor
            The output of the network after passing through all layers.
        """
        x = self.pool(x)
        x = self.cnn(x)
        x = self.rnn(x)
        return self.fc(x)


def f_run():
    gru = EGRU(10, 10)
    gru = bst.nn.EnvironContext(gru, fit=True)

    def loss():
        batch_size = 5
        bst.nn.vmap_init_all_states(gru, axis_size=batch_size, state_tag='hidden')
        print(bst.graph.states(gru, 'hidden'))

        y = bst.nn.Vmap(gru, vmap_states='hidden')(bst.random.randn(batch_size, 10))
        return (y - 1.).sum()

    f_grad = bst.augment.grad(loss, grad_states=gru.states(bst.ParamState))
    print(f_grad)
    print(f_grad())


if __name__ == '__main__':
    f_run()
