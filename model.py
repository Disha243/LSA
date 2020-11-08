import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from functools import reduce
from operator import mul
import torch.distributions as dist
import random

class BaseModule(nn.Module):
    """
    Implements the basic module.
    All other modules inherit from this one
    """
    def load_w(self, checkpoint_path):
        # type: (str) -> None
        """
        Loads a checkpoint into the state_dict.

        :param checkpoint_path: the checkpoint file to be loaded.
        """
        self.load_state_dict(torch.load(checkpoint_path))

    def __repr__(self):
        # type: () -> str
        """
        String representation
        """
        good_old = super(BaseModule, self).__repr__()
        addition = 'Total number of parameters: {:,}'.format(self.n_parameters)

        return good_old + '\n' + addition

    def __call__(self, *args, **kwargs):
        return super(BaseModule, self).__call__(*args, **kwargs)

    @property
    def n_parameters(self):
        # type: () -> int
        """
        Number of parameters of the model.
        """
        n_parameters = 0
        for p in self.parameters():
            if hasattr(p, 'mask'):
                n_parameters += torch.sum(p.mask).item()
            else:
                n_parameters += reduce(mul, p.shape)
        return int(n_parameters)

def residual_op(x, functions, bns, activation_fn):
    # type: (torch.Tensor, List[Module, Module, Module], List[Module, Module, Module], Module) -> torch.Tensor
    """
    Implements a global residual operation.

    :param x: the input tensor.
    :param functions: a list of functions (nn.Modules).
    :param bns: a list of optional batch-norm layers.
    :param activation_fn: the activation to be applied.
    :return: the output of the residual operation.
    """
    f1, f2, f3 = functions
    bn1, bn2, bn3 = bns

    assert len(functions) == len(bns) == 3
    assert f1 is not None and f2 is not None
    assert not (f3 is None and bn3 is not None)

    # A-branch
    ha = x
    ha = f1(ha)
    if bn1 is not None:
        ha = bn1(ha)
    ha = activation_fn(ha)

    ha = f2(ha)
    if bn2 is not None:
        ha = bn2(ha)

    # B-branch
    hb = x
    if f3 is not None:
        hb = f3(hb)
    if bn3 is not None:
        hb = bn3(hb)

    # Residual connection
    out = ha + hb
    return activation_fn(out)


class BaseBlock(BaseModule):
    """ Base class for all blocks. """
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.

        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(BaseBlock, self).__init__()

        assert not (use_bn and use_bias), 'Using bias=True with batch_normalization is forbidden.'

        self._channel_in = channel_in
        self._channel_out = channel_out
        self._activation_fn = activation_fn
        self._use_bn = use_bn
        self._bias = use_bias

    def get_bn(self):
        # type: () -> Optional[Module]
        """
        Returns batch norm layers, if needed.
        :return: batch norm layers or None
        """
        return nn.BatchNorm2d(num_features=self._channel_out) if self._use_bn else None

    def forward(self, x):
        """
        Abstract forward function. Not implemented.
        """
        raise NotImplementedError


class DownsampleBlock(BaseBlock):
    """ Implements a Downsampling block for images (Fig. 1ii). """
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.

        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(DownsampleBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        # Convolutions
        self.conv1a = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                                padding=1, stride=2, bias=use_bias)
        self.conv1b = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                padding=1, stride=1, bias=use_bias)
        self.conv2a = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1,
                                padding=0, stride=2, bias=use_bias)

        # Batch Normalization layers
        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn
        )


class UpsampleBlock(BaseBlock):
    """ Implements a Upsampling block for images (Fig. 1ii). """
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.

        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(UpsampleBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        # Convolutions
        self.conv1a = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5,
                                         padding=2, stride=2, output_padding=1, bias=use_bias)
        self.conv1b = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                padding=1, stride=1, bias=use_bias)
        self.conv2a = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=1,
                                         padding=0, stride=2, output_padding=1, bias=use_bias)

        # Batch Normalization layers
        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn
        )


class ResidualBlock(BaseBlock):
    """ Implements a Residual block for images (Fig. 1ii). """
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.

        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(ResidualBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        # Convolutions
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                               padding=1, stride=1, bias=use_bias)
        self.conv2 = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                               padding=1, stride=1, bias=use_bias)

        # Batch Normalization layers
        self.bn1 = self.get_bn()
        self.bn2 = self.get_bn()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op(
            x,
            functions=[self.conv1, self.conv2, None],
            bns=[self.bn1, self.bn2, None],
            activation_fn=self._activation_fn
        )

class MaskedFullyConnection(BaseModule, nn.Linear):
    """
    Implements a Masked Fully Connection layer (MFC, Eq. 6).
    This is the autoregressive layer employed for the estimation of
    densities of image feature vectors.
    """
    def __init__(self, mask_type, in_channels, out_channels, *args, **kwargs):
        """
        Class constructor.

        :param mask_type: type of autoregressive layer, either `A` or `B`.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        """
        self.mask_type = mask_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(MaskedFullyConnection, self).__init__(*args, **kwargs)

        assert mask_type in ['A', 'B']
        self.register_buffer('mask', self.weight.data.clone())

        # Build mask
        self.mask.fill_(0)
        for f in range(0 if mask_type == 'B' else 1, self.out_features // self.out_channels):
            start_row = f*self.out_channels
            end_row = (f+1)*self.out_channels
            start_col = 0
            end_col = f*self.in_channels if mask_type == 'A' else (f+1)*self.in_channels
            if start_col != end_col:
                self.mask[start_row:end_row, start_col:end_col] = 1

        self.weight.mask = self.mask

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input tensor.
        :return: the output of a MFC manipulation.
        """

        # Reshape
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(len(x), -1)

        # Mask weights and call fully connection
        self.weight.data *= self.mask
        o = super(MaskedFullyConnection, self).forward(x)

        # Reshape again
        o = o.view(len(o), -1, self.out_channels)
        o = torch.transpose(o, 1, 2).contiguous()

        return o

    def __repr__(self):
        # type: () -> str
        """
        String representation.
        """
        return self.__class__.__name__ + '(' \
               + 'mask_type=' + str(self.mask_type) \
               + ', in_features=' + str(self.in_features // self.in_channels) \
               + ', out_features=' + str(self.out_features // self.out_channels)\
               + ', in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', n_params=' + str(self.n_parameters) + ')'

class Estimator1D(BaseModule):
    """
    Implements an estimator for 1-dimensional vectors.
    1-dimensional vectors arise from the encoding of images.
    This module is employed in MNIST and CIFAR10 LSA models.
    Takes as input a latent vector and outputs cpds for each variable.
    """
    def __init__(self, code_length, fm_list, cpd_channels):
        # type: (int, List[int], int) -> None
        """
        Class constructor.

        :param code_length: the dimensionality of latent vectors.
        :param fm_list: list of channels for each MFC layer.
        :param cpd_channels: number of bins in which the multinomial works.
        """
        super(Estimator1D, self).__init__()

        self.code_length = code_length
        self.fm_list = fm_list
        self.cpd_channels = cpd_channels

        activation_fn = nn.LeakyReLU()

        # Add autoregressive layers
        layers_list = []
        mask_type = 'A'
        fm_in = 1
        for l in range(0, len(fm_list)):

            fm_out = fm_list[l]
            layers_list.append(
                MaskedFullyConnection(mask_type=mask_type,
                                      in_features=fm_in * code_length,
                                      out_features=fm_out * code_length,
                                      in_channels=fm_in, out_channels=fm_out)
            )
            layers_list.append(activation_fn)

            mask_type = 'B'
            fm_in = fm_list[l]

        # Add final layer providing cpd params
        layers_list.append(
            MaskedFullyConnection(mask_type=mask_type,
                                  in_features=fm_in * code_length,
                                  out_features=cpd_channels * code_length,
                                  in_channels=fm_in,
                                  out_channels=cpd_channels))

        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of CPD estimates.
        """
        h = torch.unsqueeze(x, dim=1)  # add singleton channel dim
        h = self.layers(h)
        o = h

        return o

def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single vlaue or a list/tuple of len 2")
        return n_val
    else:
        return value


def get_square_mask(data_shape, square_size, n_squares, noise_val=(0, 0), channel_wise_n_val=False, square_pos=None):
    """Returns a 'mask' with the same size as the data, where random squares are != 0

    Args:
        data_shape ([tensor]): [data_shape to determine the shape of the returned tensor]
        square_size ([tuple]): [int/ int tuple (min_size, max_size), determining the min and max squear size]
        n_squares ([type]): [int/ int tuple (min_number, max_number), determining the min and max number of squares]
        noise_val (tuple, optional): [int/ int tuple (min_val, max_val), determining the min and max value given in the 
                                        squares, which habe the value != 0 ]. Defaults to (0, 0).
        channel_wise_n_val (bool, optional): [Use a different value for each channel]. Defaults to False.
        square_pos ([type], optional): [Square position]. Defaults to None.
    """

    def mask_random_square(img_shape, square_size, n_val, channel_wise_n_val=False, square_pos=None):
        """Masks (sets = 0) a random square in an image"""

        img_h = img_shape[-2]
        img_w = img_shape[-1]

        img = np.zeros(img_shape)

        if square_pos is None:
            w_start = np.random.randint(0, img_w - square_size)
            h_start = np.random.randint(0, img_h - square_size)
        else:
            pos_wh = square_pos[np.random.randint(0, len(square_pos))]
            w_start = pos_wh[0]
            h_start = pos_wh[1]

        if img.ndim == 2:
            rnd_n_val = get_range_val(n_val)
            img[h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val
        elif img.ndim == 3:
            if channel_wise_n_val:
                for i in range(img.shape[0]):
                    rnd_n_val = get_range_val(n_val)
                    img[i, h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val
            else:
                rnd_n_val = get_range_val(n_val)
                img[:, h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val
        elif img.ndim == 4:
            if channel_wise_n_val:
                for i in range(img.shape[0]):
                    rnd_n_val = get_range_val(n_val)
                    img[:, i, h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val
            else:
                rnd_n_val = get_range_val(n_val)
                img[:, :, h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val

        return img

    def mask_random_squares(img_shape, square_size, n_squares, n_val, channel_wise_n_val=False, square_pos=None):
        """Masks a given number of squares in an image"""
        img = np.zeros(img_shape)
        for i in range(n_squares):
            img = mask_random_square(
                img_shape, square_size, n_val, channel_wise_n_val=channel_wise_n_val, square_pos=square_pos
            )
        return img

    ret_data = np.zeros(data_shape)
    for sample_idx in range(data_shape[0]):
        # rnd_n_val = get_range_val(noise_val)
        rnd_square_size = get_range_val(square_size)
        rnd_n_squares = get_range_val(n_squares)

        ret_data[sample_idx] = mask_random_squares(
            data_shape[1:],
            square_size=rnd_square_size,
            n_squares=rnd_n_squares,
            n_val=noise_val,
            channel_wise_n_val=channel_wise_n_val,
            square_pos=square_pos,
        )

    return ret_data

def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single vlaue or a list/tuple of len 2")
        return n_val
    else:
        return value


class Encoder(BaseModule):
    """
    MOOD model encoder based on CIFAR10.
    """
    def __init__(self, input_shape, code_length, n_starting_features=32, conv=nn.Conv2d):
        # type: (Tuple[int, int, int], int) -> None
        """
        Class constructor:

        :param input_shape: the shape of CIFAR10 samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        if len(input_shape) == 3:
          c, h, w = input_shape
          self.deepest_shape = (n_starting_features*8, h // 8, w // 8)
          

        activation_fn = nn.LeakyReLU()

        # Convolutional network
        self.conv = nn.Sequential(
            conv(in_channels=c, out_channels=n_starting_features, kernel_size=3, bias=False),
            activation_fn,
            ResidualBlock(channel_in=n_starting_features, channel_out=n_starting_features, activation_fn=activation_fn),
            DownsampleBlock(channel_in=n_starting_features, channel_out=n_starting_features*2, activation_fn=activation_fn),
            DownsampleBlock(channel_in=n_starting_features*2, channel_out=n_starting_features*4, activation_fn=activation_fn),
            DownsampleBlock(channel_in=n_starting_features*4, channel_out=n_starting_features*8, activation_fn=activation_fn),
        )

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=n_starting_features*8),
            nn.BatchNorm1d(num_features=n_starting_features*8),
            activation_fn,
            nn.Linear(in_features=n_starting_features*8, out_features=code_length),
            nn.Sigmoid()
        )
    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: the batch of latent vectors.
        """
        h = x
        h = self.conv(h)
        h = h.view(len(h), -1)
        o = self.fc(h)

        return o


class Decoder(BaseModule):
    """
    CIFAR10 model decoder.
    """
    def __init__(self, code_length, deepest_shape, output_shape, n_starting_features=32, n_channels=3, conv=nn.Conv3d):
        # type: (int, Tuple[int, int, int], Tuple[int, int, int]) -> None
        """
        Class constructor.

        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of CIFAR10 samples.
        """
        super(Decoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        activation_fn = nn.LeakyReLU()
        


        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=code_length, out_features=n_starting_features*8),
            nn.BatchNorm1d(num_features=n_starting_features*8),
            activation_fn,
            nn.Linear(in_features=n_starting_features*8, out_features=reduce(mul, deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, deepest_shape)),
            activation_fn
        )

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=n_starting_features*8, channel_out=n_starting_features*4, activation_fn=activation_fn),
            UpsampleBlock(channel_in=n_starting_features*4, channel_out=n_starting_features*2, activation_fn=activation_fn),
            UpsampleBlock(channel_in=n_starting_features*2, channel_out=n_starting_features, activation_fn=activation_fn),
            ResidualBlock(channel_in=n_starting_features, channel_out=n_starting_features, activation_fn=activation_fn),
            conv(in_channels=n_starting_features, out_channels=n_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x
        h = self.fc(h)
        h = h.view(len(h), *self.deepest_shape)
        h = self.conv(h)
        o = h

        return o


class LSAMOOD(BaseModule):
    """
    LSA model for CIFAR10 one-class classification.
    """
    def __init__(self,  input_shape, code_length, cpd_channels, d=3, n_starting_features=32, n_channels=3, out_sigmoid=True, vae_mode=False):
        # type: (Tuple[int, int, int], int, int) -> None
        """
        Class constructor.

        :param input_shape: the shape of CIFAR10 samples.
        :param code_length: the dimensionality of latent vectors.
        :param cpd_channels: number of bins in which the multinomial works.
        """
        super(LSAMOOD, self).__init__()

        if type(input_shape) is tuple:
            d=len(input_shape)-1

        self.input_shape = input_shape
        self.code_length = code_length
        self.out_sigmoid = out_sigmoid
        self.vae_mode = vae_mode
        
        self.d = d
        conv = nn.Conv2d
        

        # Build encoder
        self.encoder = Encoder(
            input_shape=input_shape,
            code_length=code_length*2 if vae_mode else code_length,
            n_starting_features=n_starting_features,
            conv=conv
        )

        # Build decoder
        self.decoder = Decoder(
            code_length=code_length,
            deepest_shape=self.encoder.deepest_shape,
            output_shape=input_shape,
            n_starting_features=n_starting_features,
            n_channels=n_channels,
            conv=conv
        )

        if not vae_mode:
            # Build estimator
            self.estimator = Estimator1D(
                code_length=code_length,
                fm_list=[n_starting_features, n_starting_features, n_starting_features, n_starting_features],
                cpd_channels=cpd_channels
            )

    def forward(self, x, reparam=True):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: a tuple of torch.Tensors holding reconstructions, latent vectors and CPD estimates.
        """
        h = x

        # Produce representations
        z = self.encoder(h)

        if self.vae_mode:
            mu, log_std = torch.chunk(z.contiguous().view(x.size(0), -1), 2, dim=1)
            std = torch.exp(log_std)
            z_dist = dist.Normal(mu, std)
            if reparam:
                z = z_dist.rsample()
            else:
                z = mu      
        else:
            # Estimate CPDs with autoregression
            z_dist = self.estimator(z)


        # Reconstruct x
        x_r = self.decoder(z)
        if self.out_sigmoid:
            x_r = nn.functional.sigmoid(x_r)
        x_r = x_r.view(-1, *self.input_shape)

        return x_r, z, z_dist

if __name__ == "__main__":
    model = LSAMOOD(input_shape=(1,256,256), code_length=64, cpd_channels=100, n_starting_features=4)

