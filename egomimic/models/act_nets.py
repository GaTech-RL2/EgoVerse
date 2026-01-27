import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Callable, Any
from torch.distributions import Normal
import math
from torchvision import models as vision_models
import abc


class PositionalEncoding(nn.Module):
    """
    Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    def __init__(self, embed_dim):
        """
        Standard sinusoidal positional encoding scheme in transformers.

        Positional encoding of the k'th position in the sequence is given by:
            p(k, 2i) = sin(k/n^(i/d))
            p(k, 2i+1) = sin(k/n^(i/d))

        n: set to 10K in original Transformer paper
        d: the embedding dimension
        i: positions along the projected embedding space (ranges from 0 to d/2)

        Args:
            embed_dim: The number of dimensions to project the timesteps into.
        """
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Input timestep of shape BxT
        """
        position = x

        # computing 1/n^(i/d) in log space and then exponentiating and fixing the shape
        div_term = (
            torch.exp(
                torch.arange(0, self.embed_dim, 2, device=x.device)
                * (-math.log(10000.0) / self.embed_dim)
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(x.shape[0], x.shape[1], 1)
        )
        pe = torch.zeros((x.shape[0], x.shape[1], self.embed_dim), device=x.device)
        pe[:, :, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)
        return pe.detach()


"""
================================================
Visual Backbone Networks
================================================
"""


class Module(torch.nn.Module):
    """
    Base class for networks. The only difference from torch.nn.Module is that it
    requires implementing @output_shape.
    """

    @abc.abstractmethod
    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError


class ConvBase(Module):
    """
    Base class for ConvNets.
    """

    def __init__(self):
        super(ConvBase, self).__init__()

    # dirty hack - re-implement to pass the buck onto subclasses from ABC parent
    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    def forward(self, inputs):
        x = self.nets(inputs)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError(
                "Size mismatch: expect size %s, but got size %s"
                % (
                    str(self.output_shape(list(inputs.shape)[1:])),
                    str(list(x.shape)[1:]),
                )
            )
        return x


class CoordConv2d(nn.Conv2d, Module):
    """
    2D Coordinate Convolution

    Source: An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    https://arxiv.org/abs/1807.03247
    (e.g. adds 2 channels per input feature map corresponding to (x, y) location on map)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        coord_encoding="position",
    ):
        """
        Args:
            in_channels: number of channels of the input tensor [C, H, W]
            out_channels: number of output channels of the layer
            kernel_size: convolution kernel size
            stride: conv stride
            padding: conv padding
            dilation: conv dilation
            groups: conv groups
            bias: conv bias
            padding_mode: conv padding mode
            coord_encoding: type of coordinate encoding. currently only 'position' is implemented
        """

        assert coord_encoding in ["position"]
        self.coord_encoding = coord_encoding
        if coord_encoding == "position":
            in_channels += 2  # two extra channel for positional encoding
            self._position_enc = None  # position encoding
        else:
            raise Exception(
                "CoordConv2d: coord encoding {} not implemented".format(
                    self.coord_encoding
                )
            )
        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # adds 2 to channel dimension
        return [input_shape[0] + 2] + input_shape[1:]

    def forward(self, input):
        b, c, h, w = input.shape
        if self.coord_encoding == "position":
            if self._position_enc is None:
                pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
                pos_y = pos_y.float().to(input.device) / float(h)
                pos_x = pos_x.float().to(input.device) / float(w)
                self._position_enc = torch.stack((pos_y, pos_x)).unsqueeze(0)
            pos_enc = self._position_enc.expand(b, -1, -1, -1)
            input = torch.cat((input, pos_enc), dim=1)
        return super(CoordConv2d, self).forward(input)


class ResNet18Conv(ConvBase):
    """
    A ResNet18 block that can be used to process input images.
    """

    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super(ResNet18Conv, self).__init__()
        if pretrained:
            net = vision_models.resnet18(weights="IMAGENET1K_V1")
        else:
            net = vision_models.resnet18(weights=None)

        if input_coord_conv:
            net.conv1 = CoordConv2d(
                input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        elif input_channel != 3:
            net.conv1 = nn.Conv2d(
                input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # cut the last fc layer
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert len(input_shape) == 3
        out_h = int(math.ceil(input_shape[1] / 32.0))
        out_w = int(math.ceil(input_shape[2] / 32.0))
        return [512, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        return header + "(input_channel={}, input_coord_conv={})".format(
            self._input_channel, self._input_coord_conv
        )


class Transformer(nn.Module):
    """
    Basic transformer implementation using torch.nn. Also added option for custom pos embeddings.
    Made to be as basic as possible but also flexible to be put into ACT.

        d: hidden dimension
        h: number of heads
        d_ff: feed forward dimension
        num_layers: number of layers for encoder and decoder
        L: sequence length
        dropout: dropout rate
        src_vocab_size: size of source vocabulary
        tgt_vocab_size: size of target vocabulary
        pos_encoding_class : nn.Module class defining custom pos encoding

    """

    def __init__(
        self,
        d: int,
        h: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        pos_encoding_class: Optional[Callable[..., nn.Module]] = None,
        **pos_encoding_kwargs: Any,  # Additional arguments for the custom encoding class
    ):
        super(Transformer, self).__init__()

        self.d = d
        self.h = h
        self.src_embed = nn.Embedding(src_vocab_size, d) if src_vocab_size else None
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d) if tgt_vocab_size else None

        if pos_encoding_class is not None:
            self.src_pos_encoding = pos_encoding_class(**pos_encoding_kwargs)
            self.tgt_pos_encoding = pos_encoding_class(**pos_encoding_kwargs)
        else:
            self.src_pos_encoding = PositionalEncoding(d)
            self.tgt_pos_encoding = PositionalEncoding(d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=h, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=h, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d, tgt_vocab_size) if tgt_vocab_size else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, auto_masks=False):
        assert auto_masks == False, "Auto mask not supported"
        src_mask = tgt_mask = None

        if self.src_embed:
            src = self.src_embed(src)
        if self.tgt_embed:
            tgt = self.tgt_embed(tgt)

        src = src.transpose(0, 1)  # [sequence_length, batch_size, hidden_dim]
        src_position_indices = (
            torch.arange(src.size(0), device=src.device)
            .unsqueeze(1)
            .expand(-1, src.size(1))
        )  # [sequence_length, batch_size]
        src_pos = self.src_pos_encoding(
            src_position_indices
        )  # [sequence_length, batch_size, hidden_dim]
        src = src + src_pos
        src = src.transpose(0, 1)

        tgt = tgt.transpose(0, 1)  # [T, B, hidden_dim]
        tgt_position_indices = (
            torch.arange(tgt.size(0), device=tgt.device)
            .unsqueeze(1)
            .expand(-1, tgt.size(1))
        )  # [sequence_length, batch_size]
        tgt_pos = self.tgt_pos_encoding(tgt_position_indices)
        tgt = tgt + tgt_pos
        tgt = tgt.transpose(0, 1)  # [B, T, hidden_dim]

        src = self.dropout(src)
        tgt = self.dropout(tgt)

        enc = self.encoder(src, src_key_padding_mask=src_mask)
        dec = self.decoder(
            tgt, enc, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask
        )

        if self.fc:
            dec = self.fc(dec)

        return dec


class StyleEncoder(nn.Module):
    def __init__(
        self,
        act_len: int,
        hidden_dim: int,
        latent_dim: int,
        h: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super(StyleEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.act_len = act_len
        self.hidden_dim = hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=h, dim_feedforward=d_ff, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_embedding = nn.Parameter(torch.rand(1, hidden_dim))
        self.latent_projection = nn.Linear(hidden_dim, latent_dim * 2)

        self.pos_encoding = PositionalEncoding(hidden_dim)

        # self.encoder_norm = nn.LayerNorm(hidden_dim)

    def forward(self, qpos, actions):
        """
        qpos: linear projection of qpos
        actions: linear projection of actions
        """
        bsz = qpos.shape[0]
        qpos = qpos.unsqueeze(1)  # [bsz, 1, hidden_dim]

        cls = self.cls_embedding.unsqueeze(0).expand(
            bsz, -1, -1
        )  # [bsz, 1, hidden_dim]

        x = torch.cat([cls, qpos, actions], dim=1)  # [bsz, act_len + 2, hidden_dim]
        assert x.shape == (bsz, self.act_len + 2, self.hidden_dim)

        pos_indices = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(bsz, -1)
        )  # [bsz, act_len + 2]
        pos_embedded = self.pos_encoding(pos_indices)  # [bsz, act_len + 2, hidden_dim]

        x = x + pos_embedded

        x = x.transpose(0, 1)  # [act_len + 2, bsz, hidden_dim]

        x = self.encoder(x)  # [act_len + 2, bsz, hidden_dim]

        # x = self.encoder_norm(x) # cuz act has weird pre and post norms

        x = x[0]  # [bsz, hidden_dim]

        x = self.latent_projection(x)  # [bsz, latent_dim * 2]
        mu = x[:, self.latent_dim :]
        logvar = x[:, : self.latent_dim]
        dist = Normal(mu, (logvar / 2).exp())  # Create Normal distribution

        return dist
