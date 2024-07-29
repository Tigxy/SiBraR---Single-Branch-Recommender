from torch import nn
from typing import Union
from collections import OrderedDict

ACTIVATION_FN_MAP = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
}


def get_activation_fn(activation_fn: str | nn.Module):
    return ACTIVATION_FN_MAP[activation_fn] if isinstance(activation_fn, str) else activation_fn


class PolyLinear(nn.Module):
    def __init__(self, layer_config: list, activation_fn: str | nn.Module = nn.ReLU(),
                 output_fn: str | nn.Module | None = nn.ReLU(), input_dropout=None, l1_weight_decay=None,
                 apply_batch_norm_every: int = 0):
        """
        Helper module to easily create multiple linear layers and pass an
        activation through them
        :param layer_config: A list containing the in_features and out_features for the linear layers
                             Example: [100,50,2] would create two linear layers: Linear(100, 50) and Linear(50, 2),
                             whereas the output of the first layer is used as input for the second layer
        :param activation_fn: The activation function to use between layers
        :param output_fn: (optional) The function to apply on the output, e.g. softmax or any other activation fn
        :param input_dropout: A possible dropout to apply to the input before passing it through the layers
        :param l1_weight_decay: Additional L1 weight normalization to induce sparsity in the layers
        :param apply_batch_norm_every: When to apply batch normalization
                                       0 ... deactivate, 1+ ... every n layers, -1 ... only last layer
        """
        super().__init__()

        assert len(layer_config) > 1, "For a linear network, we at least need one " \
                                      "input and one output dimension"

        self.layer_config = layer_config
        self.activation_fn = get_activation_fn(activation_fn)
        self.output_fn = get_activation_fn(output_fn) if output_fn is not None else output_fn

        self.n_layers = len(layer_config) - 1

        layer_dict = OrderedDict()

        if input_dropout is not None:
            layer_dict["input_dropout"] = nn.Dropout(p=input_dropout)

        for i, (d1, d2) in enumerate(zip(layer_config[:-1], layer_config[1:])):
            layer = nn.Linear(in_features=d1, out_features=d2)
            if l1_weight_decay and l1_weight_decay > 0.0:
                from torchlayers.regularization import L1
                layer = L1(layer, weight_decay=l1_weight_decay)

            layer_dict[f"linear_{i}"] = layer

            # apply batch normalization before activation fn (see http://torch.ch/blog/2016/02/04/resnets.html)
            # +1 to not immediately put normalization after first layer if 'apply_batch_norm_every' >= 2
            if apply_batch_norm_every > 0 and (i + 1) % apply_batch_norm_every == 0:
                layer_dict[f"batch_norm_{i}"] = nn.BatchNorm1d(num_features=d2)

            if i < self.n_layers - 1:
                # only add activation functions in intermediate layers
                layer_dict[f"{self.activation_fn.__class__.__name__.lower()}_{i}"] = self.activation_fn

        if apply_batch_norm_every == -1:
            layer_dict[f"batch_norm"] = nn.BatchNorm1d(num_features=layer_config[-1])

        if self.output_fn is not None:
            layer_dict[f"{self.output_fn.__class__.__name__.lower()}"] = self.output_fn

        self.layers = nn.Sequential(layer_dict)

    def forward(self, x):
        x = self.layers(x)
        return x
