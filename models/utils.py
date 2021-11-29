from torch import nn


def load_model_dict(model, pretrained_dict):
    # 1. filter out unnecessary keys
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def build_norm_layer(cfg, num_features, postfix=""):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    norm_cfg = {
        # format: layer_type: (abbreviation, module)
        "BN": ("bn", nn.BatchNorm2d),
        "BN1d": ("bn1d", nn.BatchNorm1d),
        "GN": ("gn", nn.GroupNorm)
    }

    layer_type = cfg_.pop("type")   # 'BN1d'
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        # if layer_type == 'SyncBN':
        #     layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer