# Copyright 2022 CircuitNet. All rights reserved.

import models
import torch

def build_model(opt):
    model_type = opt.pop('model_type')

    # ✅ Only allow model-specific keys
    allowed_keys = ['in_channels', 'out_channels']

    model_args = {k: v for k, v in opt.items() if k in allowed_keys}

    model = models.__dict__[model_type](**model_args)
    model.init_weights(**model_args)

    if opt.get('test_mode', False):
        model.eval()

    return model