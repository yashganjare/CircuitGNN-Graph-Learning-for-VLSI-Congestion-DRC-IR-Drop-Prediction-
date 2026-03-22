# Copyright 2022 CircuitNet. All rights reserved.

from __future__ import print_function

import os
import os.path as osp
import json
import numpy as np

from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.metrics import build_metric
from models.build_model import build_model
from utils.configs import Parser


def test():
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)

    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    arg_dict['ann_file'] = arg_dict['ann_file_test']
    arg_dict['test_mode'] = True

    print('===> Loading datasets')
    dataset = build_dataset(arg_dict)

    print('===> Building model')
    model = build_model(arg_dict)
    model = model.cpu()
    model.eval()

    # ✅ prepare save folder ALWAYS
    save_dir = osp.join(arg_dict['save_path'], 'test_result')
    os.makedirs(save_dir, exist_ok=True)

    metrics = {k: build_metric(k) for k in arg_dict['eval_metric']}
    avg_metrics = {k: 0 for k in arg_dict['eval_metric']}

    with torch.no_grad():
        with tqdm(total=len(dataset)) as bar:
            for idx, (feature, label, label_path) in enumerate(dataset):

                input = feature.cpu()
                target = label.cpu()

                prediction = model(input)

                # ✅ metric calculation
                for metric, metric_func in metrics.items():
                    val = metric_func(target, prediction.squeeze(1))
                    if val != 1:
                        avg_metrics[metric] += val

                # ✅ ALWAYS SAVE OUTPUT
                output_np = prediction.detach().cpu().numpy()
                np.save(osp.join(save_dir, f"{idx}.npy"), output_np)

                bar.update(1)

    # ✅ print metrics
    for metric, avg_metric in avg_metrics.items():
        print("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(dataset)))


if __name__ == "__main__":
    import torch
    test()