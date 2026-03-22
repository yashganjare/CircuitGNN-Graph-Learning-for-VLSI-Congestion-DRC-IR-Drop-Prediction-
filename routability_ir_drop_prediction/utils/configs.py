# Copyright 2022 CircuitNet. All rights reserved.

import argparse
import os
import sys

sys.path.append(os.getcwd())


class Parser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--task', default='congestion_gpdl')
        self.parser.add_argument('--save_path', default='work_dir/congestion_gpdl/')
        self.parser.add_argument('--pretrained', default=None)
        self.parser.add_argument('--max_iters', default=100)
        self.parser.add_argument('--plot_roc', action='store_true')
        self.parser.add_argument('--arg_file', default=None)
        self.parser.add_argument('--cpu', action='store_true')
        self.get_remainder()
        
    def get_remainder(self):
        if self.parser.parse_args().task == 'congestion_gpdl':
            # MODIFIED PATHS FOR SYNTHETIC DATA
            self.parser.add_argument('--dataroot', default='./synthetic_circuitnet_data/')
            self.parser.add_argument('--ann_file_train', default='./synthetic_circuitnet_data/congestion_annotations.txt')
            self.parser.add_argument('--ann_file_test', default='./synthetic_circuitnet_data/congestion_annotations.txt')
            self.parser.add_argument('--dataset_type', default='CongestionDataset')
            self.parser.add_argument('--batch_size', default=16)
            self.parser.add_argument('--aug_pipeline', default=['Flip'])
            
            self.parser.add_argument('--model_type', default='GPDL')
            self.parser.add_argument('--in_channels', default=3)
            self.parser.add_argument('--out_channels', default=1)
            self.parser.add_argument('--lr', default=2e-4)
            self.parser.add_argument('--weight_decay', default=0)
            self.parser.add_argument('--loss_type', default='MSELoss')
            self.parser.add_argument('--eval-metric', default=['NRMS', 'SSIM', 'EMD'])

        elif self.parser.parse_args().task == 'drc_routenet':
            # MODIFIED PATHS FOR SYNTHETIC DATA
            self.parser.add_argument('--dataroot', default='./synthetic_circuitnet_data/')
            self.parser.add_argument('--ann_file_train', default='./synthetic_circuitnet_data/drc_annotations.txt')
            self.parser.add_argument('--ann_file_test', default='./synthetic_circuitnet_data/drc_annotations.txt')
            self.parser.add_argument('--dataset_type', default='DRCDataset')
            self.parser.add_argument('--batch_size', default=8)
            self.parser.add_argument('--aug_pipeline', default=['Flip'])

            self.parser.add_argument('--model_type', default='RouteNet')
            # MODIFIED CHANNELS TO MATCH SYNTHETIC DATA (3 instead of 9)
            self.parser.add_argument('--in_channels', default=3)
            self.parser.add_argument('--out_channels', default=1)
            self.parser.add_argument('--lr', default=2e-4)
            self.parser.add_argument('--weight_decay', default=1e-4)
            self.parser.add_argument('--loss_type', default='MSELoss')
            self.parser.add_argument('--eval-metric', default=['NRMS', 'SSIM'])
            self.parser.add_argument('--threshold', default=0.1)


        elif self.parser.parse_args().task == 'irdrop_mavi':
            # MODIFIED PATHS FOR SYNTHETIC DATA
            self.parser.add_argument('--dataroot', default='./synthetic_circuitnet_data/')
            self.parser.add_argument('--ann_file_train', default='./synthetic_circuitnet_data/irdrop_annotations.txt')
            self.parser.add_argument('--ann_file_test', default='./synthetic_circuitnet_data/irdrop_annotations.txt')
            self.parser.add_argument('--dataset_type', default='IRDropDataset')
            self.parser.add_argument('--batch_size', default=2)

            self.parser.add_argument('--model_type', default='MAVI')
            # MODIFIED CHANNELS TO MATCH SYNTHETIC DATA (3 input, 1 output)
            self.parser.add_argument('--in_channels', default=1)
            self.parser.add_argument('--out_channels', default=1)
            self.parser.add_argument('--lr', default=2e-4)
            self.parser.add_argument('--weight_decay', default=1e-2)
            self.parser.add_argument('--loss_type', default='L1Loss')
            self.parser.add_argument('--eval_metric', default=['NRMS', 'SSIM'])
            self.parser.add_argument('--threshold', default=0.9885) # 5% after log

        else:
            raise ValueError