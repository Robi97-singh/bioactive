#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import wandb
import argparse

import torch
from defaults import *
from utils.system_def import *
from utils.launch import dist, launch, synchronize


global debug



MODEL_MAP = {
    "resnet":     ("resnet50",        "bioact_resnet50"),
    "dino":       ("dinov2_base",     "bioact_dinov2_base"),
    "dino_large": ("dinov2_large",    "bioact_dinov2_large"),
    "clip":       ("clip_vitl14",     "bioact_clip_vitl14"),
    "biomedclip": ("biomedclip",      "bioact_biomedclip"),
    "convnext":   ("convnext_base",   "bioact_convnext"),
    "vit":        ("vit_b_16",        "bioact_vit_b16"),
}

def _apply_resolution(params, res):
    dp = params['dataset_params']
    for key in ('train_transforms', 'val_transforms', 'test_transforms'):
        if key not in dp:
            continue
        t = dp[key]
        if res == 224:
            t.setdefault('Resize', {})
            t['Resize']['apply'] = True
            t['Resize']['height'] = 540
            t['Resize']['width']  = 540
            for crop in ('CenterCrop', 'RandomCrop'):
                if crop in t and t[crop].get('apply'):
                    t[crop]['height'] = 224
                    t[crop]['width']  = 224
        elif res == 448:
            if 'Resize' in t:
                t['Resize']['apply'] = False
            for crop in ('CenterCrop', 'RandomCrop'):
                if crop in t and t[crop].get('apply'):
                    t[crop]['height'] = 448
                    t[crop]['width']  = 448

def _fold_splits(fold, n_folds=6):
    val = (fold + 1) % n_folds
    train = [f for f in range(n_folds) if f not in (fold, val)]
    return {"train": train, "val": [val], "test": [fold]}


def parse_arguments():
    parser = argparse.ArgumentParser(description='The main takes as \
                             argument the parameters dictionary from a json file')
    parser.add_argument('--params_path', type=str, required=False, 
                        default="./params.json",
                        help='Give the path of the json file which contains the training parameters')
    parser.add_argument('--checkpoint', type=str, required=False, help='Give a valid checkpoint name')
    parser.add_argument('--test', action='store_true', default=False, help='Flag for testing')

    parser.add_argument('--debug', action='store_true', default=False, help='Flag for turning on the debug_mode')
    parser.add_argument('--data_location', type=str, required=False, help='Update the datapath')
    parser.add_argument('--dist_url', type=str, default='', required=False,
                        help='URL of master node, for use with SLURM')
    parser.add_argument('--port', type=int, required=False, default=45124, 
                        help='Explicit port selection, for use with SLURM')
    parser.add_argument('--gpu', type=str, required=False, help='The GPU to be used for this run')
    parser.add_argument('--model_name', type=str, required=False, help='Used to manipulate the model_name defined in the param file for this run')
    parser.add_argument('--save_dir', type=str, required=False, help='Change the "save_dir" in param file')
    
    parser.add_argument('--dry', action='store_true', default=False, help='Flag for changing parm file suitable for a dry run')
    parser.add_argument('--backbone_type', type=str, help='Change backbone')
    parser.add_argument('--batch_size', type=int, help='Change the batch size of all data loaders')
    
    parser.add_argument('--val_every', type=float, help='How many epochs between each validation')
    parser.add_argument('--use_mixed_precision', type=int, default=-1, help='Using this arg with value 1 (0) results in explicity turning on (off) the mixed precision. By default it follows the param file.')
        
    
    # ablation flags
    parser.add_argument('--knn', action='store_true', default=False, help='Flag for turning on the KNN eval')        
    

    parser.add_argument('--model', type=str, choices=list(MODEL_MAP.keys()),
                        help='Select model by friendly name (sets backbone_type + model_name)')
    parser.add_argument('--res', type=int, choices=[224, 448],
                        help='Input resolution. 448=crop448 (paper). 224=resize540+crop224 (same FOV).')
    parser.add_argument('--fold', type=int, choices=[0,1,2,3,4,5],
                        help='CV fold to hold out as test (rotates val/test/train).')

    return parser.parse_args()


def update_params_from_args(params, args):
    if args.gpu:
        prev_gpu = params.system_params.which_GPUs
        params.system_params.which_GPUs = args.gpu  # change the value in-place
        print('Changed GPU for this run from {} to \033[1m{}\033[0m'.format(prev_gpu, args.gpu))

    if args.model_name:
        prev_model_name = params.training_params.model_name
        params.training_params.model_name = args.model_name
        print('Changed model_name for this run from {} to \033[1m{}\033[0m'.format(prev_model_name, args.model_name))

    if args.data_location:
        params['dataset_params']['data_location'] = args.data_location
        print('Changed data_location to: "\033[1m{}\033[0m"'.format(args.data_location))

    if args.save_dir:
        params['training_params']['save_dir'] = args.save_dir
        print('Changed save_dir to: "\033[1m{}\033[0m"'.format(args.save_dir))

    if args.backbone_type:
        params['model_params']['backbone_type'] = args.backbone_type
        print('Changed backbone_type to: \033[1m{}\033[0m'.format(args.backbone_type))

    if args.model:
        backbone, mname = MODEL_MAP[args.model]
        params['model_params']['backbone_type'] = backbone
        if not args.model_name:
            params['training_params']['model_name'] = mname
        print('Selected model "\033[1m{}\033[0m" -> backbone_type={}, model_name={}'.format(
              args.model, backbone, params['training_params']['model_name']))

    if args.res:
        _apply_resolution(params, args.res)
        print('Set resolution to \033[1m{}\033[0m (transforms updated)'.format(args.res))

    if args.fold is not None:
        splits = _fold_splits(args.fold)
        params['dataset_params']['data_split_numbers'] = splits
        base_mn = params['training_params']['model_name']
        if not base_mn.endswith('_fold{}'.format(args.fold)):
            params['training_params']['model_name'] = '{}_fold{}'.format(base_mn, args.fold)
        print('Set CV fold \033[1m{}\033[0m -> splits={}, model_name={}'.format(
              args.fold, splits, params['training_params']['model_name']))


    if args.batch_size:
        for loader in ['trainloader', 'valloader', 'testloader']:
            params['dataloader_params'][loader]['batch_size'] = args.batch_size
            print('Changed \033[1m{}\033[0m batch_size to: \033[1m{}\033[0m'.format(loader, args.batch_size))

    if args.val_every is not None:
        params['training_params']['val_every'] = args.val_every
        print('Changed val_every to: \033[1m{}\033[0m'.format(args.val_every))

    if args.use_mixed_precision != -1:  # if -1, it won't change the param file
        assert args.use_mixed_precision == 0 or args.use_mixed_precision == 1, 'Argument --use_mixed_precision shold be either 1 or 0'
        if_mixed = bool(args.use_mixed_precision)  # this step is not crucial but is good for consistency
        params['training_params']['use_mixed_precision'] = if_mixed
        print('Explicitly set use_mixed_precision to: \033[1m{}\033[0m'.format(if_mixed))



def main(parameters, args):
    
    
    # define system
    define_system_params(parameters.system_params)
    
    # Instantiate wrapper with all its definitions   
    wrapper = DefaultWrapper(parameters)
    wrapper.instantiate()

    # initialize logger
    if wrapper.is_rank0:
        log_params = wrapper.parameters.log_params    
        training_params = wrapper.parameters.training_params
        if wrapper.log_params['run_name'] == "DEFINED_BY_MODEL_NAME":
            log_params['run_name'] = training_params.model_name  
        if args.debug:
            os.environ['WANDB_MODE'] = 'dryrun'
        if not (args.test):
            if parameters.training_params.use_tensorboard:
                print("Using TensorBoard logging")
                summary_writer = SummaryWriter()
            else:
                print("Using WANDB logging")
                wandb.init(project=log_params.project_name, 
                           name=log_params.run_name, 
                           config=wrapper.parameters,
                           resume=True if training_params.restore_session else False)
    
    # define trainer 
    trainer = Trainer(wrapper)
        
    if parameters.training_params.use_tensorboard:
        trainer.summary_writer = summary_writer
        
    if args.test:
        trainer.test()       
    else:
        trainer.train()
        if wrapper.is_supervised:
            trainer.test()
        
    
if __name__ == '__main__':
    args = parse_arguments()
    parameters = edict(load_params(args))
    update_params_from_args(parameters, args)

    try:
        launch(main, (parameters, args))
    except Exception as e:       
        if dist.is_initialized():
            dist.destroy_process_group()            
        raise e
    finally:
        if dist.is_initialized():
            synchronize()         
            dist.destroy_process_group()            
