#!/usr/bin/env python

import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from nets.attention_model import AttentionModel
from nets.critic_network import CriticNetwork
from nets.graph_encoder import Normalization
from nets.pointer_network import CriticNetworkLSTM, PointerNetwork
from options import get_options
from reinforce_baselines import (CriticBaseline, ExponentialBaseline,
                                 NoBaseline, RolloutBaseline, WarmupBaseline)
from train import get_inner_model, train_epoch, validate
from utils import load_problem, torch_load_cpu


def run(opts):

    rank = opts.local_rank if torch.cuda.device_count() > 1 else 0

    # Set the random seed
    torch.manual_seed(opts.seed + rank)
    random.seed(opts.seed + rank)
    np.random.seed(opts.seed + rank)

    if not os.path.exists(opts.save_dir) and rank == 0:
        os.makedirs(opts.save_dir)

    # Optionally configure wandb
    if not opts.no_wandb and rank == 0:
        wandb.login('never', '31ce01e4120061694da54a54ab0dafbee1262420')
        wandb.init(dir=opts.save_dir,
                   config=opts,
                   project='large_scale_tsp',
                   name=opts.run_name,
                   sync_tensorboard=True,
                   save_code=True)

    # Set the device
    if opts.use_cuda:
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        opts.device = torch.device("cuda", rank)

    else:
        opts.device = torch.device("cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        if rank == 0:
            print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model: torch.nn.Module = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        attention_type=opts.attention_type,
        n_encode_layers=opts.n_encode_layers,
        n_heads=opts.n_heads,
        feed_forward_dim=opts.feed_forward_dim,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size).to(opts.device)

    if opts.init_normalization_parameters:
        for m in model.modules():
            if isinstance(m, Normalization):
                m.init_parameters()

    if opts.use_cuda:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(
            opts.device)
        model = DDP(model, device_ids=[rank])

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({
        **model_.state_dict(),
        **load_data.get('model', {})
    })

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (CriticNetworkLSTM(2, opts.embedding_dim, opts.hidden_dim,
                               opts.n_encode_layers, opts.tanh_clipping)
             if opts.baseline == 'critic_lstm' else CriticNetwork(
                 2, opts.embedding_dim, opts.hidden_dim, opts.n_encode_layers,
                 opts.normalization)).to(opts.device))
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(
            opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline,
                                  opts.bl_warmup_epochs,
                                  warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam([{
        'params': model.parameters(),
        'lr': opts.lr_model
    }] + ([{
        'params': baseline.get_learnable_parameters(),
        'lr': opts.lr_critic
    }] if len(baseline.get_learnable_parameters()) > 0 else []))

    scaler = torch.cuda.amp.GradScaler() if opts.precision == 16 else None

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: opts.lr_decay**epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(size=opts.graph_size,
                                       num_samples=opts.val_size,
                                       filename=opts.val_dataset,
                                       distribution=opts.data_distribution)

    if opts.resume:
        epoch_resume = int(
            os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        if rank == 0:
            print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(model, optimizer, scaler, baseline, lr_scheduler,
                        epoch, val_dataset, problem, opts)


if __name__ == "__main__":
    run(get_options())
