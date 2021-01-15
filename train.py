import math
import os
import time

import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.attention_model import set_decode_type
from utils import move_to
from utils.dist_utils import SequentialDistributedSampler, distributed_concat
from utils.log_utils import log_values


def get_inner_model(model):
    return model.module if isinstance(model, DDP) else model


def validate(model, dataset, opts):
    # Validate
    rank = torch.distributed.get_rank()
    if rank == 0:
        print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    if rank == 0:
        print('Validation overall avg_cost: {} +- {}'.format(
            avg_cost,
            torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost
        # return cost.data.cpu()

    return distributed_concat(
        torch.cat([
            eval_model_bat(bat)
            for bat in tqdm(DataLoader(dataset,
                                       batch_size=opts.eval_batch_size,
                                       pin_memory=True,
                                       num_workers=os.cpu_count(),
                                       sampler=SequentialDistributedSampler(
                                           dataset,
                                           batch_size=opts.eval_batch_size)),
                            disable=opts.no_progress_bar)
        ], 0), len(dataset))


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else
            math.inf,  # Inf so no clipping but still call to calc
            norm_type=2) for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms
                          ] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, scaler, baseline, lr_scheduler, epoch,
                val_dataset, problem, opts):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    if rank == 0:
        print("Start train epoch {}, lr={} for run {}".format(
            epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * opts.epoch_size
    start_time = time.time()

    if not opts.no_wandb and rank == 0:
        wandb.log({'learnrate_pg0': optimizer.param_groups[0]['lr']},
                  step=step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(
        problem.make_dataset(size=opts.max_graph_size,
                             num_samples=opts.epoch_size,
                             distribution=opts.data_distribution))
    training_sampler = torch.utils.data.distributed.DistributedSampler(
        training_dataset, num_replicas=world_size, rank=rank)
    training_sampler.set_epoch(epoch)
    training_dataloader = DataLoader(training_dataset,
                                     batch_size=opts.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     sampler=training_sampler,
                                     num_workers=os.cpu_count())

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(
            tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(model, optimizer, scaler, baseline, epoch, batch_id, step,
                    batch, opts)

        step += 1

    epoch_train_duration = time.time() - start_time
    if rank == 0:
        print("Finished epoch {}, took {} s".format(
            epoch, time.strftime('%H:%M:%S',
                                 time.gmtime(epoch_train_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs
            == 0) or epoch == opts.n_epochs - 1 and rank == 0:
        print('Saving model and state...')
        if not os.path.exists(opts.save_dir):
            os.makedirs(opts.save_dir)
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'rng_state': torch.get_rng_state(),
                # 'cuda_rng_state': torch.cuda.get_rng_state_all(),
                # 'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch)))

    start_time = time.time()
    avg_reward = validate(model, val_dataset, opts)
    epoch_validate_duration = time.time() - start_time

    start_time = time.time()
    baseline.epoch_callback(model, epoch)
    epoch_blcallback_duration = time.time() - start_time

    if not opts.no_wandb and rank == 0:
        wandb.log(
            {
                'val_avg_reward': avg_reward,
                "epoch": epoch,
                'train_time': epoch_train_duration,
                'validate_time': epoch_validate_duration,
                'blcallback_time': epoch_blcallback_duration
            },
            step=step)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(model, optimizer, scaler, baseline, epoch, batch_id, step,
                batch, opts):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    if scaler is not None:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            cost, log_likelihood = model(
                x,
                move_to(
                    torch.randint(opts.min_graph_size,
                                  opts.max_graph_size - 1,
                                  size=[x.size(0), 1]), opts.device))
            bl_val, bl_loss = baseline.eval(
                x, cost) if bl_val is None else (bl_val, 0)
            reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
            loss = reinforce_loss + bl_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norms = clip_grad_norms(optimizer.param_groups,
                                     opts.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

    else:
        # Evaluate model, get costs and log probabilities
        cost, log_likelihood = model(
            x)

        # Evaluate baseline, get baseline loss if any (only for critic)
        bl_val, bl_loss = baseline.eval(
            x, cost) if bl_val is None else (bl_val, 0)

        # Calculate loss
        reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
        loss = reinforce_loss + bl_loss

        # Perform backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups,
                                     opts.max_grad_norm)
        optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0 and torch.distributed.get_rank() == 0:
        log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood,
                   reinforce_loss, bl_loss, opts)
