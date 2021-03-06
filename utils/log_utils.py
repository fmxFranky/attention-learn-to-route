import torch
import wandb


def log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood,
               reinforce_loss, bl_loss, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(
        epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0],
                                              grad_norms_clipped[0]))

    # Log values via wandb
    if not opts.no_wandb and torch.distributed.get_rank() == 0:
        metrics = {
            'avg_cost': avg_cost,
            'actor_loss': reinforce_loss.item(),
            'nll': -log_likelihood.mean().item(),
            'grad_norm': grad_norms[0],
            'grad_norm_clipped': grad_norms_clipped[0]
        }
        if opts.baseline == 'critic':
            metrics.update({
                'critic_loss': bl_loss.item(),
                'critic_grad_norm': grad_norms[1],
                'critic_grad_norm_clipped': grad_norms_clipped[1],
            })
        wandb.log(metrics)
