OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 ../run.py \
--min_graph_size 20 \
--max_graph_size 50 \
--batch_size 1280 \
--eval_batch_size 2000 \
--precision 32 \
--run_name debug \
--seed 1235 \
--no_wandb \
--attention_type original
