OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 ../run.py \
--graph_size 100 \
--batch_size 512 \
--precision 32 \
--run_name debug \
--seed 1235 \
--no_wandb \
--attention_type original \
--encoding_knn_size 100 \
--decoding_knn_size 100 
