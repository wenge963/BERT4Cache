set -x

CKPT_DIR="/tmp/BERT4rec"
dataset_name="ml-1m"
max_seq_length=200
masked_lm_prob=0.2
max_predictions_per_seq=40

dim=64
batch_size=256
num_train_steps=400000

prop_sliding_window=0.5
mask_prob=1.0
dupe_factor=10
pool_size=10

signature="-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mpps${max_predictions_per_seq}-msl${max_seq_length}"


CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --train_input_file=./data/${dataset_name}${signature}.train.tfrecord \
    --test_input_file=./data/${dataset_name}${signature}.test.tfrecord \
    --vocab_filename=./data/${dataset_name}${signature}.vocab \
    --user_history_filename=./data/${dataset_name}${signature}.his \
    --checkpointDir=${CKPT_DIR}/${dataset_name} \
    --signature=${signature}-${dim} \
    --do_train=False \
    --do_eval=True \
    --bert_config_file=./bert_train/bert_config_${dataset_name}_${dim}.json \
    --save_predictions_file=./ml-1m_pred.txt \
    --batch_size=${batch_size} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=100 \
    --learning_rate=1e-4

