export CUDA_VISIBLE_DEVICES=1

model_name=FreDF

python -u main.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --freq m\
  --learning_rate 0.0001\
  --d_model 512\
  --is_emb True\
  --embed 'timeF'\
  --dropout 0\
  --c_out 137 \
  --des 'Exp' \
  --itr 1