export CUDA_VISIBLE_DEVICES=1

model_name=FreDF

python -u main.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --learning_rate 0.001\
  --d_model 128\
  --is_emb True\
  --embed 'learned'\
  --dropout 0\
  --c_out 21 \
  --des 'Exp' \
  --itr 1