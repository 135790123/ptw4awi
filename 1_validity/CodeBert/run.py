import os
if __name__ =="__main__":
    os.system("python codebert_main.py \
              --output_dir=../results/saved_models \
              --model_type=roberta \
              --do_train \
              --do_test \
              --train_data_file=../../data/awi_train.csv \
              --eval_data_file=../../data/awi_val.csv \
              --test_data_file=../../data/awi_test.csv \
              --rq ori \
              --epochs 10 \
              --block_size 512 \
              --train_batch_size 16 \
              --eval_batch_size 16 \
              --learning_rate 2e-5 \
              --max_grad_norm 1.0 \
              --evaluate_during_training \
              --model_name codebert_ori.bin \
              --n_gpu 1\
              --seed 123456  2>&1 | tee train.log")

    os.system("python codebert_main.py \
                  --output_dir=../results/saved_models \
                  --model_type=roberta \
                  --do_train \
                  --do_test \
                  --train_data_file=../../data/awi_abstract_train.csv \
                  --eval_data_file=../../data/awi_abstract_val.csv \
                  --test_data_file=../../data/awi_abstract_test.csv \
                  --rq abs \
                  --epochs 10 \
                  --block_size 512 \
                  --train_batch_size 16 \
                  --eval_batch_size 16 \
                  --learning_rate 2e-5 \
                  --max_grad_norm 1.0 \
                  --evaluate_during_training \
                  --model_name codebert_abs.bin \
                  --n_gpu 1\
                  --seed 123456  2>&1 | tee train.log")

    os.system("python codebert_main.py \
                  --output_dir=../results/saved_models \
                  --model_type=roberta \
                  --do_train \
                  --do_test \
                  --train_data_file=../../data/awi_context_train.csv \
                  --eval_data_file=../../data/awi_context_val.csv \
                  --test_data_file=../../data/awi_context_test.csv \
                  --rq ctx \
                  --epochs 10 \
                  --block_size 512 \
                  --train_batch_size 16 \
                  --eval_batch_size 16 \
                  --learning_rate 2e-5 \
                  --max_grad_norm 1.0 \
                  --evaluate_during_training \
                  --model_name codebert_ctx.bin \
                  --n_gpu 1\
                  --seed 123456  2>&1 | tee train.log")


