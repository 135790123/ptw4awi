import os
if __name__ =="__main__":
    projects = ['collections', 'net', 'mavendp', 'dbcp', 'fileupload', 'configuration', 'codec', 'bcel', 'pool', 'digester']
    for p in projects:
        # os.system("python unixcoder_main.py \
        #       --output_dir=../results/saved_models \
        #       --model_type=roberta \
        #       --do_train \
        #       --do_test \
        #       --train_data_file=../../data/rq4/{}/awi_train.csv \
        #       --eval_data_file=../../data/rq4/{}/awi_test.csv \
        #       --test_data_file=../../data/rq4/{}/awi_test.csv \
        #       --rq {} \
        #       --epochs 10 \
        #       --block_size 512 \
        #       --train_batch_size 16 \
        #       --eval_batch_size 16 \
        #       --learning_rate 2e-5 \
        #       --max_grad_norm 1.0 \
        #       --evaluate_during_training \
        #       --model_name unixcoder_{}.bin \
        #       --n_gpu 1\
        #       --seed 123456  2>&1 | tee train.log".format(p,p,p,p,p))
        os.system("python unixcoder_main.py \
                      --output_dir=../results/saved_models/rq42 \
                      --model_type=roberta \
                      --do_test \
                      --train_data_file=../../data/rq42/{}/awi_train.csv \
                      --eval_data_file=../../data/rq42/{}/awi_test.csv \
                      --test_data_file=../../data/rq42/{}/awi_test.csv \
                      --rq rq42 \
                      --project {} \
                      --epochs 10 \
                      --block_size 512 \
                      --train_batch_size 16 \
                      --eval_batch_size 16 \
                      --learning_rate 2e-5 \
                      --max_grad_norm 1.0 \
                      --evaluate_during_training \
                      --model_name unixcoder_{}.bin \
                      --n_gpu 1\
                      --seed 123456  2>&1 | tee train.log".format(p, p, p, p, p))
        os.system("python unixcoder_main.py \
                      --output_dir=../results/saved_models/rq42 \
                      --model_type=roberta \
                      --do_train \
                      --do_test \
                      --train_data_file=../../data/rq43/{}/awi_train.csv \
                      --eval_data_file=../../data/rq43/{}/awi_test.csv \
                      --test_data_file=../../data/rq43/{}/awi_test.csv \
                      --rq rq43 \
                      --project {} \
                      --epochs 10 \
                      --block_size 512 \
                      --train_batch_size 16 \
                      --eval_batch_size 16 \
                      --learning_rate 2e-5 \
                      --max_grad_norm 1.0 \
                      --evaluate_during_training \
                      --model_name unixcoder_{}.bin \
                      --n_gpu 1\
                      --seed 123456  2>&1 | tee train.log".format(p, p, p, p, p))
        os.system("python unixcoder_main.py \
                      --output_dir=../results/saved_models/rq42 \
                      --model_type=roberta \
                      --do_train \
                      --do_test \
                      --train_data_file=../../data/rq42/{}/awi_train.csv \
                      --eval_data_file=../../data/rq42/{}/awi_test.csv \
                      --test_data_file=../../data/rq42/{}/awi_test.csv \
                      --rq rq44 \
                      --project {} \
                      --epochs 10 \
                      --block_size 512 \
                      --train_batch_size 16 \
                      --eval_batch_size 16 \
                      --learning_rate 2e-5 \
                      --max_grad_norm 1.0 \
                      --evaluate_during_training \
                      --model_name unixcoder_{}.bin \
                      --n_gpu 1\
                      --seed 123456  2>&1 | tee train.log".format(p, p, p, p, p))


