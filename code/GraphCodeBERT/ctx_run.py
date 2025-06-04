import os

# nohup python ctx_run.py > ctx.out 2>&1 &

def run_graphcodebert(train_file, eval_file, test_file, model_name, log_file, rq="ori", 
                 epochs=10, block_size=512, train_batch_size=32, eval_batch_size=16, 
                 learning_rate=2e-5, max_grad_norm=1.0, n_gpu=2, seed=123456):
    # 构建命令字符串
    command = f"""
    CUDA_VISIBLE_DEVICES=6,7 python graphcodebert_main.py \
        --output_dir=../results/saved_models \
        --model_type=roberta \
        --do_train \
        --do_test \
        --train_data_file={train_file} \
        --eval_data_file={eval_file} \
        --test_data_file={test_file} \
        --rq {rq} \
        --epochs {epochs} \
        --block_size {block_size} \
        --train_batch_size {train_batch_size} \
        --eval_batch_size {eval_batch_size} \
        --learning_rate {learning_rate} \
        --max_grad_norm {max_grad_norm} \
        --evaluate_during_training \
        --model_name {model_name} \
        --n_gpu {n_gpu} \
        --seed {seed} \
        2>&1 | tee {log_file}
    """

    # 打印命令（调试用）
    print("Executing command:")
    print(command)
    
    # 执行命令
    os.system(command)

if __name__ =="__main__":

     # ori
    # run_graphcodebert(
    #     train_file="/root/sy/ori_java_dataset/train_fold_1.csv",
    #     eval_file="/root/sy/ori_java_dataset/test_fold_1.csv",
    #     test_file="/root/sy/ori_java_dataset/test_fold_1.csv",
    #     model_name="graphcodebert_ori1.bin",
    #     log_file="ori_train_fold_1.log",
    #     rq="ori_1"
    # )
    # run_graphcodebert(
    #     train_file="/root/sy/ori_java_dataset/train_fold_2.csv",
    #     eval_file="/root/sy/ori_java_dataset/test_fold_2.csv",
    #     test_file="/root/sy/ori_java_dataset/test_fold_2.csv",
    #     model_name="graphcodebert_ori2.bin",
    #     log_file="ori_train_fold_2.log",
    #     rq="ori_2"
    # )
    # run_graphcodebert(
    #     train_file="/root/sy/ori_java_dataset/train_fold_3.csv",
    #     eval_file="/root/sy/ori_java_dataset/test_fold_3.csv",
    #     test_file="/root/sy/ori_java_dataset/test_fold_3.csv",
    #     model_name="graphcodebert_ori3.bin",
    #     log_file="ori_train_fold_3.log",
    #     rq="ori_3"
    # )
    # run_graphcodebert(
    #     train_file="/root/sy/ori_java_dataset/train_fold_4.csv",
    #     eval_file="/root/sy/ori_java_dataset/test_fold_4.csv",
    #     test_file="/root/sy/ori_java_dataset/test_fold_4.csv",
    #     model_name="graphcodebert_ori4.bin",
    #     log_file="ori_train_fold_4.log",
    #     rq="ori_4"
    # )
    # run_graphcodebert(
    #     train_file="/root/sy/ori_java_dataset/train_fold_5.csv",
    #     eval_file="/root/sy/ori_java_dataset/test_fold_5.csv",
    #     test_file="/root/sy/ori_java_dataset/test_fold_5.csv",
    #     model_name="graphcodebert_ori5.bin",
    #     log_file="ori_train_fold_5.log",
    #     rq="ori_5"
    # )

    # # # abs
    # run_graphcodebert(
    #     train_file="/root/sy/abs_java_dataset/train_fold_1.csv",
    #     eval_file="/root/sy/abs_java_dataset/test_fold_1.csv",
    #     test_file="/root/sy/abs_java_dataset/test_fold_1.csv",
    #     model_name="graphcodebert_abs1.bin",
    #     log_file="abs_train_fold_1.log",
    #     rq="abs_1"
    # )
    # run_graphcodebert(
    #     train_file="/root/sy/abs_java_dataset/train_fold_2.csv",
    #     eval_file="/root/sy/abs_java_dataset/test_fold_2.csv",
    #     test_file="/root/sy/abs_java_dataset/test_fold_2.csv",
    #     model_name="graphcodebert_abs2.bin",
    #     log_file="abs_train_fold_2.log",
    #     rq="abs_2"
    # )
    # run_graphcodebert(
    #     train_file="/root/sy/abs_java_dataset/train_fold_3.csv",
    #     eval_file="/root/sy/abs_java_dataset/test_fold_3.csv",
    #     test_file="/root/sy/abs_java_dataset/test_fold_3.csv",
    #     model_name="graphcodebert_abs3.bin",
    #     log_file="abs_train_fold_3.log",
    #     rq="abs_3"
    # )
    # run_graphcodebert(
    #     train_file="/root/sy/abs_java_dataset/train_fold_4.csv",
    #     eval_file="/root/sy/abs_java_dataset/test_fold_4.csv",
    #     test_file="/root/sy/abs_java_dataset/test_fold_4.csv",
    #     model_name="graphcodebert_abs4.bin",
    #     log_file="abs_train_fold_4.log",
    #     rq="abs_4"
    # )
    # run_graphcodebert(
    #     train_file="/root/sy/abs_java_dataset/train_fold_5.csv",
    #     eval_file="/root/sy/abs_java_dataset/test_fold_5.csv",
    #     test_file="/root/sy/abs_java_dataset/test_fold_5.csv",
    #     model_name="graphcodebert_abs5.bin",
    #     log_file="abs_train_fold_5.log",
    #     rq="abs_5"
    # )

    # ctx
    run_graphcodebert(
        train_file="/root/sy/ctx_java_dataset/train_fold_1.csv",
        eval_file="/root/sy/ctx_java_dataset/test_fold_1.csv",
        test_file="/root/sy/ctx_java_dataset/test_fold_1.csv",
        model_name="graphcodebert_ctx1.bin",
        log_file="ctx_train_fold_1.log",
        rq="ctx_1"
    )
    run_graphcodebert(
        train_file="/root/sy/ctx_java_dataset/train_fold_2.csv",
        eval_file="/root/sy/ctx_java_dataset/test_fold_2.csv",
        test_file="/root/sy/ctx_java_dataset/test_fold_2.csv",
        model_name="graphcodebert_ctx2.bin",
        log_file="ctx_train_fold_2.log",
        rq="ctx_2"
    )
    run_graphcodebert(
        train_file="/root/sy/ctx_java_dataset/train_fold_3.csv",
        eval_file="/root/sy/ctx_java_dataset/test_fold_3.csv",
        test_file="/root/sy/ctx_java_dataset/test_fold_3.csv",
        model_name="graphcodebert_ctx3.bin",
        log_file="ctx_train_fold_3.log",
        rq="ctx_3"
    )
    run_graphcodebert(
        train_file="/root/sy/ctx_java_dataset/train_fold_4.csv",
        eval_file="/root/sy/ctx_java_dataset/test_fold_4.csv",
        test_file="/root/sy/ctx_java_dataset/test_fold_4.csv",
        model_name="graphcodebert_ctx4.bin",
        log_file="ctx_train_fold_4.log",
        rq="ctx_4"
    )
    run_graphcodebert(
        train_file="/root/sy/ctx_java_dataset/train_fold_5.csv",
        eval_file="/root/sy/ctx_java_dataset/test_fold_5.csv",
        test_file="/root/sy/ctx_java_dataset/test_fold_5.csv",
        model_name="graphcodebert_ctx5.bin",
        log_file="ctx_train_fold_5.log",
        rq="ctx_5"
    )