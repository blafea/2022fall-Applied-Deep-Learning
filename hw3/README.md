# Homework 3 ADL NTU 111

## To train
```shell
python ./run_summarization.py \
    --model_name_or_path google/mt5-small \
    --do_train \
    --do_eval \
    --train_file <train_file> \
    --validation_file <valid_file> \
    --output_dir ./output_dir \
    --cache_dir ./cache \
    --per_device_train_batch_size 4 \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --text_column "maintext" \
    --summary_column "title" \
    --predict_with_generate True
```


## To reproduce
```shell
bash download.sh
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

