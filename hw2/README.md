# Homework 2 ADL NTU 111

## Context Selection
```shell
python run_swag_no_trainer.py \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --context_file <context_file> \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --pad_to_max_length \
  --max_length 512 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
```

## Question Answering
```shell
python run_qa_no_trainer.py \
  --model_name_or_path <model_name_or_path> \
  --output_dir <output_dir> \
  --train_file <train_file> \
  --validation_file <validation_file> \
  --context_file <context_file> \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --pad_to_max_length \
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --doc_stride 128 \
```


## To reproduce
```shell
bash ./download.sh
bash ./run.sh /path/to/context.json /path/to/test.json  /path/to/pred/prediction.csv
```

