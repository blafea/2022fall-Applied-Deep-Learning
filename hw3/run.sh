python run_summarization.py \
    --model_name_or_path ./output_dir \
    --do_predict \
    --test_file $1 \
    --output_dir ./output_dir \
    --output_file $2 \
    --predict_with_generate True \
    --text_column "maintext" \
    --per_device_eval_batch_size 4 \
    --num_beams 3