# Sample Code for Homework 1 ADL NTU 109 Spring

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
conda activate adl-hw1
pip install -r requirements.txt
# otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
python3 train_intent.py --data_dir <data_dir> --cache_dir <cache_dir> --ckpt_dir <ckpt_dir> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout>
```

### To reproduce
```shell
bash download.sh
bash intent_cls.sh /path/to/test.json /path/to/pred.csv
```

## Slot tagging
### train
```shell
python3 train_slot.py --data_dir <data_dir> --cache_dir <cache_dir> --ckpt_dir <ckpt_dir> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout>
```


### To reproduce
```shell
bash download.sh
bash slot_tag.sh /path/to/test.json /path/to/pred.csv
```

