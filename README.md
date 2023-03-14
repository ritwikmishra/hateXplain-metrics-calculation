# For hateXplain-training.py




## Python environment & Dependencies
* create a python environment using above requirement.txt file with python version-3.8 


## data folder
This 'data' folder should contain following 17-files:
* dataset.json
* hateXplain.json
* train_split1.json, test_split1.json, val_split1.json (for hatespeech model training on split1 data)
* train_split2.json, test_split2.json, val_split2.json (for hatespeech model training on split2 data)
* train_split3.json, test_split3.json, val_split3.json (for hatespeech model training on split3 data)
* test_split1.jsonl, test_split2.jsonl, test_split3.jsonl (these .jsonl file will be used during hateXplain metrics calculation)
* post_id_divisions.json
* post_id_division_split1_seed_1234.json
* post_id_division_split2_seed_12345.json

## utility files
Following three files should be in the same folder in which ajeet_calculate_metrics.py file is.
* classes.npy
* classes_two.npy
* metrics.py


# FOR HATESPEECH MODEL TRAINING, example command

`CUDA_VISIBLE_DEVICES=0 python hatespeech-training.py --split 1 --bias_in_fc True --add_cls_sep_tokens False --epochs 5 --encoder_frozen True --encoder_name bert-base-cased --data_path DATA_PATH --checkpoint_path CHECKPOINT_PATH --run_ID RUN_ID`


# FOR HATEXPLAIN METRIC CALCULATION, example command
`CUDA_VISIBLE_DEVICES=1 python ajeet-calculate-metrics.py.py --method lrp --faithfullness_filtering top-k --split 1 --model_path /home/ajeet/ajeet_falcon/ajeet/ritwik_experiments/trained_models/hateSpeechModel-bert-base-cased-freezed-bias-in-fc-cls-token-dataSplit1_epoch_5.pth.t7 --data_path DATA_PATH --encoder_name bert-base-cased`



