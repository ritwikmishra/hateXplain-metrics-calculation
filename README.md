



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
`CUDA_VISIBLE_DEVICES=1 python ajeet-calculate-metrics.py.py --method lrp --faithfullness_filtering top-k --split 1 --model_path PATH_TO_MODEL --data_path PATH_TO_DATA_FOLDER --encoder_name bert-base-cased`

# Cite

```
@inproceedings{mishra2023explaining,
  title={Explaining Finetuned Transformers on Hate Speech Predictions Using Layerwise Relevance Propagation},
  author={Mishra, Ritwik and Yadav, Ajeet and Shah, Rajiv Ratn and Kumaraguru, Ponnurangam},
  booktitle={International Conference on Big Data Analytics},
  pages={201--214},
  year={2023},
  organization={Springer}
}

```
If you use the dataset in your work then consider citing the following:
```
@inproceedings{mathew2021hatexplain,
  title={HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection},
  author={Mathew, Binny and Saha, Punyajoy and Yimam, Seid Muhie and Biemann, Chris and Goyal, Pawan and Mukherjee, Animesh},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={17},
  pages={14867--14875},
  year={2021}
}
```
If you use ferret library then consider citing the following:
```
@inproceedings{attanasio-etal-2023-ferret,
    title = "ferret: a Framework for Benchmarking Explainers on Transformers",
    author = "Attanasio, Giuseppe and Pastor, Eliana and Di Bonaventura, Chiara and Nozza, Debora",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = may,
    year = "2023",
    publisher = "Association for Computational Linguistics",
}
```



