# Text Detoxification

Name: Oksana Konovalova

Email: o.konovalova@innopolis.university

Group number: B21-DS-02

This work was done for the first assessment of the PMLDL course.

## Task description

Text Detoxification Task is a process of transforming the text with toxic style into the text with the same meaning but with neutral style.

## Data Description

The dataset is a subset of the ParaNMT corpus (50M sentence pairs). The filtered ParaNMT-detox corpus (500K sentence pairs) can be downloaded from [here](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip). 

The data is given in the `.tsv` format.

| Column | Type | Discription | 
| ----- | ------- | ---------- |
| reference | str | First item from the pair | 
| ref_tox | float | toxicity level of reference text | 
| translation | str | Second item from the pair - paraphrazed version of the reference|
| trn_tox | float | toxicity level of translation text |
| similarity | float | cosine similarity of the texts |
| lenght_diff | float | relative length difference between texts |

## Solution 
[BERT base model (uncased)](https://huggingface.co/bert-base-uncased)

## Running the model

- For data preprocessing
```
python src/data/make_dataset.py
```
- For make a predictions 
 ```
python src/models/predict_model.py
```



