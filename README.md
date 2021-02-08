# Tag-enhancedRS
This repository contains a implementation of our "Reorganizing Users by Item Tags for Recommendation".

## Environment Setup
1. Pytorch 1.4.0
2. Python 3.6+

## Guideline

### data

We provide two dataset, ciao and toys, which have been divided into train/valid/test set. The ratio is [0.6, 0.2, 0.2]

```item_cat.dat``` is the tag labels of items for each dataset. 
```negatives.dat``` is the result of negative sampling. We make sure that each positive interaction maps to 20 negative items.

### code

The implementation of a baseline model SML(```SML.py```); 

The implementation of our proposed model TSML(```TSML.py```);

A Base class to train and evaluate the models(```BaseModel.py```)

Some utility functions for running program(```util.py```)

## Example to run the codes

Run TSML under ciao dataset
```
python main.py --model TSML --dataset ciao
```

Run TSML under toys dataset
```
python main.py --model TSML --dataset toys
```
