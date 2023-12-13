# ICCAD TINYML IEGM Training Code, Team ICCL

## Requirements
```
numpy==1.21.5
torch==1.11.0
torchvision==0.12.0
```
*May work on older versions as well but not fully tested on older versions. The shared requirements are the tested versions.

## Run a simple training with default settings
To run the training

```
python3 trainf2_classes.py --data [PATH TO DATASET]
```

## Dataset
The dataset is the IEGM dataset provided by the organizers for the contest. However, to obtain a stable training we randomly shuffle the dataset between validation and train data. The provided implementation randomly splits the dataset into train data (random 56 subjects) and validation data (random 20 subjects).
 
To select a diverse train split which ensure the random train split has adequate number of instances for each of the real label:

```
python3 trainf2_classes.py --data [PATH TO DATASET] --diverse_train_split
```

## Stable Metric to Choose Best Model

Since test data is hidden and given the variability in performance across subjects, there may be high difference between training and validation performance. In some cases training score is superior whereas in some cases validation score is superior. To remedy this we used a stablitity metric which selects the best model only if the Trainset F-2 score and Valset F-2 score are similar. To enable stability metric:
```
python3 trainf2_classes.py --data [PATH TO DATASET] --stable_metric
```

## Converting to ONNX

To convert your trained model to onnx:

```
python3 trainf2_classes.py --data [PATH TO DATASET] --batch_size 1 --onnx --trained_path [PATH TO TRAINED PYTORCH MODEL]
```

Please use batch size 1 for this step. Secondly please give path to the trained pytorch model and not the trained pytorch weights.

## To see number of ops and parameters for the architecture

```
python3 trainf2_classes.py --data [PATH TO DATASET] --batch_size 1 --get_ops 
```
Use batch size 1. E.g our submitted model has:
```
========================================
#OPS: tensor([281.8160])    #Kops
#parmas: tensor([570.])     #Total Parms
========================================
```


## Training hyperparameters

All the default values in the parsed arguments in `trainf2_claseses.py` are the best found values but can be used to modify the training hyperparameters.