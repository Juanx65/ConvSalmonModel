# ConvSalmonModel
## Benchmarking IPD441
### Juan Aguilera - Felipe Villenas

This repository contains the data and code for the CNN approach for the underwater fish recognition based on dot's skin recognition.

# INSTALLATION:

## Using `virtualenv` in linux

First, create an env using `virtualenv` once inside the Folder of the repository as follows:

```
 virtualenv ConvSalmonModel
```

where, `ConvSallmonModedl` is the name of the env.
Enter the env as follows:

```
 source ConvSalmonModel/bin/activate
```
Then install the requirements:

```
pip install -r requirements.txt
```
note: tensorflow==2.9.1
#### As an alternative, you can try using conda:
For the moment, this method fails to save the checkpoints (at least in windows), so we recomend to try virtualenv.

  ```
  conda create -n ConvSallmonModedl tensorflow-gpu
  conda activate ConvSallmonModedl
  ```
# DATASET
To structure the datasets we use the folders `/rois` as train/valid and `/rois_test` as test. They use the following structure:

```
/rois
  /salmon1
    scene00001.png
    ...
  /salmon2
    ...
  ...
```
Where every `salmon` folder represents the label of every `scene` image inside it.

To create the datasets, we use the functions on `getRoI.py` and `pre_data.py`, that use the image and binaries on the `frames_salmones` folder to get the roi that will be use as a dataset for the model.

# EVALUATE
To evaluate the trained model, use the following example as a guide:
```
python eval.py --image /rois_tests/salmon5/scene00161.png --weights /checkpoints/best.ckpt
```
Where  `--image` is the path to the image to evaluate and  `--weights` is the path to the checkpoints (trained weights of the model).

# TEST
To test the model, use the following example as a guide:
```
python test.py --data_dir 'rois_tests/' --weights '/checkpoints/best.ckpt'
```
Where `--data_dir` is the path to the dataset to test and  `--weights` is the path to the checkpoints (trained weights of the model).

###### `test.py` will display the confusion matrix of a given dataset for the checkpoints of the model.

* Confusion Matrix on the training dataset:
  ![confusion matrix of training dataset.](/images_readme/conf_roi_bench.png)

* Confusion Matrix on a testing dataset:
  ![confusion matrix of test dataset.](/images_readme/conf_roi_tests_bench.png)

# TRAIN
To train the model, use the following example as a guide:
```
python train.py --data_dir "rois/" --epochs 500 --batch_size 32 --save True --optimizer 'adam' --dropout 0.8
```
Where `--save True` saves `best.ckpt` in the `checkpoints` folder, following the validation accuracy of the model. if it is not specified, it wont be saved.

* accuracy on train and validation data over the training process:
  ![confusion matrix of test dataset.](/images_readme/accuracy_bench.png)
