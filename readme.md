
# INSTALLATION:

## Using `virtualenv` in linux (tensorflow 2.9.1)


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

## Using Conda in any OS.
For the moment, this method fails to save the checkpoints (at least in windows), so we recomend to try virtualenv.

  ```
  conda create -n ConvSallmonModedl tensorflow-gpu
  conda activate ConvSallmonModedl
  ```


#  EVALUATE
To evaluate the trained model, use the following example as a guide:
```
python eval.py --image /rois_tests/salmon5_tests/scene00161.png --weights /checkpoints/best.ckpt
```

Where  `--image` is the path to the image to evaluate and  `--weights` is the path to the checkpoints (trained weights of the model).

# TRAIN
To train the model, use the following example as a guide:
```
python train.py --data_dir "rois/" --epochs 500 --batch_size 32 --save True --optimizer 'adam' --dropout 0.8
```

Where `--save True` saves `best.ckpt` in the `checkpoints` folder, following the validation accuracy of the model. if it is not specified, it wont be saved.


## MORE
* recordar actualizar las clases dependiendo del modelo para las funciones en `train.py`,       `eval.py`, `model.py` y `pre_data.py` donde el ultimo usa la funcion en `getRoI.py` para estructurar la carpeta utilizada para generar el datasert del modelo a entrenar utilizando la imagen original y el binario.
```
names = ['salmon1_tests','salmon2_tests','salmon3_tests','salmon5_tests']
```
* En la funcion `getRoI` se puede elegir retornar la ROI segun el paper, o simplemente `first_roi` que seria el pez entero recortado ajaja
* Para el bench usaremos el RoI, pero por los problemas que trae porobablemente usar una red solo con `first_roi` tenga mejores resultados.
