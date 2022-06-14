
## instalacion:

# usando conda en windows 11:
fracase miserablemente entrenando la red en windows 11 porque tensorflow 2.9.1 tira un error de que no hay permisos para escribir los checkpoints, puedes entrenar pero no guardara los pesos... de todas formas, asi lo isntale:
  Usando conda env tenemos que para que tensorflow pesque la gpu debemos inicial el env asi:

  ```
  conda create -n name tensorflow-gpu
  ```

  de lo contrario no se podra usar la gpu.
  dsp instalar los requirements.

## usando virtualenv en manjaro linux

creamos un pipenv de la siguiente manera:

```
 virtualenv ConvSalmonModel
```
y entramos a él de la siguiente forma:
```
 source ConvSalmonModel/bin/activate
```

luego instalamos los requirementes:

```
pip install -r requirements.txt
```

isi, aun asi tensorflow no se que tiene que tira más warnings que la perra

## Para evaluar el modelo entrenado usar por ejemplo:

```
python eval.py --image /rois/salmon5/scene00161.png --weights /checkpoints/best.ckpt
```

donde `--image` es el path a la imagen a evaluar y `--weights` es el path a los pesos entrenados.


## Para entrenar modelo, guiarse por el siguiente ejemplo:

```
python train.py --data_dir "rois/" --epochs 500 --batch_size 32 --save True --optimizer 'adam' --dropout 0.8
```

donde `--save True` guardara `best.ckpt` segun la validacion del modelo, si no se especifica no guardara nada.


## Adicionales

recprdar actualizar las clases dependiendo del modelo para las funciones en `train.py`,  `eval.py`, `model.py` y `pre_data.py` donde el ultimo usa la funcion en `getRoI.py` para estructurar la carpeta utilizada para generar el datasert del modelo a entrenar utilizando la imagen original y el binario.
```
names = ['salmon1_tests','salmon2_tests','salmon3_tests','salmon5_tests']
```


## NOTAS ADICIONALES PARA LOS AUTORES:

* En la funcion `getRoI` se puede elegir retornar la ROI segun el paper, o simplemente `first_roi` que seria el pez entero recortado ajaja
* Para el bench usaremos el RoI, pero por los problemas que trae porobablemente usar una red solo con `first_roi` tenga mejores resultados.
