
## instalacion:
Usando conda env tenemos que para que tensorflow pesque la gpu debemos inicial el env asi:

```
conda create -n name tensorflow-gpu
```

de lo contrario no se podra usar la gpu.
dsp instalar los requirements.

## Para evaluar el modelo entrenado usar por ejemplo:

```
python eval.py --image /rois/salmon5/scene00161.png --weights /checkpoints/cp-0100.ckpt
```

donde `--image` es el path a la imagen a evaluar y `--weights` es el path a los pesos entrenados.
