# Deteccion de objetos en video 
Este repo basado en el proyecto [PyTorch YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) para correr detección de objetos sobre video. Construí sobre este proyecto para añadir la capacidad de detectar objetos en un stream de video en vivo.

[YOLO](https://pjreddie.com/darknet/yolo/) (**You Only Look Once** o Tú Solo Ves Una Vez, pero TSVUV no suena tan bien) es un modelo el cual esta optimizado para generar detecciones de elementos a una velocidad muy alta, es por eso que es una muy buena opción para usarlo en video. Tanto el entrenamiento como predicciones con este modelo se ven beneficiadas si se cumple con una computadora que tenga una GPU NVIDIA.

Por default este modelo esta pre entrenado para detecta 80 distintos objetos, la lista de estos se encuentra en el archivo [data/coco.names](https://github.com/puigalex/deteccion-objetos-video/blob/master/data/coco.names)

Los pasos a seguir para poder correr detección de objetos en el video de una webcam son los siguientes (La creación del ambiente asume que Anaconda esta instalado en la computadora):

# Crear ambiente
Para tener en orden nuestras paqueterias de python primero vamos a crear un ambiente llamado "deteccionobj" el cual tiene la version 3.6 de python
``` 
conda create -n deteccionobj python=3.6
```

Activamos el ambiente deteccionobj para asegurarnos que estemos en el ambiente correcto al momento de hacer la instalación de todas las paqueterias necesarias
```
source activate deteccionobj
```

# Instalación de las paqueterias
Estando dentro de nuestro ambiente vamos a instalar todas las paqueterias necesarias para correr nuestro detector de objetos en video, la lista de los paqueter y versiones a instalar están dentro del archivo requirements.txt por lo cual instalaremos haciendo referencia a ese archivo
```
pip install -r requirements.txt
```

# Descargar los pesos del modelo entrenado 
Para poder correr el modelo de yolo tendremos que descargar los pesos de la red neuronal, los pesos son los valores que tienen todas las conexiones entre las neuronas de la red neuronal de YOLO, este tipo de modelos son computacionalmente muy pesados de entrenar desde cero por lo cual descargar el modelo pre entrenado es una buena opción.

```
bash weights/download_weights.sh
```

Movemos los pesos descargados a la carpeta llamada weights
```
mv yolov3.weights weights/
```

# Correr el detector de objetos en video 
Por ultimo corremos este comando el cual activa la camara web para poder hacer deteccion de video sobre un video "en vivo"
```
python deteccion_video.py
```

# Modificaciones
Si en vez de correr detección de objetos sobre la webcam lo que quieres es correr el modelo sobre un video que ya fue pre grabado tienes que cambiar el comando para correr el codigo a:

```
python deteccion_video.py --webcam 0 --directorio_video <directorio_al_video.mp4>
```

# Entrenamiento 

Ahora, si lo que quieres es entrenar un modelo con las clases que tu quieras y no utilizar las 80 clases que vienen por default podemos entrenar nuestro propio modelo. Estos son los pasos que deberás seguir:

Primero deberás etiquetar las imagenes con el formato VOC, aqui tengo un video explicando como hacer este etiquetado: 

Desde la carpeta config correremos el archivo create_custom_model para generar un archivo .cfg el cual contiene información sobre la red neuronal para correr las detecciones
```
cd config
bash create_custom_model.sh <Numero_de_clases_a_detectar>
cd ..
```
Descargamos la estructura de pesos de YOLO para poder hacer transfer learning sobre esos pesos
```
cd weights
bash download_darknet.sh
cd ..
```

## Poner las imagenes y archivos de metadata en las carpetar necesarias

Las imagenes etiquetadas tienen que estar en el directorio **data/custom/images** mientras que las etiquetas/metadata de las imagenes tienen que estar en **data/custom/labels**.
Por cada imagen.jpg debe de existir un imagen.txt (metadata con el mismo nombre de la imagen)

El archivo ```data/custom/classes.names``` debe contener el nombre de las clases, como fueron etiquetadas, un renglon por clase.

Los archivos ```data/custom/valid.txt``` y ```data/custom/train.txt``` deben contener la dirección donde se encuentran cada una de las imagenes.

## Entrenar

 ```
 python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74 --batch_size 2
 ```

## Correr deteccion de objetos en video con nuestras clases
```
python deteccion_video.py --model_def config/yolov3-custom.cfg --checkpoint_model checkpoints/yolov3_ckpt_99.pth --class_path data/custom/classes.names  --weights_path checkpoints/yolov3_ckpt_99.pth  --conf_thres 0.85
```
