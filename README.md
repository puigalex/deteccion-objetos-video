# Deteccion de objetos en video 
Este repo basado en el proyecto [PyTorch YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) para correr detección de objetos sobre video. Construí sobre este proyecto para añadir la capacidad de detectar objetos en un stream de video en vivo.

[YOLO](https://pjreddie.com/darknet/yolo/) es un modelo el cual esta optimizado para generar detecciones de elementos a una velocidad muy alta, es por eso que es una muy buena opción para usarlo en video. Tanto el entrenamiento como predicciones con este modelo se ven beneficiadas si se cumple con una computadora que tenga una GPU NVIDIA.

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
python detec_video.py
```

# Modificaciones
Si en vez de correr detección de objetos sobre la webcam lo que quieres es correr el modelo sobre un video que ya fue pre grabado tienes que cambiar el comando para correr el codigo a:

```
python detec_video.py --webcam 0 --directorio_video <directorio_al_video.mp4>
```

