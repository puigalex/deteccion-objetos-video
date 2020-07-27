import os
import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument("--directorio_origen", type=str, default="data/custom/images", help="Directorio donde se encuentran todas las imagenes y las etiquetas")
parser.add_argument("--directorio_destino", type=str, default="data/custom", help="directorio donde se escribira train y test txt")
opt = parser.parse_args()

path = opt.directorio_origen

files = os.listdir(path)
random.shuffle(files)
train = files[:int(len(files)*0.9)]
val = files[int(len(files)*0.9):]

with open('{}/train2.txt'.format(opt.directorio_destino), 'w') as f:
    for item in train:
        f.write("{}/{} \n".format(path, item))

with open('{}/valid2.txt'.format(opt.directorio_destino), 'w') as f:
    for item in val:
        f.write("{}/{} \n".format(path, item))