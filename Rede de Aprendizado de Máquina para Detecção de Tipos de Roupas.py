# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:41:37 2024

@author: marce
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
import pandas as pd
import joblib as jl



(imagens_treino, rótulos_treino), (imagens_teste, rótulos_teste) = mnist.load_data()

nomes_classes = ['T-shirt/top', 'Trouser',
 'Pullover', 'Dress', 'Coat',
 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

imagens_treino = imagens_treino/255.0
imagens_teste = imagens_teste/255.0


modelo = tf.keras.Sequential([
    Flatten(input_shape=(28,28)),
    BatchNormalization(),
    Dense(1024, activation = 'relu'),
    Dropout(0.4),
    BatchNormalization(),
    Dense(1024, activation = 'relu'),
    Dropout(0.4),
    BatchNormalization(),
    Dense(1024, activation = 'relu'),
    Dropout(0.4),
    BatchNormalization(),
    Dense(10, activation = 'softmax')
    ])

modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])


paralização = EarlyStopping(
    patience=3,
    min_delta=0.001,
    restore_best_weights=True
    )

historico = modelo.fit(imagens_treino, rótulos_treino,batch_size=500, epochs = 30, validation_split=0.2, callbacks=[paralização], verbose = 2)


(erro, acerto) = modelo.evaluate(imagens_teste, rótulos_teste)

print(modelo.summary())
print(acerto)

#Analisa uma imagem interna 
imagem = imagens_teste[20]
imagem = (np.expand_dims(imagem,0))
predição = modelo.predict(imagem, steps=1)
print ("Resultado da predição")
print(predição)
print()
Numero_do_elemento = predição.argmax()
Elemento = np.amax(predição)

print("A imagem '20' é um/a: "+nomes_classes[Numero_do_elemento] )
print (str(int(Elemento*100)) + "% Nivel de confiança")


#Analisa uma imagem externa 
imagem_externa = r'C:\Users\marce\OneDrive\Documentos\AI\Rede de Aprendizado de Máquina para Detecção de Tipos de Roupas\Rede-de-Aprendizado-de-Maquina-para-Deteccao-de-Tipos-de-Roupas\Sandalha.jpg'

ImagemT = Image.open(imagem_externa)
ImagemT.load()
informação = np.asarray( ImagemT, dtype='float')
informação = tf.image.rgb_to_grayscale(informação)
informação = informação/255.0
informação = tf.transpose(informação, perm=[2,0,1])

predição1 = modelo.predict(informação,steps=1 )
print ("Resultado da predição")
print(predição1)
print()
Numero_do_elemento1 = predição1.argmax()
Elemento1 = np.amax(predição1)

print("A imagem é um/a: "+nomes_classes[Numero_do_elemento1] )
print (str(int(Elemento1*100)) + "% Nivel de confiança")

historico = pd.DataFrame(historico.history)
historico.loc[:, ['loss', 'val_loss']].plot()
historico.loc[:, ['accuracy', 'val_accuracy']].plot()
