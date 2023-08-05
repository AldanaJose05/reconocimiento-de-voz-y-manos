# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 08:56:12 2023

"""
#______________________________________________________________________________
#---------------------------Se importan las librerias -------------------------

import math  # Importa el módulo math para operaciones matemáticas.
import cv2  # Importa la biblioteca OpenCV para procesamiento de imágenes y videos.
import mediapipe as mp  # Importa la biblioteca Mediapipe para la detección de manos.
import time  # Importa el módulo time para trabajar con el tiempo.
import mediapipe.python.solutions.drawing_utils as mp_drawing_utils  # Importa utilidades de dibujo de Mediapipe.
from scipy.io import wavfile
from scipy import signal
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import numpy as np
import winsound 
import pyaudio
import wave

#---------------------------Clase que detecta las manos------------------------
#En una clase llamada detector_manos que se va a hacer la deteccion de manos 


# Definimos la clase que va a hacer la deteccion de manos.
class detector_manos():
    
    
#-------------Se inicializan los parámetros de detección de manos--------------
    def __init__(self, mode=False, maxManos=1, Confdeteccion=0.75, Confseguimiento=0.5):
        self.mode = mode #modo de deteccion
        self.maxManos = maxManos #número máximo de manos a detectar
        self.Confdeteccion = int(Confdeteccion)  #La confianza mínima requerida para considerar que una detección es válida
        self.Confseguimiento = Confseguimiento   # La confianza de seguimiento
        self.mpmanos = mp.solutions.hands  # Crea una instancia de la solución de detección de manos de Mediapipe.
        self.manos = self.mpmanos.Hands(self.mode, self.maxManos, self.Confdeteccion, self.Confseguimiento)  # Crea un objeto para detectar manos.
        self.dibujo = mp_drawing_utils  # Se usan las utilidades del dibujo de Mediapipe.
        self.tip = [4, 8, 12, 16, 20]  # Índices de las puntas de los dedos.


#------Este metodo toma una imagen y realiza la deteccion de mano--------------
    def encontrar_manos(self, frame, dibujar=True):
        imgcolor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# Convierte la imagen de BGR a RGB 
        self.resultados = self.manos.process(imgcolor)   #Se procesa la detección de mano

        if self.resultados.multi_hand_landmarks:         #Se verifica si se han detectado la mano o no
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    # si se detecto una imagen se dibujan los puntos y las conexiones de las manos en la imagen .
                    self.dibujo.draw_landmarks(frame,mano,
                        self.mpmanos.HAND_CONNECTIONS,
                        self.dibujo.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        self.dibujo.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
        return frame #Se devuelve la imagen con los puntos y conexiones de las manos dibujados en ella


#------Se toma la imagen y devuelve la posicion de los puntos importantes de la mano------
    def encontrar_posicion(self, frame, ManoNum=0, dibujar=True):
        xlista = []  #Se guarda la coordenada x
        ylista = []  #Se guarda la coordenada y   
        bbox = []    #Para el contorno de la imagen    
        self.lista = []

        if self.resultados.multi_hand_landmarks:  #Sirve para verificar si se han detectado múltiples manos en la imagen. 
            miMano = self.resultados.multi_hand_landmarks[ManoNum]
            for id, lm in enumerate(miMano.landmark):#Se itera sobre cada punto clave de la mano y se hace una tipo etiqueta
                alto, ancho, c = frame.shape         #Dimensiones de la imagen
                cx, cy = int(lm.x * ancho), int(lm.y * alto)#Datos normalizados
                xlista.append(cx)
                ylista.append(cy)
                self.lista.append([id, cx, cy])#Se agrega a la lista el identificador y las coordenadas x e y del punto clave 
                if dibujar:
                    # Dibuja un círculo en cada punto de la mano.
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)
                    
            #Son los límites del cuadro delimitador que rodea a la mano.
            xmin, xmax = min(xlista), max(xlista)
            ymin, ymax = min(ylista), max(ylista)
            bbox = xmin, ymin, xmax, ymax
            if dibujar:
                # Dibuja un rectángulo alrededor de la mano.
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
        return self.lista, bbox


#--------Este metodo determina si los dedos están levantados o no -------------
    def dedo_arriba(self):  
        #Devuelve una lista que indica si cada dedo está levantado o no.
        dedos = []
        if self.lista[self.tip[0]][1] > self.lista[self.tip[0] - 1][1]:
            dedos.append(1)
        else:
            dedos.append(0)

        for id in range(1, 5):
            if self.lista[self.tip[id]][2] < self.lista[self.tip[id] - 2][2]:
                dedos.append(1)
            else:
                dedos.append(0)

        return dedos

#---------------------------Funcion del DTW---------------------------------

def dista_dtw(matriz_distancia):
    N,M = matriz_distancia.shape
    matriz_costo = np.zeros((N+1,M+1))
    for i in range(1,N+1):
        matriz_costo[i,0] = np.inf
    for i in range(1,M+1):
        matriz_costo[0,i] = np.inf
    for i in range(N):
        for j in range(M):
            penalty = [matriz_costo[i,j],
                       matriz_costo[i,j+1],
                       matriz_costo[i+1,j]]
            i_penalty = np.argmin(penalty)
            matriz_costo[i+1,j+1] = matriz_distancia[i,j]+penalty[i_penalty]
    matriz_costo = matriz_costo[1:,1:]
    return matriz_costo

#############################################################################################
#---------------------------Reconocimiento de la mano-------------------------

# Crea una instancia del detector de manos.
detector = detector_manos(Confdeteccion=0.75)

# Crea una instancia de VideoCapture para capturar el video de la cámara web.
cap = cv2.VideoCapture(0)
z = 0
while True:
    # Lee la cámara web.
    ret, frame = cap.read()
    
    # Detecta y dibuja las manos 
    frame = detector.encontrar_manos(frame, dibujar=True)
    
    # Obtiene las dimensiones del la imagen
    alto, ancho, _ = frame.shape
    
    # Coordenadas del rectángulo y el texto en la esquina superior izquierda
    x1, y1 = 20, 20
    x2, y2 = 175, 135
    
    # Dibuja el rectángulo en la esquina superior izquierda
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)
    
    # Escribe el texto "Dedos" en la esquina superior izquierda
    cv2.putText(frame, "Dedos levantados", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    
    # Encuentra la posición de las manos y extrae la información.
    manosInfo, cuadro = detector.encontrar_posicion(frame, dibujar=False)
    #print(manosInfo)
    # Realiza el conteo de los dedos levantados y muestra el resultado.
    if len(manosInfo) != 0:
        dedos = detector.dedo_arriba()
        #print(dedos)
        contar = dedos.count(1)
        # Espera la pulsación de una tecla durante 1 milisegundo.
        t = cv2.waitKey(1)
    
        # Si la tecla espacio (código 32) es presionada
        if t == 32:
            if z == 0:
                a = contar
                print("Primer numero guardado:")
                print(contar)
                z=1
            else:
                b = contar
                print("Segundo numero guardado:")
                print(contar)
                z=0
            
        # Si la tecla Esc (código 27) es presionada, se libera la captura de video y se cierran todas las ventanas.
        if t == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("Fin del video")
            break
        
        #cv2.putText(frame, str(contar), (445, 375), cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 25)
        cv2.putText(frame, str(contar), (x1+45,y1+100), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 8)
    
    # Muestra el fotograma en una ventana.
    cv2.imshow("Contando Dedos", frame)
    
    t = cv2.waitKey(1)
    if t == 27:
        cap.release()
        cv2.destroyAllWindows()
        print("Fin del video")
        break

#----------------------------GRABACION DE AUDIO ------------------------------
chunk=512   #paquetes
sample_format=pyaudio.paInt16   #
channels=2  #Canales de la computadora ,eso varia en la computadora 
fs=48000     #Frecuencia de muestreo ,se capturan 8000 muestras por segundo
seconds=2   #el tiempo en el que puedes decir una palabra ,eso depende del usuario
filename='Prueba.wav'     #archivo completo que no tiene compresion 
audio_obj=pyaudio.PyAudio() #Es un streaming de audio con un solo vector 
input('Graba palabra de operacion')
print('Inicia Grabacion')
stream=audio_obj.open(format=sample_format,channels=channels,
                      rate=fs,frames_per_buffer=chunk,input=True)  #podemos hacer la grabacion
tramas=[]
sonido=[]
for i in range(0,int(fs/chunk*seconds)):
    datos=stream.read(chunk)
    tramas.append(datos)
    sonido.append(np.frombuffer(datos,dtype=np.int16))
stream.stop_stream()
stream.close()
audio_obj.terminate()
print('termina grabacion')

wf=wave.open(filename,'wb')
wf.setnchannels(channels)
wf.setsampwidth(audio_obj.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(tramas))
wf.close
winsound.PlaySound(filename, winsound.SND_FILENAME|winsound.SND_ASYNC)
#____________________________________________________________________________

# PRUEBA-->PARA EL AUDIO GRABADO
muestreop,datosp = wavfile.read('Prueba.wav')
tp = np.arange(len(datosp))/float(muestreop)
datosp = datosp[:,0]/(2**15)
#------------------------------------------------------------------------------
#------------------------OPERACIONES ------------------------------------------
# SUMA
muestreo1,datos1 = wavfile.read('Suma.wav')
t1 = np.arange(len(datos1))/float(muestreo1)
datos1 = datos1[:,0]/(2**15)
# RESTA
muestreo2,datos2 = wavfile.read('Resta.wav')
t2 = np.arange(len(datos2))/float(muestreo2)
datos2 = datos2[:,0]/(2**15)
# MULTIPLICACION 
muestreo3,datos3 = wavfile.read('Multiplicacion.wav')
t3 = np.arange(len(datos3))/float(muestreo3)
datos3 = datos3[:,0]/(2**15)
# DIVISION
muestreo4,datos4 = wavfile.read('Division.wav')
t4 = np.arange(len(datos4))/float(muestreo4)
datos4 = datos4[:,0]/(2**15)

frecuenciap,tiempop,espectrop = signal.spectrogram(datosp,muestreop,nfft=1024,noverlap=100,nperseg=1024)
frecuencia1,tiempo1,espectro1 = signal.spectrogram(datos1,muestreo1,nfft=1024,noverlap=100,nperseg=1024)
frecuencia2,tiempo2,espectro2 = signal.spectrogram(datos2,muestreo2,nfft=1024,noverlap=100,nperseg=1024)
frecuencia3,tiempo3,espectro3 = signal.spectrogram(datos3,muestreo3,nfft=1024,noverlap=100,nperseg=1024)
frecuencia4,tiempo4,espectro4 = signal.spectrogram(datos4,muestreo4,nfft=1024,noverlap=100,nperseg=1024)


distancia1 = dist.cdist(np.log(espectro1.T), np.log(espectrop.T), 'euclidean')
costo1 = dista_dtw(distancia1)
distancia2 = dist.cdist(np.log(espectro2.T), np.log(espectrop.T), 'euclidean')
costo2 = dista_dtw(distancia2)
distancia3 = dist.cdist(np.log(espectro3.T), np.log(espectrop.T), 'euclidean')
costo3 = dista_dtw(distancia3)
distancia4 = dist.cdist(np.log(espectro4.T), np.log(espectrop.T), 'euclidean')
costo4 = dista_dtw(distancia4)

costos = [costo1[-1,-1],costo2[-1,-1],costo3[-1,-1],costo4[-1,-1]]
#print(costos)
#______________________________________________________________________________

if (min(costos) == costos[0]):
    #SUMA
    c=a+b
    print("El resultado de la suma es")
    print(c)
    
elif (min(costos) == costos[1]):
    #RESTA
    c=a-b
    print("El resultado de la resta es")
    print(c)
    
elif (min(costos) == costos[2]):
    #MULTIPLICACION 
    c=a*b
    print("El resultado de la multiplicacion es")
    print(c)
    
elif (min(costos) == costos[3]):
    #DIVICION
    c=a/b
    print("El resultado de la divicion es")
    print(c)