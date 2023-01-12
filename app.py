from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import glob

app = Flask(__name__, template_folder='templateFiles',static_folder='staticFiles')

@app.route('/')
def upload_file():

   return render_template('index.html')
	

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file2():

   # Distance euclidienne
   def calcul_distance(sig1, sig2):
      similarity = np.linalg.norm(sig1 - sig2)
      return similarity


   # Descripteur de couleur + descripteur de texture 
   def calcul_signature(image):
      # Descripteur de Couleur HSV (Hue, Saturation, Value)
      # Convertir l'image en HSV
      hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      # Diviser l'image en 3 canaux (H, S, V)
      channels = cv2.split(hsv)
      # Calculer l'histogramme pour chaque canal
      histograms = []
      for channel in channels:
         hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
         histograms.append(hist)
      
      # Descripteur de Texture Gabor
      g_kernel = cv2.getGaborKernel((21, 21), 5, 0, 10, 0.5, 0, cv2.CV_32F)
      gabor = cv2.filter2D(image, cv2.CV_8UC3, g_kernel)
      # Convertir l'image en niveaux de gris
      gray = cv2.cvtColor(gabor, cv2.COLOR_BGR2GRAY)
      # Calculer la transformée de Gabor
      gabor_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
      histograms.append(gabor_hist)

      # Normaliser les histogrammes
      histograms = [hist / np.sum(hist) for hist in histograms]
      # Concaténer les histogrammes
      descriptor = np.concatenate(histograms)

      return descriptor.flatten()


   # Calcul des signatures
   def signatures():
      # Obtenir une liste des path des images
      file_paths = glob.glob("C:/Users/amala/Desktop/ImageSearchEngine/MiniProjet/ImageDB/*.jpg")
      # Le path qu'on va utiliser pour l'affichage des images dans la page de résultat.
      # On utilise http-server pour contourner l'erreur -> Cannot open local file - Chrome: Not allowed to load local resource
      # npm install -g http-server
      # Executer cette ligne de commande dans le dossier principal -> http-server ./
      path = "http://127.0.0.1:8080/ImageDB/"

      # Array pour stocker les signatures
      signatures = []

      # Boucle pour lire toutes les images
      for file_path in file_paths:
         # Lire l'image
         img = cv2.imread(file_path)
         # Calculer la signature et l'ajouter dans l'Array avec le path de l'image
         signatures.append([path+file_path[60:],calcul_signature(img)])

      return signatures

   

   # Main

   signatures = signatures()

   if request.method == 'POST':
      # Lire l'image et la stocker
      f = request.files['file']
      f.save(secure_filename(f.filename))
      img = cv2.imread('C:/Users/amala/Desktop/ImageSearchEngine/MiniProjet/'+f.filename)
      target = "http://127.0.0.1:8080/"+f.filename

      #Calculer la signature de l'image
      imgSig = calcul_signature(img)

      # Array pour stocker les distances
      distances = []

      # Calculer toutes les distances, stocker distance avec le path de l'image
      for sig in signatures:
         distances.append([sig[0],calcul_distance(sig[1],imgSig)])

      # Trier l'Array par distances
      distances.sort(key=lambda x:x[1])

      # On retourne les 20 premieres images les plus pertinentes
      return render_template('resultat.html', var=distances[0:20], target=target)

if __name__ == '__main__':
   app.run(debug = True)

