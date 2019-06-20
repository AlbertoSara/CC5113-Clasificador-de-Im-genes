# CC5113 Clasificador de Imágenes

Primera fase del clasificador.

Funciona entrenando un autoencoder, y utlizando su capacidad de compresión para poder extraer features relevantes sobre las cuales se clusteriza posteriormente.

Ya vienen 2000 imágenes elegidas al azar, aunque se puede descargar el resto corriendo imgdownload.py

Instrucciones:

-Entrenar autoencoder. Para eso se debe correr autoencoder.py. Ya se trae un modelo pre-entrenado en la carpeta models por lo que no es necesario entrenar, aunque se puede hacer para probar que funcione o para probar con más datos. Por defecto se entrena con todas las imágenes que hayan en la carpeta "proyecto imagenes".

-Clusterizar. Para esto, se corre encoder-cluster.py. Este programa carga el modelo del autoencoder ya entrenado y toma las capas del encoder. Con esto produce el encoding de todos los datos de la carpeta "proyecto imagenes", le aplica PCA para reducir su dimensionalidad y clusteriza sobre estos datos de dimensionalidad reducida usando KMeans.

Ambos programas producen gráficos. Los entrenamientos no son muy lentos, pero la carga de imágenes puede ser lenta si se cargan muchas (las 2000 incluidas son una cantidad razonable, pero representan aproximadamente el 5% de la base de datos completa).
