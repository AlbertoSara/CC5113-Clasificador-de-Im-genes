# CC5113 Clasificador de Imágenes

Primera fase del clasificador.

Funciona entrenando un autoencoder, y utlizando su capacidad de compresión para poder extraer features relevantes sobre las cuales se clusteriza posteriormente.

Ya vienen 2000 imágenes elegidas al azar, aunque se puede descargar el resto corriendo imgdownload.py

Instrucciones:

-Entrenar autoencoder. Para eso se debe correr autoencoder.py. Ya se trae un modelo pre-entrenado en la carpeta models por lo que no es necesario entrenar, aunque se puede hacer para probar que funcione o para probar con más datos. Por defecto se entrena con todas las imágenes que hayan en la carpeta "imagenes proyecto".

-Clusterizar. Para esto, se corre encoder-cluster.py. Este programa carga el modelo del autoencoder ya entrenado y toma las capas del encoder. Con esto produce el encoding de todos los datos de la carpeta "imagenes proyecto", le aplica PCA para reducir su dimensionalidad y clusteriza sobre estos datos de dimensionalidad reducida usando KMeans.

Ambos programas producen gráficos. Los entrenamientos no son muy lentos, pero la carga de imágenes puede ser lenta si se cargan muchas (las 2000 incluidas son una cantidad razonable, pero representan aproximadamente el 5% de la base de datos completa).


-Clasificar usando ResNet50 y clusterizacion. Para entrenar nuestro cluster de kmeans y nuestro espacio PCA, se corre resnet_train.py. Ya se incluyen modelos entrenados en la carpeta models (models/pca.joblib y models/kmeans.joblib). Es importante notar que nombrar los clusters es un proceso manual que se debe hacer cada vez que se entrene un par (kmeans, pca), por lo que es recomendado no entrenar y usar los modelos provistos para que las clases entregadas en class.csv etiqueten al cluster correspondiente. Para clasificar, se ponen las imagenes deseadas en la carpeta "imagenes clasificar". Luego, se corre el programa resnet_test.py, que clasifica las imagenes. Las primeras 15 se grafican con su clase predicha, pero todas las demás son clasificadas y su clase predicha se imprime en la salida estandar.
