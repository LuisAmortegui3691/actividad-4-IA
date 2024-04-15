import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(0)
n_samples = 1000
latitudes = np.random.uniform(37.5, 38.0, n_samples)
longitudes = np.random.uniform(-122.5, -122.0, n_samples)
tiempos = np.random.randint(0, 24, n_samples)  # Hora del día (0-23)

df = pd.DataFrame({
    'Latitud': latitudes,
    'Longitud': longitudes,
    'Tiempo': tiempos
})
print(df)

# Aplicar algoritmo de clustering (K-means)
kmeans = KMeans(n_clusters=5)
df['Cluster'] = kmeans.fit_predict(df[['Latitud', 'Longitud']])

# Visualizar los clusters
plt.scatter(df['Longitud'], df['Latitud'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.title('Clustering de Datos de GPS')
plt.colorbar(label='Cluster')
plt.show()