import random
import math

# Datos de las ciudades
ciudades = {
    "Bogotá": [103.5, 7.18, 10.5, 32, 52, 48, 18],
    "Medellín": [44.1, 2.57, 11.2, 31, 53, 47, 7.5],
    "Cali": [22.4, 2.23, 13.8, 30, 52, 48, 4.2],
    "Barranquilla": [16.8, 1.23, 12.4, 29, 51, 49, 3.1],
    "Cartagena": [10.5, 1.03, 10.9, 30, 51, 49, 2.8],
    "Bucaramanga (test)": [7.3, 0.58, 9.2, 33, 52, 48, 1.5],
    "Pereira": [6.2, 0.48, 12, 32, 52, 48, 1.3],
    "Cúcuta (test)": [5.1, 0.76, 16.3, 28, 51, 49, 1.2],
    "Ibagué (test)": [4.8, 0.53, 13.4, 31, 52, 48, 1.1],
    "Santa Marta": [4, 0.52, 11.6, 29, 51, 49, 0.9],
    "Manizales": [3.8, 0.43, 10.7, 32, 53, 47, 0.8],
    "Villavicencio": [3.5, 0.5, 13, 30, 51, 49, 0.8],
    "Pasto": [3.2, 0.45, 12.9, 31, 52, 48, 0.7],
    "Montería": [3, 0.49, 13.5, 29, 51, 49, 0.7],
    "Valledupar": [2.8, 0.47, 14.8, 28, 51, 49, 0.6],
    "Neiva": [2.5, 0.35, 14.1, 30, 52, 48, 0.6],
    "Popayán": [2.3, 0.33, 15.2, 31, 52, 48, 0.5],
    "Armenia": [2.1, 0.3, 13.3, 32, 53, 47, 0.5],
    "Sincelejo": [2, 0.28, 16.5, 29, 51, 49, 0.5],
    "Tunja": [1.8, 0.25, 10, 31, 52, 48, 0.4],
    "Florencia": [1.7, 0.2, 17.5, 28, 51, 49, 0.4],
    "Riohacha": [1.5, 0.22, 15.7, 27, 51, 49, 0.3],
    "Quibdó": [1.3, 0.13, 18.2, 26, 52, 48, 0.3],
    "San Andrés": [1.2, 0.08, 14, 27, 50, 50, 0.2],
    "Yopal": [1.1, 0.15, 11.5, 29, 51, 49, 0.2],
    "Leticia": [1, 0.05, 13.6, 26, 51, 49, 0.1],
    "Arauca (test)": [0.9, 0.08, 12.2, 29, 51, 49, 0.1],
    "Mocoa": [0.8, 0.04, 15, 28, 52, 48, 0.1],
    "Mitú": [0.7, 0.01, 20, 25, 51, 49, 0.05],
    "Puerto Carreño (test)": [0.6, 0.01, 22, 24, 50, 50, 0.05]
}

# Extraemos los nombres de las ciudades de prueba
ciudades_test = ["Bucaramanga (test)", "Cúcuta (test)", "Ibagué (test)", "Arauca (test)", "Puerto Carreño (test)", "Mocoa"]

# Función para calcular la distancia euclidiana entre dos puntos (vectores)
def calcular_distancia(punto1, punto2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(punto1, punto2)))


# Función para inicializar centroides aleatoriamente
def inicializar_centroides(ciudades, k):
    return random.sample(list(ciudades.values()), k)


# Función para asignar cada ciudad a un cluster (centroide más cercano)
def asignar_clusters(ciudades, centroides):
    clusters = {}
    for ciudad, datos in ciudades.items():
        distancias = [calcular_distancia(datos, centroide) for centroide in centroides]
        cluster = distancias.index(min(distancias))
        clusters[ciudad] = cluster
    return clusters


# Función para recalcular centroides basados en los clusters actuales
def recalcular_centroides(ciudades, clusters, k):
    centroides = [[0] * len(next(iter(ciudades.values()))) for _ in range(k)]
    conteo = [0] * k

    for ciudad, cluster in clusters.items():
        for i, valor in enumerate(ciudades[ciudad]):
            centroides[cluster][i] += valor
        conteo[cluster] += 1

    # Calcular promedio para cada cluster
    for i in range(k):
        if conteo[i] > 0:
            centroides[i] = [x / conteo[i] for x in centroides[i]]
    return centroides


# Función para implementar el algoritmo K-means
def kmeans(ciudades, k=6, max_iter=1000000):
    centroides = inicializar_centroides(ciudades, k)
    for _ in range(max_iter):
        clusters = asignar_clusters(ciudades, centroides)
        nuevos_centroides = recalcular_centroides(ciudades, clusters, k)
        if nuevos_centroides == centroides:
            break
        centroides = nuevos_centroides
    return clusters, centroides

# Ejecutar K-means
clusters, centroides = kmeans(ciudades, k=6)

# Obtener las ciudades sin el conjunto de prueba
ciudades_entrenamiento = {ciudad: datos for ciudad, datos in ciudades.items() if ciudad not in ciudades_test}

# Asignar las ciudades de prueba a su cluster más cercano
def asignar_ciudades_prueba(ciudades_test, centroides):
    resultados = {}
    for ciudad_test in ciudades_test:
        datos_test = ciudades[ciudad_test]
        distancias = [calcular_distancia(datos_test, centroide) for centroide in centroides]
        cluster_cercano = distancias.index(min(distancias))
        resultados[ciudad_test] = cluster_cercano
    return resultados

# Asignar cada ciudad de prueba a su cluster
clusters_test = asignar_ciudades_prueba(ciudades_test, centroides)

# Encontrar la ciudad más cercana dentro del mismo cluster para cada ciudad de prueba
def encontrar_ciudad_mas_cercana(ciudades, clusters, clusters_test):
    resultados_finales = {}
    for ciudad_test, cluster_test in clusters_test.items():
        # Filtrar las ciudades en el mismo cluster (excepto la ciudad de prueba)
        ciudades_mismo_cluster = [ciudad for ciudad, cluster in clusters.items() if cluster == cluster_test and ciudad != ciudad_test]
        # Verificar que haya al menos una ciudad en el cluster
        if ciudades_mismo_cluster:
            distancias = {ciudad: calcular_distancia(ciudades[ciudad_test], ciudades[ciudad]) for ciudad in ciudades_mismo_cluster}
            ciudad_mas_cercana = min(distancias, key=distancias.get)
            resultados_finales[ciudad_test] = ciudad_mas_cercana
        else:
            resultados_finales[ciudad_test] = None  # Si no hay ciudades en el mismo cluster
    return resultados_finales

# Ejecutar el código corregido
resultados_kaggle = encontrar_ciudad_mas_cercana(ciudades, clusters, clusters_test)

# Mostrar el resultado en formato ID, label
print("ID,label")
for ciudad_test, ciudad_cercana in resultados_kaggle.items():
    if ciudad_cercana:
        print(f"{ciudad_test},{ciudad_cercana}")
    else:
        print(f"{ciudad_test},No disponible")