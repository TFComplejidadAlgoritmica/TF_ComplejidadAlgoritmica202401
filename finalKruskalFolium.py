import pandas as pd
import folium
import webbrowser
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Clase que representa y maneja el Grafo
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    # Agregar arista al Grafo
    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    # Función de búsqueda Union Find
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    # Realiza UNION de "x" y "y"
    def apply_union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        # Coloca la raíz del árbol más pequeño bajo la raíz del árbol más grande
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    # Aplicando el Algoritmo Kruskal
    def kruskal_algo(self):
        result = [] # Resultado
        i, e, tot_weight = 0, 0, 0  # "i": índice usado para las aristas ordenadas. "e" usado para result[]
        self.graph = sorted(self.graph, key=lambda item: item[2]) # Ordena el Grafo por los Costos de las aristas, en orden creciente
        parent = []
        rank = []
        # Crea subconjuntos a partir de los Vertices V
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1: # Mientras el número de aristas a tomar es menor que V-1
            # Elegimos el borde más pequeño e incrementamos el índice para la próxima iteración
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            # Si incluir este borde no provoca un ciclo, inclúyalo en el resultado e incremente el índice del resultado para el borde siguiente
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)

        # Aristas que forman parte del MST
        for u, v, weight in result:
            tot_weight = tot_weight + weight
            print("%d - %d: %f" % (u, v, weight))

        print("Costo total del MST para el grafo: ", tot_weight)

        return result

ubicacion_base = [-24.1858, -65.2992]

mapa = folium.Map(location=ubicacion_base, zoom_start=13)

archivo = 'dataset-jujuy.csv'
datos = pd.read_csv(archivo)

datos[['longitud', 'latitud']] = datos['geojson'].str.split(',', expand=True)

datos['latitud'] = datos['latitud'].astype(float)
datos['longitud'] = datos['longitud'].astype(float)

folium.Marker(location=ubicacion_base, icon=folium.Icon(color='red')).add_to(mapa)


nodos_medio = []
nodos_alta = []
ubicaciones = [ubicacion_base]  

for indice, fila in datos.iterrows():
    ubicacion = [fila['latitud'], fila['longitud']]
    folium.Marker(location=ubicacion, icon=folium.Icon(color='orange')).add_to(mapa)
    ubicaciones.append(ubicacion)
    if fila['tension'] <= 20:
        nodos_medio.append(ubicacion)
    elif fila['tension'] == 33:
        nodos_alta.append(ubicacion)

g = Graph(len(ubicaciones))

for i in range(len(ubicaciones)):
    for j in range(i + 1, len(ubicaciones)):
        dist = haversine(ubicaciones[i][0], ubicaciones[i][1], ubicaciones[j][0], ubicaciones[j][1])
        g.add_edge(i, j, dist)

mst_edges = g.kruskal_algo()

for edge in mst_edges:
    u, v, w = edge
    folium.PolyLine([ubicaciones[u], ubicaciones[v]], color="blue", weight=2.5, opacity=1).add_to(mapa)

mapa.save('mapaGrafo.html')
webbrowser.open('mapaGrafo.html')

