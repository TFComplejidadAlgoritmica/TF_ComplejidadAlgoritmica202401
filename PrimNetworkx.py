import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import sys

# Función para calcular la distancia Haversine entre dos puntos geográficos
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radio de la Tierra en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Clase para el grafo y el algoritmo de Prim
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    def add_edge(self, u, v, w):
        self.graph[u][v] = w
        self.graph[v][u] = w  # Asegura que sea bidireccional

    def min_key(self, key, mst_set):
        min_val = float('inf')
        min_index = -1
        for v in range(self.V):
            if key[v] < min_val and not mst_set[v]:
                min_val = key[v]
                min_index = v
        return min_index

    def prim_algo(self):
        key = [float('inf')] * self.V
        parent = [None] * self.V
        key[0] = 0
        mst_set = [False] * self.V
        parent[0] = -1
        mst_edges = []

        for _ in range(self.V):
            u = self.min_key(key, mst_set)
            mst_set[u] = True
            for v in range(self.V):
                if self.graph[u][v] > 0 and not mst_set[v] and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        for i in range(1, self.V):
            mst_edges.append((parent[i], i, self.graph[i][parent[i]]))

        return mst_edges

def main(num_filas):
    # Coordenadas base
    ubicacion_base = [-24.1858, -65.2992]

    # Leer los datos del archivo CSV
    archivo = 'dataset-jujuy.csv'
    datos = pd.read_csv(archivo, nrows=num_filas)

    # Extraer y convertir las coordenadas de los datos
    datos[['longitud', 'latitud']] = datos['geojson'].str.strip(' "').str.split(',', expand=True)
    datos['latitud'] = datos['latitud'].astype(float)
    datos['longitud'] = datos['longitud'].astype(float)

    ubicaciones = [ubicacion_base]
    nodos_medio = []
    nodos_alta = []

    for indice, fila in datos.iterrows():
        ubicacion = [fila['latitud'], fila['longitud']]
        ubicaciones.append(ubicacion)
        if fila['tension'] <= 20:
            nodos_medio.append(ubicacion)
        elif fila['tension'] == 33:
            nodos_alta.append(ubicacion)

    g = Graph(len(ubicaciones))

    # Conectar ubicación base a nodos de alta tensión primero, si existen
    if nodos_alta:
        for nodo_alta in nodos_alta:
            dist = round(haversine(ubicacion_base[0], ubicacion_base[1], nodo_alta[0], nodo_alta[1]), 2)
            g.add_edge(0, ubicaciones.index(nodo_alta), dist)
    else:
        for nodo_medio in nodos_medio:
            dist = round(haversine(ubicacion_base[0], ubicacion_base[1], nodo_medio[0], nodo_medio[1]), 2)
            g.add_edge(0, ubicaciones.index(nodo_medio), dist)

    # Añadir aristas entre todos los demás nodos
    for i in range(1, len(ubicaciones)):
        for j in range(i + 1, len(ubicaciones)):
            dist = round(haversine(ubicaciones[i][0], ubicaciones[i][1], ubicaciones[j][0], ubicaciones[j][1]), 2)
            g.add_edge(i, j, dist)

    mst_edges = g.prim_algo()

    G = nx.Graph()

    for i, ubicacion in enumerate(ubicaciones):
        G.add_node(i, pos=(ubicacion[1], ubicacion[0]))

    for i in range(len(ubicaciones)):
        for j in range(i + 1, len(ubicaciones)):
            dist = round(haversine(ubicaciones[i][0], ubicaciones[i][1], ubicaciones[j][0], ubicaciones[j][1]), 2)
            G.add_edge(i, j, weight=dist)

    MST = nx.Graph()
    mst_total_weight = 0.0
    for edge in mst_edges:
        u, v, w = edge
        MST.add_edge(u, v, weight=w)
        mst_total_weight += w

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    node_pos = nx.spring_layout(G, seed=42)  # Semilla para layout consistente
    nx.draw(G, pos=node_pos, with_labels=True, node_size=20, edge_color='blue')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=node_pos, edge_labels=labels)
    plt.title('Grafo Original')

    plt.subplot(1, 2, 2)
    nx.draw(MST, pos=node_pos, with_labels=True, node_size=20, edge_color='red')
    mst_labels = nx.get_edge_attributes(MST, 'weight')
    nx.draw_networkx_edge_labels(MST, pos=node_pos, edge_labels=mst_labels)
    plt.title('MST de Prim\nPeso total: {:.2f} km'.format(mst_total_weight))

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python PrimNetwork.py <num_filas>")
    else:
        num_filas = int(sys.argv[1])
        main(num_filas)
