import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import sys
from collections import defaultdict

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)

    def add_edge(self, src, dest, weight):
        new_edge = [dest, weight]
        self.graph[src].insert(0, new_edge)
        new_edge = [src, weight]
        self.graph[dest].insert(0, new_edge)

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
            for neighbor in self.graph[u]:
                v, weight = neighbor
                if not mst_set[v] and weight < key[v]:
                    key[v] = weight
                    parent[v] = u

        for i in range(1, self.V):
            mst_edges.append((parent[i], i, key[i]))

        return mst_edges

def cargar_y_procesar_datos(archivo, num_filas):
    ubicacion_base = [-24.1858, -65.2992]

    datos = pd.read_csv(archivo, nrows=num_filas)
    datos[['longitud', 'latitud']] = datos['latitud_y_longitud'].str.strip(' "').str.split(',', expand=True)
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

    if nodos_alta:
        for nodo_alta in nodos_alta:
            dist = haversine(ubicacion_base[0], ubicacion_base[1], nodo_alta[0], nodo_alta[1])
            g.add_edge(0, ubicaciones.index(nodo_alta), dist)
    else:
        for nodo_medio in nodos_medio:
            dist = haversine(ubicacion_base[0], ubicacion_base[1], nodo_medio[0], nodo_medio[1])
            g.add_edge(0, ubicaciones.index(nodo_medio), dist)

    for i in range(1, len(ubicaciones)):
        for j in range(i + 1, len(ubicaciones)):
            dist = haversine(ubicaciones[i][0], ubicaciones[i][1], ubicaciones[j][0], ubicaciones[j][1])
            g.add_edge(i, j, dist)

    return g, ubicaciones

def main(num_filas):

    ubicacion_base = [-24.1858, -65.2992]  

    archivo = 'dataset-jujuy.csv'  
    datos = pd.read_csv(archivo, nrows=num_filas)

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

    if nodos_alta:
        for nodo_alta in nodos_alta:
            dist = haversine(ubicacion_base[0], ubicacion_base[1], nodo_alta[0], nodo_alta[1])
            g.add_edge(0, ubicaciones.index(nodo_alta), dist)
    else:
        for nodo_medio in nodos_medio:
            dist = haversine(ubicacion_base[0], ubicacion_base[1], nodo_medio[0], nodo_medio[1])
            g.add_edge(0, ubicaciones.index(nodo_medio), dist)

    for i in range(1, len(ubicaciones)):
        for j in range(i + 1, len(ubicaciones)):
            dist = haversine(ubicaciones[i][0], ubicaciones[i][1], ubicaciones[j][0], ubicaciones[j][1])
            g.add_edge(i, j, dist)
            
    mst_edges = g.prim_algo()

    G = nx.Graph()

    for i, ubicacion in enumerate(ubicaciones):
        G.add_node(i, pos=(ubicacion[1], ubicacion[0]))

    for u in range(g.V):
        for neighbor in g.graph[u]:
            v, w = neighbor
            if w > 0:
                G.add_edge(u, v, weight=w)

    mst_total_weight = sum(w for u, v, w in mst_edges)

    plt.figure(figsize=(14, 7))

    print("Lista de adyacencia:")
    for u in range(g.V):
        print(f"{u}: {g.graph[u]}")

    plt.subplot(1, 2, 1)
    node_pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos=node_pos, with_labels=True, node_size=20, edge_color='blue')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=node_pos, edge_labels=labels)
    plt.title('Grafo Original')

    plt.subplot(1, 2, 2)
    MST = nx.Graph()
    for edge in mst_edges:
        u, v, w = edge
        MST.add_edge(u, v, weight=w)
    nx.draw(MST, pos=node_pos, with_labels=True, node_size=20, edge_color='red')
    mst_labels = nx.get_edge_attributes(MST, 'weight')
    nx.draw_networkx_edge_labels(MST, pos=node_pos, edge_labels=mst_labels)
    plt.title('MST de Prim\nPeso total: {:.6f} km'.format(mst_total_weight))
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python PrimNetwork.py <num_filas>")
    else:
        num_filas = int(sys.argv[1])
        main(num_filas)
