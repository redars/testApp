# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import random
import sys
import numpy as np
        
def DrawNetArea(array : np.array): # Отрисовка зоны действия вышек
    plt.axes()
    width = len(array)-1
    i = width
    j = 0
    while i >= 0:
        while j <=len(array[0])-1:
            if array[i][j] == 1:
                rectangle = plt.Rectangle((j,width-i), 1, 1, fc='green',ec="black")
                plt.gca().add_patch(rectangle)
            if array[i][j] == 0:
                rectangle = plt.Rectangle((j,width-i), 1, 1, fc='red',ec="black")
                plt.gca().add_patch(rectangle)
            j +=1
        i-=1
        j = 0
    plt.axis('scaled')
    frame1 = plt.gca()
    for pos in ['right', 'top', 'bottom', 'left']: 
        frame1.spines[pos].set_visible(False) 
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    frame1.set_title("Network coverage")
    plt.show()
    
def DrawGrid(array : np.array): # Отрисовка местности. Красный - нельзя установить вышку; Зеленый - можно установить; Желтый - вышка установлена
    plt.axes()
    width = len(array)-1
    i = width
    j = 0
    while i >= 0:
        while j <=len(array[0])-1:
            if array[i][j] == -1:
             rectangle = plt.Rectangle((j,width-i), 1, 1, fc='red',ec="black")
             plt.gca().add_patch(rectangle)
            if array[i][j] == 0:
                rectangle = plt.Rectangle((j,width-i), 1, 1, fc='green',ec="black")
                plt.gca().add_patch(rectangle)
            if array[i][j] > 0:
                rectangle = plt.Rectangle((j,width-i), 1, 1, fc='yellow',ec="black")
                plt.gca().add_patch(rectangle)
                plt.annotate(f'{array[i][j]}', (j+0.5,width-i+0.5), color='black', fontweight='book', fontsize=12, ha='center', va='center')
            j +=1
        i-=1
        j = 0
    plt.axis('scaled')
    frame1 = plt.gca()
    for pos in ['right', 'top', 'bottom', 'left']: 
        frame1.spines[pos].set_visible(False) 
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    frame1.set_title("City grid")
    plt.show()

def ValidateGrid(array : np.array) -> np.array: # Метод для выявления зоны действия вышек
    result = np.zeros(array.shape)
    for i in range(0,len(array)):
        for j in range(0,len(array[0])):
            if array[i][j] > 0:
                temp_j = j - array[i][j]
                if temp_j < 0:
                    temp_j = 0
                while temp_j < len(array[0]) and temp_j <= j + array[i][j]:
                    temp_i = i - array[i][j]
                    if temp_i < 0:
                        temp_i = 0
                    while temp_i < len(array) and temp_i <= i + array[i][j]:
                        result[temp_i][temp_j] = 1
                        temp_i+=1
                    temp_j+=1
    return result

def RandomChoice(outputs : list, choice_count : int, probs : list) -> np.array :
    if len(outputs) != len(probs):
        pass
    if sum(probs) != 1:
        pass
    result = []
    for i in range(0,choice_count):
        res = outputs[random.randint(0,(len(probs)-1))]
        ind = outputs.index(res)
        while result.count(res) >= probs[ind] * choice_count:
            res = outputs[random.randint(0,len(probs)-1)]
            ind = outputs.index(res)
        result.append(res)
    return np.array(result)

def NormalizeMatrix(matrix : np.array , axs : int) -> np.array : # normalize adjacency matrix
    for i in range(0,axs):
        for j in range (0,axs):
            if matrix[i][j] == 1:
                matrix[j][i] = 1
    return matrix

def GetTowersMatrix(towers : list) -> np.array : # get adjacency matrix
    towersL = len(towers)
    result = np.zeros((towersL,towersL))
    n = 0
    for tower in towers:
        i,j,rng = tower
        n2 = 0
        for item in towers:
            i2,j2,rng2 = item
            if i2 >= i - rng and i2 <= i + rng:
                if j2 >= j - rng and j2 <= j + rng:
                    result[n][n2] = 1
            n2 +=1
        n +=1
    result = NormalizeMatrix(result, len(towers))
    return result

def Dijkstra(graph : np.array , start : int , end : int): # Реализация алгоритма Дейкстры
    num_vertices = len(graph)
    visited = [False] * num_vertices
    distance = [sys.maxsize] * num_vertices
    parent = [-1] * num_vertices
    
    distance[start] = 0
    for _ in range(num_vertices):
        min_distance = sys.maxsize
        for i in range(num_vertices):
            if not visited[i] and distance[i] < min_distance:
                min_distance = distance[i]
                u = i
        visited[u] = True
        for v in range(num_vertices):
            if graph[u][v] > 0 and not visited[v] and distance[v] > distance[u] + graph[u][v]:
                distance[v] = distance[u] + graph[u][v]
                parent[v] = u
    path = []
    curr = end
    while curr != -1:
        path.insert(0, curr)
        curr = parent[curr]
    return (path, distance[end])
                
class CityGrid:
    def __init__(self, N : int, M : int,obstructed : float = 0.5):
        self.grid = RandomChoice([-1, 0], N * M, [obstructed, 1-obstructed]).reshape(N, M)
        self.towers = []
        
    def PlaceTower(self, rng : int, i : int, j : int):
        if self.grid[i][j] == 0:
            self.grid[i][j] = rng
            self.towers.append((i,j,rng))
            print(f"Tower placed at position ({j},{i})")
        else:
            print(f"A tower can't be placed at position ({j},{i})")
    
    def GetGridShape(self):
        return self.grid.shape
    
    def SetBudget(self,budget):
        self.budget = budget
        
    def ShowGrid(self):
        DrawGrid(self.grid)
        
    def SmartPlaceTowers(self):
        budget = self.budget
        print(budget)
        pass
    
    def RandomPlaceTowers(self, try_num : int):
        for i in range(0,try_num):
            max_range = int(max(self.GetGridShape())/3)
            c.PlaceTower(random.randint(1, max_range), random.randint(0,len(self.grid)-1), random.randint(0,len(self.grid[0])-1))
            
    def PlaceMinTowers(self): # Наименьшим количеством вышек будет 1, которая покрывает все поле (ее зона действия равна большей грани)
        width, height = self.GetGridShape()
        lenght = width * height
        if width > height:
            while sum(map(sum, ValidateGrid(self.grid))) != lenght:
                self.PlaceTower(width-1, random.randint(0,width-1), random.randint(0,height))
        else:
            while sum(map(sum, ValidateGrid(self.grid))) != lenght:
                self.PlaceTower(height-1, random.randint(0,width-1), random.randint(0,height))
    
    def GetNetArea(self):
        return ValidateGrid(self.grid)
    
    def ShowNetArea(self):
        DrawNetArea(self.GetNetArea())
    
    def FindBestWay(self, t1 : int , t2 : int): # Нахождения наименьшего пути между вышками
        i1,j1, rng = self.towers[t1]
        i2,j2,rng = self.towers[t2]
        print(f"Finding best path between {(j1,i1)} and {(j2,i2)}")
        adj_matrix = GetTowersMatrix(self.towers)
        path,dis = Dijkstra(adj_matrix,t1,t2)
        res_path = []
        for el in path:
            tw = self.towers[el]
            loc, lc, rng = tw
            res_path.append((loc,lc))
        print("Best path: "+str(res_path))
        return res_path

c = CityGrid(10, 20)
c.ShowGrid()
c.ShowNetArea()
#c.PlaceMinTowers()
c.RandomPlaceTowers(40)
c.ShowGrid()
c.ShowNetArea()
c.FindBestWay(0,1)