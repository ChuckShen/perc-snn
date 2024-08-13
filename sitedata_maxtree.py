import string, time
from random import random
import numpy as np
import multiprocessing, sys, os
import matplotlib.pyplot as plt
import matplotlib

sys.setrecursionlimit(64 * 64 * 1000)

plist = [0.0 + x*0.01 for x in range(101)]
NOT_VISITED = 1
t = 1000
N = 40
data_path = '2D_percolation/' + str(N) + '/'

def grid(p):
    cell = np.random.random([t, N, N])
    grid = cell < p
    return grid.astype(int)

def maxtree(grid):
    max_grid = np.zeros(np.shape(grid))
    for j in range(N):
        for i in range(N):
            new_grid = np.zeros(np.shape(grid))
            if grid[j][i] == NOT_VISITED:
                dfs(grid, new_grid, j, i)
            if np.sum(max_grid) < np.sum(new_grid):
                max_grid = new_grid
    return max_grid.astype(int)

def dfs(grid, new_grid, i, j):
    grid[i][j] = 0
    new_grid[i][j] = 1
    if i > 0 and grid[i - 1][j] == NOT_VISITED:
        dfs(grid, new_grid, i - 1, j)
    if i < N - 1 and grid[i + 1][j] == NOT_VISITED:
        dfs(grid, new_grid, i + 1, j)
    if j < N - 1 and grid[i][j + 1] == NOT_VISITED:
        dfs(grid, new_grid, i, j + 1)
    if j > 0 and grid[i][j - 1] == NOT_VISITED:
        dfs(grid, new_grid, i, j - 1)

def save_to_npy(data, p):
    file_name = data_path + f'{p:.2f}.npy'
    np.save(file_name, data)
    print(f'Saved: {file_name}')

if __name__ == '__main__':
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    for p in plist:
        pool = multiprocessing.Pool(30)
        cell = grid(p)
        result = []

        for i in range(t):
            try:
                new_cell = pool.apply(maxtree, (cell[i],))
            except Exception as e:
                print(f"Exception occurred: {e}")
                new_cell = maxtree(cell[i])
            result.append(new_cell)

        pool.close()
        pool.join()

        result = np.array(result)
        save_to_npy(result, p)



   