import numpy as np
import sympy as sp
from sympy import (symbols,sin,simplify,diff,integrate,pi,)
from scipy import integrate as sci_integrate
import matplotlib.pyplot as plt

import time
seed = int(time.time() * 1000000) % 10**9
print(f"\n random seed: {seed}\n")
np.random.seed(seed)


def task1():
    matrix = np.random.randint(-5, 6, (7, 7))
    print("Исходная матрица 7x7:")
    print(matrix)

    M3_7=matrix[4:, :]
    M7_2=matrix[:,:2]
    print('\n Три последние строки:\n', M3_7)
    print('\n Два первых столбца:\n', M7_2)
    
    return M3_7, M7_2

def task2(A,B):
    print("\nСпособ 1: ВЕКТОРНЫЙ АЛГОРИТМ\n")
    

    m, n = A.shape  # 3, 7
    n2, p = B.shape  # 7, 2

    result_vector = np.zeros((m, p), dtype=int)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result_vector[i, j] += A[i, k] * B[k, j]

    print(result_vector)

    print("\nСпособ 2: МАТРИЧНЫЙ АЛГОРИТМ \n")

    result_matrix = np.zeros((m, p), dtype=int)
    for i in range(m):
        result_matrix[i, :] = A[i, :] @ B

    
    print(result_matrix)

    print("\nСпособ 3: ПРОВЕРКА ЧЕРЕЗ np.dot\n")

    result_dot = np.dot(A, B)
    print(result_dot)


def task3( ):
    A = np.random.randint(-5, 6, (4, 4))
    print("Исходная матрица 4x4:")
    print(A)
    A=np.triu(A)
    print("\nВерхнетреугольная матрица 4x4:")
    print(A)
    B = np.random.randint(-5, 6, 4)
    
    try:
        X = np.linalg.solve(A, B)
        print(f"Решение: \nx1 = {X[0]:.6f}, \nx2 = {X[1]:.6f}, \nx3 = {X[2]:.6f}, \nx4 = {X[3]:.6f}")
    except np.linalg.LinAlgError:
        print("Матрица вырождена, решение невозможно.")

def task4():
    return 0

def PS():
    print("\n" + "+-" * 60 + "\n")

def main():
    PS()
    A,B=task1()
    PS()
    task2(A,B)
    PS()
    task3()
    PS()
    task4()
    PS()



if __name__=="__main__":
    main()