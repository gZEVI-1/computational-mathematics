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
    print("Способ 1: ВЕКТОРНЫЙ АЛГОРИТМ\n")
    

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
    print("\nВектор-столбец B:")
    print(B)


    print("\n Решение системы AX=B с помощью обратной подстановки:")
    try:
        X=np.zeros(4)
        for i in range(3, -1, -1):
            if A[i, i] == 0:
                raise ZeroDivisionError
            X[i] = (B[i] - np.dot(A[i, i+1:], X[i+1:])) / A[i, i]
        
        print(f"Решение: \nx1 = {X[0]:.6f}, \nx2 = {X[1]:.6f}, \nx3 = {X[2]:.6f}, \nx4 = {X[3]:.6f}")
    except ZeroDivisionError or RuntimeWarning:
        print("Матрица вырождена, решение невозможно.")

   
    print("\n Проверка решения системы AX=B с помощью np.linalg.solve:")
    try:
        X = np.linalg.solve(A, B)
        print(f"Решение: \nx1 = {X[0]:.6f}, \nx2 = {X[1]:.6f}, \nx3 = {X[2]:.6f}, \nx4 = {X[3]:.6f}")
    except np.linalg.LinAlgError:
        print("Матрица вырождена, решение невозможно.")

def LU_(A):
    A = A.astype(float)
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    for i in range(n):
        
        for k in range(i, n):
            s = 0.0
            for j in range(i):
                s += L[i, j] * U[j, k]
            U[i, k] = A[i, k] - s

        
        L[i, i] = 1.0
        for k in range(i + 1, n):
            s = 0.0
            for j in range(i):
                s += L[k, j] * U[j, i]
            if U[i, i] == 0:
                raise ZeroDivisionError()
            L[k, i] = (A[k, i] - s) / U[i, i]

    return L, U

def forward(L, b):
    
    n = L.shape[0]
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = b[i] - s
    return y

def backward(U, y):
    
    n = U.shape[0]
    x = np.zeros_like(y, dtype=float)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
        if U[i, i] == 0:
            raise ZeroDivisionError()
        x[i] = (y[i] - s) / U[i, i]
    return x

def lu_solve(A, b):
    
    L, U = LU_(A)
    y = forward(L, b) #Решает L y = b 
    x = backward(U, y) #Решает U x = y
    return x

def task4():
    # 6.1x1 + 6.2x2 - 6.3x3 + 6.4x4 = 6.5
    # 1.1x1 - 1.5x2 + 2.2x3 - 3.8x4 = 4.2
    # 5.1x1 - 5.0x2 + 4.9x3 - 4.8x4 = 4.7
    # 1.8x1 + 1.9x2 + 2.0x3 - 2.1x4 = 2.2

    A = np.array([
        [6.1, 6.2, -6.3, 6.4],
        [1.1, -1.5, 2.2, -3.8],
        [5.1, -5.0, 4.9, -4.8],
        [1.8, 1.9, 2.0, -2.1]
    ], dtype=float)

    B = np.array([6.5, 4.2, 4.7, 2.2], dtype=float)

    print("Матрица A:")
    print(A)
    print(f"\nВектор B: {B}")

    # LU-разложение (метод Гаусса с выбором главного элемента)
    
    L, U = LU_(A)


    print(f"\nНижняя треугольная L:\n{L}")
    print(f"\nВерхняя треугольная U:\n{U}")

    LU=L @ U
    print(f"\nПроверка  A = L @ U")
    print(f"Матрица A\t\t   Матрица L @ U")
    for rowa, rowlu in zip(A, LU):
        print(f"{rowa}  ==  {rowlu}")
    
    
    X = lu_solve(A, B)
    print(f"\nРешение системы AX=B:\nx1 = {X[0]:.6f}, \nx2 = {X[1]:.6f}, \nx3 = {X[2]:.6f}, \nx4 = {X[3]:.6f}")
    

def PS(msg: str|None=None):
    print( "+-" * 60  )
    if msg is not None:
        print('\n' + msg)
        print("\n" + "+-" * 60 )
    
def main():
    PS('ЗАДАНИЕ 1')
    A,B=task1()
    PS('ЗАДАНИЕ 2')
    task2(A,B)
    PS('ЗАДАНИЕ 3')
    task3()
    PS('ЗАДАНИЕ 4')
    task4()
    PS()

if __name__=="__main__":
    main()