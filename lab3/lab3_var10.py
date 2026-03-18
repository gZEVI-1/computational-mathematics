import numpy as np


import time
def rand():
    seed = int(time.time() * 1000000) % 10**9
    print(f"\n random seed: {seed}\n")
    np.random.seed(seed)

def PS(msg: str|None=None):
    print( "+-" * 60  )
    if msg is not None:
        print( msg)
        print( "+-" * 60 )

def give_me_vector():
    return np.random.randint(0, 10, (1, 7))

def give_me_matrix():
    return np.random.randint(-5, 6, (10, 10))

def task1(vector):
    
    print("\nВектор-строка:", vector)

    def custom_norm(vector):
        return np.sum(np.abs(vector) ** 3) ** (1/3)

    norm_custom = custom_norm(vector)
    norm_numpy = np.linalg.norm(vector[0], ord=3)

    print("Норма 3 (самостоятельная функция):", norm_custom)
    print("Норма 3 (numpy):", norm_numpy, "\n")

def task2(matrix):
    print("\nИсходная матрица 10x10:")
    print(matrix)   

    def custom_inf_norm(matrix):
        abs_matrix = np.abs(matrix)
        
        my_result = 0
        for i in range(matrix.shape[0]):
            row = matrix[i]
            abs_row = np.abs(row)
            row_sum = np.sum(abs_row)
            my_result = max(my_result, row_sum) 

        return my_result

    my_norm = custom_inf_norm(matrix)
    norm_numpy = np.linalg.norm(matrix, ord=np.inf)

    print("Норма inf (самостоятельная функция):", my_norm)
    print("Норма inf (numpy):", norm_numpy, "\n")

def task3(x):
    # Для вектора, созданного в первом пункте, найти отражение 
    # Хаусхолдера, которое обнуляет координтаы, начиная с третьей 
    # Например, вектор (1,2,3,4,5,6,7) переводит  в вектор (1,2,0,0,0, 0,0) 
    
    print("\nВектор-строка:", x)
    

    print ('\n')

def main():
    PS()
    rand()
    vector = give_me_vector()
    matrix = give_me_matrix()
    PS("Задание 1:")
    task1(vector)
    PS("Задание 2:")
    task2(matrix)
    PS("Задание 3:")
    task3(vector)
    #PS("Задание 4:")
    #task4()
    PS()

if __name__ == "__main__":   
    main()