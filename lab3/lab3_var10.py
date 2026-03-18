import numpy as np
import time

Task4Matrix = np.array(
    [
        [6.1, 6.2, -6.3, 6.4],
        [1.1, -1.5, 2.2, -3.8],
        [5.1, -5.0, 4.9, -4.8],
        [1.8, 1.9, 2.0, -2.1],
    ]
)
Task4Array = np.array([6.5, 4.2, 4.7, 2.2], dtype=float).reshape(-1, 1)

np.set_printoptions(precision=2, suppress=True, floatmode="fixed")


def rand():
    seed = int(time.time() * 1000000) % 10**9
    # seed=123456789
    print(f"\n random seed: {seed}\n")
    np.random.seed(seed)


def PS(msg: str | None = None):
    print("+-" * 60)
    if msg is not None:
        print(msg)
        print("+-" * 60)


def give_me_vector():
    return np.random.randint(0, 10, (1, 7))


def give_me_matrix():
    return np.random.randint(-5, 6, (10, 10))


def task1(vector):

    print("\nВектор-строка:", vector)

    def custom_norm(vector):
        return np.sum(np.abs(vector) ** 3) ** (1 / 3)

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
    print(f"\nИсходный вектор x:\n{x.flatten()}")

    def householder_reflection(v, k):
        n = len(v)
        v = v.flatten().astype(float)
        x = v[k:].copy()

        alpha = np.linalg.norm(x)
        if alpha < 1e-10:
            return np.eye(n)

        u = x.copy()
        u[0] = u[0] - alpha

        norm_u = np.linalg.norm(u)
        if norm_u < 1e-10:
            return np.eye(n)
        u = u / norm_u

        H_sub = np.eye(len(x)) - 2 * np.outer(u, u)
        H = np.eye(n)
        H[k:, k:] = H_sub

        return H

    x = np.transpose(x)
    H = householder_reflection(x, 2)
    x_reflected = H @ x

    print(
        f"\nH @ x = [{ ''.join('{:.6f}, '.format(val) for val in x_reflected.flatten()) }]"
    )
    print(
        f"\nПроверка: (H^T @ H) - E = {np.max(np.abs(H.T @ H - np.eye(7))):.3e} = {np.max(np.abs(H.T @ H - np.eye(7))):.20f}\n"
    )


def qr_householder(A):

    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)

    for k in range(min(m, n)):
        print(f"\n--- обработка столбца {k} ---")

        x = R[k:, k].copy()
        print(f"Подстолбец x = R[{k}:, {k}]:\n{x}\n")

        norm_x = np.linalg.norm(x)
        print(f"Норма ||x|| = {norm_x:.6f}")

        if norm_x < 1e-10:
            print("Норма слишком мала, пропускаем")
            continue

        e1 = np.zeros_like(x)
        e1[0] = 1
        alpha = -np.sign(x[0]) * norm_x if abs(x[0]) > 1e-10 else -norm_x
        u = x - alpha * e1

        norm_u = np.linalg.norm(u)
        if norm_u > 1e-10:
            u = u / norm_u

        print(f"Вектор u{k} (нормированный):\n{u}\n")

        H_k = np.eye(m)
        H_sub = np.eye(len(x)) - 2 * np.outer(u, u)
        H_k[k:, k:] = H_sub

        R = H_k @ R
        Q = Q @ H_k.T

        print(f"После применения H_{k+1}:")
        print(f"R[{k}:, {k}] = {R[k, k]:.6f}")
        print(f"R[{k+1}:, {k}] ≈ {R[k+1:, k]}")
    print(f"\n------------------------------")

    return Q, R


def task4(A, b):

    Q, R = qr_householder(A)

    print("РЕЗУЛЬТАТЫ QR-РАЗЛОЖЕНИЯ")

    print(f"\nМатрица Q :\n{Q}")
    print(f"\nМатрица R :\n{R}")

    A_reconstructed = Q @ R
    print(f"\nQ · R =\n{A_reconstructed}")
    print(f"\nИсходная матрица A =\n{A}")
    print(f"\nРазница ||Q·R - A|| = {np.linalg.norm(A_reconstructed - A):.2e}")

    print(f"\nПроверка ортогональности Q:")
    print(f"Q^T · Q =\n{Q.T @ Q}")
    print(f"||Q^T·Q - I|| = {np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0])):.2e}")

    print(f"\nПроверка, что R верхняя треугольная:")

    print(f"Нижняя часть R (должна быть 0):\n{R}")
    print(f"||нижняя часть R|| = {np.linalg.norm(np.tril(R, -1)):.2e}")
    return Q, R


def check_task4_solution(A, b, Q, R):
    print("\nРЕШЕНИЕ СИСТЕМЫ : RX = Q^T·b")

    Q_T_b = Q.T @ b
    print(f"\nQ^T · b =\n{Q_T_b}")

    U = R
    y = Q_T_b

    print(f"\nВерхняя треугольная матрица R:\n{U}")
    print(f"\nQ^T·b:\n{y}")

    #  (из лабораторной работы 2)
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

    x_solution = backward(U, y)
    print(f"\nРешение x :\n{x_solution}")

    x_numpy = np.linalg.solve(U, y)
    print(f"\nРешение x (.solve):\n{x_numpy}")

    print(f"\nРазница между методами: {np.linalg.norm(x_solution - x_numpy):.2e}")

    print("ПРОВЕРКА РЕШЕНИЯ")

    residual = A @ x_solution - b
    print(f"\nНевязка A·x - b:\n{residual}")
    print(f"Норма невязки ||A·x - b||_2 = {np.linalg.norm(residual):.6f}")

    print("СРАВНЕНИЕ С ВСТРОЕННЫМ QR-РАЗЛОЖЕНИЕМ NUMPY")

    Q_np, R_np = np.linalg.qr(A, mode="complete")
    print(f"\nQ из numpy.linalg.qr:\n{Q_np}")
    print(f"\nнаш Q:\n{Q}")
    print(f"\nR из numpy.linalg.qr:\n{R_np}")
    print(f"\nнаш R:\n{R}")

    print(f"\nПроверка Q·R:\n{Q_np @ R_np}")
    print(f"\nИсходная матрица A:\n{A}")
    print(f"Разница с исходной матрицей: {np.linalg.norm(Q_np @ R_np - A):.2e}")


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
    PS("Задание 4:")
    A = Task4Matrix
    b = Task4Array
    # print("\nИсходная матрица 4x4:")
    # print(A)
    # print("\nВектор правой части :")
    # print(b.T)
    Q, R = task4(A, b)
    print("хотите проверить решение дополнительно? (y/n)")
    if input().lower() == "y":
        PS("Проверка решения задания 4:")
        check_task4_solution(A, b, Q, R)
    PS()


if __name__ == "__main__":
    main()
