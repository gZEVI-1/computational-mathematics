import numpy as np
import sympy as sp
from sympy import (
    symbols,
    sin,
    simplify,
    diff,
    integrate,
    pi,
)
from scipy import integrate as sci_integrate
import matplotlib.pyplot as plt


import time

seed = int(time.time() * 1000000) % 10**9
print(f"\n random seed: {seed}\n")
np.random.seed(seed)


def task_1():
    # ЗАДАНИЕ 1: матрица 7x7, транспонирование, определитель
    print("ЗАДАНИЕ 1: Матрица 7x7 ")

    matrix = np.random.randint(0, 2, size=(7, 7))
    print("Исходная матрица 7x7:")
    print(matrix)

    matrix_T = matrix.T
    print("\nТранспонированная матрица:")
    print(matrix_T)
    
    det = np.linalg.det(matrix)
    print(f"\nОпределитель исходной матрицы: {det:.6f}")


def task_2():
    # ЗАДАНИЕ 2:  умножение двух матриц
    print("ЗАДАНИЕ 2: Умножение матриц ")

    A = np.random.randint(0, 11, size=(3, 4))
    B = np.random.randint(0, 11, size=(4, 5))

    print("Матрица A :")
    print(A)
    print("\nМатрица B :")
    print(B)

    C = np.dot(A, B)
    print("\nРезультат умножения A x B:")
    print(C)


def task_3():
    # ЗАДАНИЕ 3: упрощение выражения и вычисление значения
    print("ЗАДАНИЕ 3: Упрощение выражения ")

    x, y = symbols("x y")

    func = (7 * x * y / 4) * (x + y) - (x - y) ** 2

    print("Исходное выражение:")
    print(f"(7xy/4)(x+y) - (x-y)² = {func}")

    S_func = simplify(func)
    print(f"\nУпрощённое выражение: {S_func}")
    print(f"Развёрнутая форма: {sp.expand(func)}\n")

   

    x_val = -1.23
    y_val = np.sqrt(8)

    # result_sympy = S_func.subs([(x, Rational(-123, 100)), (y, sqrt(8))])
    # result_numeric = float(result_sympy.evalf())
    # print(result_numeric)

    result_sympy = S_func.subs([(x, x_val), (y, y_val)])
    print(f"\nПри x = -1.23, y = √8 :")
    print(f"Значение выражения = {result_sympy}")


def task_4():
    # ЗАДАНИЕ 4: частные производные
    print("ЗАДАНИЕ 4: Частные производные ")

    x, y = symbols("x y")
    func = (7 * x * y / 4) * (x + y) - (x - y) ** 2

    df_dx = diff(func, x)
    print(f"Выражение: {func}")
    print(f"\nЧастная производная по x: {simplify(df_dx)}")
    print(f"Развёрнутая форма: {sp.expand(df_dx)}")

    df_dy = diff(func, y)
    print(f"\nЧастная производная по y: {simplify(df_dy)}")
    print(f"Развёрнутая форма: {sp.expand(df_dy)}")

    print(f"\nВторая производная по x: {simplify(diff(func, x, 2))}")
    print(f"Вторая производная по y: {simplify(diff(func, y, 2))}")
    print(f"Смешанная производная: {simplify(diff(func, x, y))}")


def task_5():
    # ЗАДАНИЕ 5: Решение системы уравнений
    print("ЗАДАНИЕ 5: Решение системы уравнений")

    # Система:
    # 3x1 + 2x2 + x3 = 5
    # 3x1 + 3x2 + 2x3 = 7
    # 5x1 + 5x2 + 3x3 = 11

    print("Система уравнений:")
    print("  3x1 + 2x2 + x3 = 5")
    print("  3x1 + 3x2 + 2x3 = 7")
    print("  5x1 + 5x2 + 3x3 = 11")

    print("\n \t\tСпособ 1: NumPy\n")
    A_np = np.array([[3, 2, 1], [3, 3, 2], [5, 5, 3]], dtype=float)
    b_np = np.array([5, 7, 11], dtype=float)

    solution_np = np.linalg.solve(A_np, b_np)
    print(
        f"Решение: x1 = {solution_np[0]:.6f}, x2 = {solution_np[1]:.6f}, x3 = {solution_np[2]:.6f}"
    )

    check_arr = np.dot(A_np, solution_np) - b_np
    print(
        f"Проверка (A·x - b): {check_arr[0]:.6f}, {check_arr[1]:.6f}, {check_arr[2]:.6f}"
    )

    print("\n \t\tСпособ 2: SymPy\n")
    x1, x2, x3 = symbols("x1 x2 x3")

    eq1 = sp.Eq(3 * x1 + 2 * x2 + x3, 5)
    eq2 = sp.Eq(3 * x1 + 3 * x2 + 2 * x3, 7)
    eq3 = sp.Eq(5 * x1 + 5 * x2 + 3 * x3, 11)

    solution_sympy = sp.solve([eq1, eq2, eq3], [x1, x2, x3])
    print(
        f"Решение: x1 = {solution_sympy[x1]}, x2 = {solution_sympy[x2]}, x3 = {solution_sympy[x3]}"
    )


def task_6():
    # ЗАДАНИЕ 6: Вычисление интеграла
    print("ЗАДАНИЕ 6: Интеграл ")

    x = symbols("x")
    expr = 1 / (1 + 2 * sin(x) ** 2)

    def func(x):
        return 1 / (1 + 2 * sin(x) ** 2)

    print("\n \t\tСпособ 1: SciPy\n")

    result_scipy, error_scipy = sci_integrate.quad(func, 0, np.pi / 4)
    print(f"Численное значение: {result_scipy:.10f}")
    print(f"Оценка ошибки: {error_scipy:.2e}")

    print("\n \t\tСпособ 2: SymPy \n")

    # integral_indef = integrate(expr, x)
    # print(f"Неопределённый интеграл: {integral_indef}")
    integral_def = integrate(expr, (x, 0, pi / 4))
    # print(f"\nОпределённый интеграл: {integral_def}")

    print(f"Численное значение: {float(integral_def.evalf()):.10f}")


def task_7():
    # ЗАДАНИЕ 7: Двойной интеграл
    print("ЗАДАНИЕ 7: Двойной интеграл ")

    print("\n \t\tСпособ 1: SciPy\n")

    def IN_func(x, y):
        return y**2

    result_scipy, error_scipy = sci_integrate.dblquad(
        IN_func,
        -1,
        2,
        lambda y: y**2,
        lambda y: y + 2,
    )

    print(f"Результат: {result_scipy:.10f}")
    print(f"Оценка ошибки: {error_scipy:.2e}")

    print("\n \t\tСпособ 2: SymPy \n")
    x, y = symbols("x y", real=True)

    inner = integrate(y**2, (x, y**2, y + 2))
    print(f"Внутренний интеграл: {sp.expand(inner)}")

    outer = integrate(inner, (y, -1, 2))
    print(f"\nРезультат: {outer} = {float(outer.evalf()):.10f}")

    print("\n \t\tСпособ 3: Ручная подстановка \n")
    val_at_2 = 2**4 / 4 + 2 * 2**3 / 3 - 2**5 / 5
    val_at_minus1 = (-1) ** 4 / 4 + 2 * (-1) ** 3 / 3 - (-1) ** 5 / 5

    print(f"При y=2: {val_at_2} = {float(val_at_2):.6f}")
    print(f"При y=-1: {val_at_minus1} = {float(val_at_minus1):.6f}")
    print(
        f"\nРезультат: {val_at_2 - val_at_minus1} = {float(val_at_2 - val_at_minus1):.10f}"
    )


def task_8():
    # ЗАДАНИЕ 8: Построение графиков функций

    print("ЗАДАНИЕ 8: Построение графиков функций")

    x1 = np.linspace(-5, 5, 10000)
    y1 = 1 - np.cos(x1)

    x2 = np.linspace(-5, 3, 8000)
    y2 = np.sqrt(3 - x2)

    import scipy.optimize as scp

    # y = 1 - cos(x)
    # y = √(3 - x)
    def eq(coords):
        x, y = coords
        eq1 = y - (1 - np.cos(x))
        eq2 = y - np.sqrt(3 - x)
        return [eq1, eq2]

    solution = scp.fsolve(eq, [1, 0], full_output=1)
    sol_x, sol_y = solution[0]

    print(f"\nТочка пересечения: x = {sol_x:.6f}, y = {sol_y:.6f}")

    # print(f"Проверка: 1 - cos({intersection_x:.6f}) = {1 - np.cos(intersection_x):.6f}")
    # print(f"Проверка: √({3 - intersection_x:.6f}) = {np.sqrt(3 - intersection_x):.6f}")

    fig, axes = plt.subplots(figsize=(16, 9))

    axes.plot(x1, y1, "r-", linewidth=1, label=r"$y = 1 - \cos(x)$")
    axes.plot(x2, y2, "y-", linewidth=1, label=r"$y = \sqrt{3 - x}$")

    axes.plot(
        sol_x,
        sol_y,
        "o",
        color="orange",
        markersize=7,
        label=f"Пересечение ({sol_x:.3f}, {sol_y:.3f})",
    )

    # Оформление графика
    axes.axhline(y=0, color="k", linewidth=1.5)
    axes.axvline(x=0, color="k", linewidth=1.5)
    axes.grid(True, alpha=0.5)
    axes.set_xlabel("x", fontsize=20, loc="left")
    axes.set_ylabel("y", fontsize=20, loc="bottom")
    axes.legend(loc="upper right", fontsize=10)

    axes.set_xlim(-5, 5)
    axes.set_ylim(-0.1, 3)

    # plt.tight_layout()
    plt.savefig("task8_graph.png", dpi=100, bbox_inches="tight")
    # plt.show()

    print("\nГрафик сохранён!")


def print_separator():
    print("\n" + "+-" * 60 + "\n")


def main():
    print_separator()
    task_1()
    print_separator()
    task_2()
    print_separator()
    task_3()
    print_separator()
    task_4()
    print_separator()
    task_5()
    print_separator()
    task_6()
    print_separator()
    task_7()
    print_separator()
    task_8()
    print_separator()


if __name__ == "__main__":
    main()
