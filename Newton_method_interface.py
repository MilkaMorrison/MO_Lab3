import math
from typing import List, Tuple, Dict
import numpy as np

def f(a: float, b: float, c: float, x1: float, x2: float) -> float:
    return a * x1 ** 2 + b * x1 * x2 + c * x2 ** 2

def gradient(a: float, b: float, c: float, x1: float, x2: float) -> List[float]:
    df_dx1 = 2 * a * x1 + b * x2
    df_dx2 = b * x1 + 2 * c * x2
    return [df_dx1, df_dx2]

def method(a: float, b: float, c: float, x: List[float],
           e1: float, e2: float, M: int) -> Tuple[Dict, List[List[float]]]:
    points = [x.copy()]
    prev_x = x.copy()

    for k in range(M):
        print(f"\nИтерация: {k}")
        grad = gradient(a, b, c, x[0], x[1])
        grad_norm = math.sqrt(pow(grad[0], 2) + pow(grad[1], 2))
        print(f"gradient(x) = ({grad[0]:.3f}, {grad[1]:.3f})")
        print(f"gradient_norm(x) = {grad_norm:.3f}")

        if grad_norm <= e1:
            print("\nОстановка: gradient_norm(x) <= e1")
            print(f"Найденная точка: ({x[0]:.3f}, {x[1]:.3f})")
            print(f"Значение функции: {f(a, b, c, x[0], x[1]):.3f}")
            print(f"Количество итераций: {k + 1}")
            return {
                'point': x,
                'value': f(a, b, c, x[0], x[1]),
                'iterations': k + 1
            }, points

        if k >= M - 1:
            print("\nОстановка: k = M")
            print(f"Найденная точка: ({x[0]:.3f}, {x[1]:.3f})")
            print(f"Значение функции: {f(a, b, c, x[0], x[1]):.3f}")
            print(f"Количество итераций: {k + 1}")
            return {
                'point': x,
                'value': f(a, b, c, x[0], x[1]),
                'iterations': k + 1
            }, points

        Hesse_mat = np.array([[2 * a, b], [b, 2 * c]])
        inverse_Hesse_mat = np.linalg.inv(Hesse_mat)

        det_inverse_Hesse_mat = inverse_Hesse_mat[0][0] * inverse_Hesse_mat[1][1] \
                                - inverse_Hesse_mat[0][1] * inverse_Hesse_mat[1][0]

        if det_inverse_Hesse_mat > 0:
            d = [-grad[0] * inverse_Hesse_mat[0][0] - grad[1] * inverse_Hesse_mat[0][1],
                -grad[0] * inverse_Hesse_mat[1][0] - grad[1] * inverse_Hesse_mat[1][1]]
            t = 1
            new_x = [x[0] + t * d[0], x[1] + t * d[1]]
        else:
            t = 1
            d = [-grad[0], -grad[1]]
            max_trials = 100
            for _ in range(max_trials):
                new_x = [x[0] + t * d[0], x[1] + t * d[1]]
                if f(a, b, c, new_x[0], new_x[1]) < f(a, b, c, x[0], x[1]):
                    break
                t *= 0.5
            else:
                print("Не удалось найти подходящий шаг t")
                return {
                    'point': x,
                    'value': f(a, b, c, x[0], x[1]),
                    'iterations': k + 1
                }, points

        print(f"d = ({d[0]:.3f}, {d[1]:.3f})")
        print(f"new_x = ({new_x[0]:.3f}, {new_x[1]:.3f})")

        delta_x = (pow(new_x[0] - x[0], 2) + pow(new_x[1] - x[1], 2)) ** 0.5
        delta_f = abs(f(a, b, c, new_x[0], new_x[1]) - f(a, b, c, x[0], x[1]))
        if k > 0 and delta_x < e2 and delta_f < e2:
            delta_x = (pow(x[0] - prev_x[0], 2) + pow(x[1] - prev_x[1], 2)) ** 0.5
            delta_f = abs(f(a, b, c, x[0], x[1]) - f(a, b, c, prev_x[0], prev_x[1]))
            if delta_x < e2 and delta_f < e2:
                print(f"Найденная точка: ({new_x[0]:.3f}, {new_x[1]:.3f})")
                print(f"Значение функции: {f(a, b, c, new_x[0], new_x[1]):.3f}")
                print(f"Количество итераций: {k + 1}")
                return {
                    'point': new_x,
                    'value': f(a, b, c, new_x[0], new_x[1]),
                    'iterations': k + 1
                }, points

        prev_x = x.copy()
        x = new_x.copy()
        points.append(x.copy())

    print(f"Найденная точка: ({x[0]:.3f}, {x[1]:.3f})")
    print(f"Значение функции: {f(a, b, c, x[0], x[1]):.3f}")
    print(f"Количество итераций: {M}")
    return {
        'point': x,
        'value': f(a, b, c, x[0], x[1]),
        'iterations': M,
        'message': 'Достигнуто максимальное количество итераций'
    }, points
