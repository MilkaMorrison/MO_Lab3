from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QComboBox, QTextEdit, QGroupBox,
                             QHBoxLayout, QPushButton, QLineEdit)
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib
from Newton_method_interface import method, f

class InterfaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Метод Ньютона")
        self.resize(900, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        func_group = QGroupBox("Параметры функции")
        func_layout = QVBoxLayout()
        self.a_input = self.create_input("Коэффициент a:")
        self.b_input = self.create_input("Коэффициент b:")
        self.c_input = self.create_input("Коэффициент c:")
        func_layout.addWidget(self.a_input)
        func_layout.addWidget(self.b_input)
        func_layout.addWidget(self.c_input)
        func_group.setLayout(func_layout)

        algo_group = QGroupBox("Параметры алгоритма")
        algo_layout = QVBoxLayout()
        self.x_input = self.create_input("Начальный вектор x:")
        self.e1_input = self.create_input("Точность e1:")
        self.e2_input = self.create_input("Точность e2:")
        self.M_input = self.create_input("Максимальное число итераций M:")
        algo_layout.addWidget(self.x_input)
        algo_layout.addWidget(self.e1_input)
        algo_layout.addWidget(self.e2_input)
        algo_layout.addWidget(self.M_input)
        algo_group.setLayout(algo_layout)

        self.calc_button = QPushButton("Рассчитать")
        self.calc_button.clicked.connect(self.run_calculation)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        left_layout.addWidget(func_group)
        left_layout.addWidget(algo_group)
        left_layout.addWidget(self.calc_button)
        left_layout.addWidget(self.output)

        self.figure_3d = Figure(figsize=(8, 5), dpi=80)
        self.canvas_3d = FigureCanvas(self.figure_3d)
        self.figure_3d.set_tight_layout(True)
        self.figure_2d = Figure(figsize=(8, 4), dpi=80)
        self.canvas_2d = FigureCanvas(self.figure_2d)
        self.figure_3d.set_tight_layout(True)

        right_layout.addWidget(self.canvas_3d)
        right_layout.addWidget(self.canvas_2d)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def create_input(self, label_text, default=""):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        label = QLabel(label_text)
        input_field = QLineEdit(default)
        layout.addWidget(label)
        layout.addWidget(input_field)
        return widget

    def run_calculation(self):
        try:
            a = float(self.a_input.findChild(QLineEdit).text())
            b = float(self.b_input.findChild(QLineEdit).text())
            c = float(self.c_input.findChild(QLineEdit).text())
            x = list(map(float, self.x_input.findChild(QLineEdit).text().split()))
            e1 = float(self.e1_input.findChild(QLineEdit).text())
            e2 = float(self.e2_input.findChild(QLineEdit).text())
            M = int(self.M_input.findChild(QLineEdit).text())

            result, points = method(a, b, c, x, e1, e2, M)

            self.output.clear()
            self.output.append(f"Найденная точка: ({result['point'][0]:.3f}, {result['point'][1]:.3f})")
            self.output.append(f"Значение функции: {result['value']:.3f}")
            self.output.append(f"Количество итераций: {result['iterations']}")
            self.draw_plots(a, b, c, points)

        except Exception as e:
            self.output.append(f"Ошибка: {str(e)}")

    def draw_plots(self, a, b, c, points):
        points = np.array(points)
        x1 = points[:, 0]
        x2 = points[:, 1]

        # 3D график
        self.figure_3d.clear()
        ax1 = self.figure_3d.add_subplot(111, projection='3d')

        X1, X2 = np.meshgrid(np.linspace(min(x1) - 1, max(x1) + 1, 50),
                             np.linspace(min(x2) - 1, max(x2) + 1, 50))
        Y = f(a, b, c, X1, X2)

        ax1.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.7)
        ax1.plot(x1, x2, f(a, b, c, x1, x2), 'r.-', markersize=5)
        ax1.set_title('3D График функции с траекторией')

        # 2D график
        self.figure_2d.clear()
        ax2 = self.figure_2d.add_subplot(111)

        cntr = ax2.contourf(X1, X2, Y, levels=15, cmap='viridis', alpha=0.5)
        ax2.contour(X1, X2, Y, levels=15, colors='gray', linewidths=0.5)

        ax2.plot(x1, x2, 'r-', linewidth=1)
        ax2.scatter(x1, x2, c='red', s=50, edgecolors='white')
        ax2.scatter(x1[0], x2[0], c='green', s=100, label='Начало')
        ax2.scatter(x1[-1], x2[-1], c='yellow', s=100, label='Конец')
        ax2.legend()
        ax2.set_title('Линии уровня с траекторией')

        self.figure_3d.tight_layout()
        self.figure_2d.tight_layout()
        self.canvas_3d.draw()
        self.canvas_2d.draw()

if __name__ == "__main__":
    app = QApplication([])
    window = InterfaceApp()
    window.show()
    app.exec()