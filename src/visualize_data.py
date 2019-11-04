import matplotlib.pyplot as plt
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFrame, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# plt.style.use("ggplot") # ggplot seaborn
from matplotlib.figure import Figure
import numpy as np


class ImageBook(QWidget):

    def __init__(self, image_set):
        super().__init__()
        self.title = 'Network Results Flipbook'
        self.results = image_set

        self.left = 0
        self.top = 0
        self.width = 600
        self.height = 400
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.layout = QVBoxLayout(self)
        self.displayWidget = QFrame(self)
        self.canvasLayout = QVBoxLayout()
        self.displayWidget.setLayout(self.canvasLayout)

        self.chooseWidget = QComboBox(self)
        self.chooseWidget.activated.connect(self.result_choice)
        self.chooseWidget.addItems(self.results.keys())

        self.layout.addWidget(self.displayWidget)
        self.layout.addWidget(self.chooseWidget)

        self.show()

    def result_choice(self, text):
        for i in reversed(range(self.canvasLayout.count())):
            self.canvasLayout.itemAt(i).widget().setParent(None)
        self.canvasLayout.addWidget(self.results[str(text)])
        self.show()


class ImageView(QWidget):
    """
    Usage:
        app = QApplication(sys.argv)

        imageshow = ImageView()

        imageshow.show_image([a_crater, a_guess_mask])

        imageshow.show()

        sys.exit(app.exec_())
    """
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self.dpi = 100
        self.fig = Figure((5.0, 3.0), dpi=self.dpi, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.axes = []
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)
        self.layout.setStretchFactor(self.canvas, 1)
        self.setLayout(self.layout)
        self.font = QtGui.QFont()
        self.font.setPointSize(1)
        self.canvas.show()

    def set_image(self, images, target):
        for i in range(len(images)):
            self.axes.append(self.fig.add_subplot(1, len(images), i+1))
            self.axes[i].imshow(images[i])
            self.axes[i].grid()
        self.fig.suptitle("AS16-M-" + "0" + str(target['filename'].numpy()) + ".jpg")
        self.canvas.draw()
