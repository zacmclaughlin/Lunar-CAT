import matplotlib.pyplot as plt
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
plt.style.use("ggplot") # ggplot seaborn
from matplotlib.figure import Figure


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
        self.axes =[]
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

    def show_image(self, images):
        if (len(images) % 2) is not 0:
            images = images[:-1]
        for i in range(len(images)):
            self.axes.append(self.fig.add_subplot(1, len(images), i+1))
            self.axes[i].imshow(images[i])
        self.canvas.draw()
