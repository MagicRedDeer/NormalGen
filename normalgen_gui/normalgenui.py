import sys
import os
import cv2

from PySide2 import QtWidgets, QtUiTools, QtCore, QtGui

import normalgen

dirname = os.path.dirname(__file__)


def loadUI(filename):
    loader = QtUiTools.QUiLoader()
    uifile = QtCore.QFile(filename)
    uifile.open(QtCore.QFile.ReadOnly)
    ui = loader.load(uifile)
    uifile.close()
    return ui


def convertToPixmap(cvImg):
    height, width, channel = cvImg.shape
    bytesPerLine = 3 * width
    qImg = QtGui.QImage(
            cvImg.data, width, height, bytesPerLine,
            QtGui.QImage.Format_RGB888).rgbSwapped()
    return QtGui.QPixmap(qImg)


def connect_slider_to_spinbox(slider, spinbox):
    pass


class NormalGenDialog(QtWidgets.QWidget):
    def __init__(self, ui):
        super().__init__()
        self.dialog = ui
        self.setupUI()
        self.image = None
        self.hmap = None
        self.normalmap = None
        self.aomap = None

    def setupUI(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.dialog)
        self.setLayout(self.layout)
        self.setWindowTitle('NormalGen')
        self.dialog.load_image_button.clicked.connect(self.browse_image)
        self.dialog.generate_normal_button.clicked.connect(
                self.generate_normals)
        self.dialog.generate_ao_button.clicked.connect(
                self.generate_ao)
        self.dialog.generate_all_button.clicked.connect(
                self.generate_ao)
        self.dialog.close_button.clicked.connect(
                self.deleteLater)

    def connect_sliders_and_spinboxes(self):
        connect_slider_to_spinbox(
                self.dialog.normal_strength_slider,
                self.dialog.normal_strength_spinbox)
        connect_slider_to_spinbox(
                self.dialog.normal_level_slider,
                self.dialog.normal_level_spinbox)
        connect_slider_to_spinbox(
                self.dialog.ao_size_slider,
                self.dialog.ao_size_spinbox)

    def browse_image(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Image", r'c:\Users\Quixel\Downloads',
            "Image Files (*.png *.jpg *.bmp)")
        if filename:
            self.filename = filename
            self.read_image()
            self.show_image()

    def read_image(self):
        self.empty_maps()
        self.image = normalgen.load_image(self.filename)
        self.hmap = normalgen.makeGray(self.image)
        self.normalmap = None
        self.aomap = None

    def show_image(self):
        pixmap = convertToPixmap(self.image)
        scaledPixmap = pixmap.scaled(
                self.dialog.image_label.size(), QtCore.Qt.KeepAspectRatio)
        self.dialog.image_label.setPixmap(scaledPixmap)

    def empty_maps(self):
        self.dialog.normal_image_label.clear()
        self.dialog.ao_image_label.clear()

    def show_normalmap(self):
        pixmap = convertToPixmap(self.normalmap)
        scaledPixmap = pixmap.scaled(
                self.dialog.normal_image_label.size(),
                QtCore.Qt.KeepAspectRatio)
        self.dialog.normal_image_label.setPixmap(scaledPixmap)
        self.dialog.output_tab_widget.setCurrentIndex(0)

    def show_aomap(self):
        pixmap = convertToPixmap(self.aomap)
        scaledPixmap = pixmap.scaled(
                self.dialog.ao_image_label.size(),
                QtCore.Qt.KeepAspectRatio)
        self.dialog.ao_image_label.setPixmap(scaledPixmap)
        self.dialog.output_tab_widget.setCurrentIndex(1)

    def generate_normals(self):
        self.empty_maps()
        if self.image is None:
            raise ValueError('No Image Loaded')
        self.normalmap = normalgen.generateNormals(
                self.hmap,
                method=self.dialog.normal_method_combobox.currentText(),
                strength=self.dialog.normal_strength_spinbox.value(),
                level=self.dialog.normal_level_spinbox.value())
        self.show_normalmap()

    def generate_ao(self):
        if self.normalmap is None:
            self.generate_normals()
        self.aomap = normalgen.generateAmbientOcclusion(
                self.hmap, self.normalmap,
                size=self.dialog.ao_size_spinbox.value(),
                height_scaling=self.dialog.ao_hscale_spinbox.value(),
                scale=(self.dialog.ao_xscale_spinbox.value(),
                       self.dialog.ao_yscale_spinbox.value()),
                intensity=self.dialog.ao_intensity_spinbox.value(),
                max_samples=self.dialog.ao_samples_spinbox.value(),
                seed=self.dialog.ao_seed_spinbox)
        self.show_aomap()


def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = loadUI(os.path.join(dirname, 'normalgenui.ui'))
    main_window = NormalGenDialog(ui)
    main_window.show()
    sys.exit(app.exec_())
