import sys
import os
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread
from voc_translator import VocTranslator


class DatasetConverter(QMainWindow):
    def __init__(self, ):
        super(DatasetConverter, self).__init__()
        loadUi('./DatasetConverter.ui', self)

        # slots
        self.pushButtonSelectDatasetFolder.clicked.connect(self.on_pushButtonSelectDatasetFolder)
        self.pushButtonSelectSaveFolder.clicked.connect(self.on_pushButtonSelectSaveFolder)
        self.pushButtonStart.clicked.connect(self.on_pushButtonStart)
        # self.statusBar().showMessage("hello")

    @pyqtSlot()
    def on_pushButtonSelectDatasetFolder(self):
        self.dataset_directory = QFileDialog.getExistingDirectory(self, 'select database directory', "")
        self.lineEditDatasetDir.setText(self.dataset_directory)

    @pyqtSlot()
    def on_pushButtonSelectSaveFolder(self):
        self.save_directory = QFileDialog.getExistingDirectory(self, 'select save directory', "")
        self.lineEditSelectSaveFolder.setText(self.save_directory)

    def updateStatsBar(self, info):
        self.statusBar().showMessage(info)

    @pyqtSlot()
    def on_pushButtonStart(self):
        self._translate_thread = \
            VocTranslator(self.lineEditDatasetDir.text()+'/', self.lineEditSelectSaveFolder.text()+'/',
                            self.lineEditAnnotationFolder.text(), self.lineEditImageFolder.text(),
                            b'JPEG', self.lineEditOutPrefixName.text(),
                            self.checkBoxShuffling.isChecked(), self.plainTextEditLabel.toPlainText(),
                            int(self.lineEditSampelsPerFile.text()))

        self._translate_thread.current_status.connect(self.updateStatsBar)
        self._translate_thread.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DatasetConverter()
    ex.show()
    sys.exit(app.exec_())
