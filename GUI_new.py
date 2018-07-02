import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QColor, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QAction, QMessageBox, QLabel, QVBoxLayout
from PyQt5.QtWidgets import QCalendarWidget, QFontDialog, QColorDialog, QTextEdit, QFileDialog
from PyQt5.QtWidgets import QCheckBox, QProgressBar, QComboBox, QLabel, QStyleFactory, QLineEdit, QInputDialog

perfect_file_name = None
videos_folder_path = None

# class MainWindow(QWidget):
#
#     def __init__(self):
#         super(MainWindow, self).__init__()
#         self.setGeometry(50, 50, 700, 700)
#         self.setWindowTitle('e-Yantra Auto Evaluation')
#         self.setWindowIcon(QIcon('EyantraLogoLarge.png'))
#
#         label = QLabel(self)
#         pixmap = QPixmap('eyantra.png')
#         label.setPixmap(pixmap)
#         label.setGeometry(0, 0, 800, 210)
#
#         self.layout = QVBoxLayout()
#         self.label = QLabel("Upload Reference Video")
#         self.layout.addWidget(self.label)
#         self.setLayout(self.layout)
#
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     mw = MainWindow()
#     mw.show()
#     sys.exit(app.exec_())


class window(QMainWindow):

    def __init__(self):
        super(window, self).__init__()
        self.setGeometry(50, 50, 700, 700)
        self.setWindowTitle('e-Yantra')
        self.setWindowIcon(QIcon('EyantraLogoLarge.png'))
        self.home()

    def home(self):
        label = QLabel(self)
        pixmap = QPixmap('eyantra.png')
        label.setPixmap(pixmap)
        label.setGeometry(0, 0, 800, 210)

        # self.layout = QVBoxLayout()
        # self.label = QLabel("Upload Reference Video")
        # self.layout.addWidget(self.label)
        # self.setLayout(self.layout)
        btn_ref_file = QPushButton('Upload Reference Video', self)
        btn_ref_file.clicked.connect(self.file_open)
        btn_ref_file.resize(500, 100)
        btn_ref_file.move(100, 200)


        btn_folder = QPushButton('Upload Folder Video', self)
        btn_folder.clicked.connect(self.folder_path)
        btn_folder.resize(500, 100)
        btn_folder.move(100, 300)



        self.styleChoise = QLabel('Color Marker', self)
        comboBox = QComboBox(self)
        comboBox.addItem('Magenta')
        comboBox.addItem('Blue')
        comboBox.addItem('Green')
        comboBox.addItem('Neon Green')
        comboBox.move(100, 420)
        self.styleChoise.move(105,400)
        comboBox.activated[str].connect(self.style_choise)

        self.styleChoise1 = QLabel('Physical Marker', self)
        comboBox = QComboBox(self)
        comboBox.addItem('Magenta')
        comboBox.addItem('Blue')
        comboBox.addItem('Green')
        comboBox.addItem('Neon Green')
        comboBox.move(300, 420)
        self.styleChoise1.move(305,400)
        comboBox.activated[str].connect(self.style_choise1)

        self.styleChoise2 = QLabel('Width', self)
        comboBox = QComboBox(self)
        comboBox.addItem('10')
        comboBox.addItem('15')
        comboBox.addItem('20')
        comboBox.addItem('25')
        comboBox.addItem('30')
        comboBox.addItem('35')
        comboBox.move(500, 420)
        self.styleChoise2.move(505,400)
        comboBox.activated[str].connect(self.style_choise2)

        btn_execute = QPushButton('Start Execution', self)
        btn_execute.clicked.connect(self.start_execute)
        btn_execute.resize(500, 100)
        btn_execute.move(100, 470)


        btn2 = QPushButton('Quit', self)
        btn2.clicked.connect(QCoreApplication.instance().quit)
        btn2.resize(500, 100)
        btn2.move(100, 570)



        self.show()



        self.show()



    def file_open(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "", "Video Files (*.mp4 *.mov)",
                                                options=options)
        if files:
            print(files)
            global perfect_file_name
            perfect_file_name = files[0]

    def style_choise(self, text):

        print(text)

    def style_choise1(self, text):

        print(text)

    def style_choise2(self, text):

        print(text)


    def folder_path(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        if directory:
            print(directory)
            global videos_folder_path
            videos_folder_path = directory
            #videos_folder_path = ("'" + str(videos_folder_path) + "'")

    def start_execute(self):
        if perfect_file_name != None:
            if videos_folder_path != None:
                

            else:
                choice = QMessageBox.critical(self, 'Error',
                                              "You have not selected the path of Folder!", QMessageBox.Ok,
                                              QMessageBox.Ok)
        else:
            choice = QMessageBox.critical(self, 'Error',
                                          "You have not selected the Reference Video!", QMessageBox.Ok, QMessageBox.Ok)

def run():
    app = QApplication(sys.argv)
    Gui = window()

    sys.exit(app.exec_())

run()