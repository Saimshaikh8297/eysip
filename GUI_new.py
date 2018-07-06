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
from standardization import standard
from threaded_final import mainplot
from evaluate_final_1 import evaluate
import os.path
#from main_file import mainplot
perfect_file_name = None
videos_folder_path = None
color_dict = {'Magenta':0,'Neon Green':1,'Green':2,'Blue':3 }
width = 20
color_marker_top = 0
color_marker_physical = 3
list_weights = [20,20,20,20,20]

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
        self.setGeometry(50, 50, 700, 800)
        self.setWindowTitle('e-Yantra')
        self.setWindowIcon(QIcon('EyantraLogoLarge.png'))
        self.home()

    def home(self):
        label = QLabel(self)
        pixmap = QPixmap('eyantra.png')
        label.setPixmap(pixmap)
        label.setGeometry(180, 10, 350, 70)

        # self.layout = QVBoxLayout()
        # self.label = QLabel("Upload Reference Video")
        # self.layout.addWidget(self.label)
        # self.setLayout(self.layout)
        btn_ref_file = QPushButton('Upload Reference Video', self)
        btn_ref_file.clicked.connect(self.file_open)
        btn_ref_file.resize(500, 80)
        btn_ref_file.move(100, 200)


        btn_folder = QPushButton('Upload Folder Video', self)
        btn_folder.clicked.connect(self.folder_path)
        btn_folder.resize(500, 80)
        btn_folder.move(100, 300)



        self.styleChoise = QLabel('Color Marker', self)
        comboBox = QComboBox(self)
        comboBox.addItem('Magenta')
        comboBox.addItem('Neon Green')
        comboBox.addItem('Green')
        comboBox.addItem('Blue')
        comboBox.move(80, 440)
        self.styleChoise.move(85,400)
        comboBox.activated[str].connect(self.style_choise)

        self.styleChoise1 = QLabel('Physical Marker', self)
        comboBox = QComboBox(self)
        comboBox.addItem('None')
        comboBox.addItem('Magenta')
        comboBox.addItem('Neon Green')
        comboBox.addItem('Green')
        comboBox.addItem('Blue')
        comboBox.move(320, 440)
        self.styleChoise1.move(325,400)
        comboBox.activated[str].connect(self.style_choise1)

        self.styleChoise2 = QLabel('Width', self)
        comboBox = QComboBox(self)
        comboBox.addItem('10')
        comboBox.addItem('15')
        comboBox.addItem('20')
        comboBox.addItem('25')
        comboBox.addItem('30')
        comboBox.addItem('35')
        comboBox.move(520, 440)
        self.styleChoise2.move(529,400)
        comboBox.activated[str].connect(self.style_choise2)

        self.styleChoise4 = QLabel('Programmatic', self)
        comboBox = QComboBox(self)
        comboBox.addItem('10')
        comboBox.addItem('15')
        comboBox.addItem('20')
        comboBox.addItem('25')
        comboBox.addItem('30')
        comboBox.addItem('35')
        comboBox.addItem('40')
        comboBox.addItem('45')
        comboBox.addItem('50')
        comboBox.addItem('55')
        comboBox.addItem('60')
        comboBox.addItem('65')
        comboBox.addItem('70')
        comboBox.addItem('75')
        comboBox.addItem('80')
        comboBox.addItem('85')
        comboBox.addItem('90')
        comboBox.addItem('95')
        comboBox.addItem('100')
        comboBox.move(30, 490)
        self.styleChoise4.move(35,470)
        comboBox.activated[str].connect(self.style_choise4)

        self.styleChoise5 = QLabel('Off-set Reduction', self)
        comboBox = QComboBox(self)
        comboBox.addItem('10')
        comboBox.addItem('15')
        comboBox.addItem('20')
        comboBox.addItem('25')
        comboBox.addItem('30')
        comboBox.addItem('35')
        comboBox.addItem('40')
        comboBox.addItem('45')
        comboBox.addItem('50')
        comboBox.addItem('55')
        comboBox.addItem('60')
        comboBox.addItem('65')
        comboBox.addItem('70')
        comboBox.addItem('75')
        comboBox.addItem('80')
        comboBox.addItem('85')
        comboBox.addItem('90')
        comboBox.addItem('95')
        comboBox.addItem('100')
        comboBox.move(155, 490)
        self.styleChoise5.move(160,470)
        comboBox.activated[str].connect(self.style_choise5)

        self.styleChoise6 = QLabel('Physical Markers', self)
        comboBox = QComboBox(self)
        comboBox.addItem('0')
        comboBox.addItem('10')
        comboBox.addItem('15')
        comboBox.addItem('20')
        comboBox.addItem('25')
        comboBox.addItem('30')
        comboBox.addItem('35')
        comboBox.addItem('40')
        comboBox.addItem('45')
        comboBox.addItem('50')
        comboBox.addItem('55')
        comboBox.addItem('60')
        comboBox.addItem('65')
        comboBox.addItem('70')
        comboBox.addItem('75')
        comboBox.addItem('80')
        comboBox.addItem('85')
        comboBox.addItem('90')
        comboBox.addItem('95')
        comboBox.addItem('100')
        comboBox.move(275, 490)
        self.styleChoise6.move(280,470)
        comboBox.activated[str].connect(self.style_choise6)

        self.styleChoise7 = QLabel('Feature Matching', self)
        comboBox = QComboBox(self)
        comboBox.addItem('10')
        comboBox.addItem('15')
        comboBox.addItem('20')
        comboBox.addItem('25')
        comboBox.addItem('30')
        comboBox.addItem('35')
        comboBox.addItem('40')
        comboBox.addItem('45')
        comboBox.addItem('50')
        comboBox.addItem('55')
        comboBox.addItem('60')
        comboBox.addItem('65')
        comboBox.addItem('70')
        comboBox.addItem('75')
        comboBox.addItem('80')
        comboBox.addItem('85')
        comboBox.addItem('90')
        comboBox.addItem('95')
        comboBox.addItem('100')
        comboBox.move(405, 490)
        self.styleChoise7.move(410,470)
        comboBox.activated[str].connect(self.style_choise7)

        self.styleChoise8 = QLabel('HOG', self)
        comboBox = QComboBox(self)
        comboBox.addItem('10')
        comboBox.addItem('15')
        comboBox.addItem('20')
        comboBox.addItem('25')
        comboBox.addItem('30')
        comboBox.addItem('35')
        comboBox.addItem('40')
        comboBox.addItem('45')
        comboBox.addItem('50')
        comboBox.addItem('55')
        comboBox.addItem('60')
        comboBox.addItem('65')
        comboBox.addItem('70')
        comboBox.addItem('75')
        comboBox.addItem('80')
        comboBox.addItem('85')
        comboBox.addItem('90')
        comboBox.addItem('95')
        comboBox.addItem('100')
        comboBox.move(545, 490)
        self.styleChoise8.move(550,470)
        comboBox.activated[str].connect(self.style_choise8)

        btn_execute = QPushButton('Start Execution', self)
        btn_execute.clicked.connect(self.start_execute)
        btn_execute.resize(500, 100)
        btn_execute.move(100, 570)


        btn2 = QPushButton('Quit', self)
        btn2.clicked.connect(QCoreApplication.instance().quit)
        btn2.resize(500, 100)
        btn2.move(100, 670)



        self.show()


    def file_open(self):
        global perfect_file_name
        if os.path.isfile("Results/results_perfect.csv"):
            choice = QMessageBox.question(self, 'Standard Video',
                                          "Standard file already exists. Do you want to replace it?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if choice == QMessageBox.Yes:
                options = QFileDialog.Options()
                options |= QFileDialog.DontUseNativeDialog
                files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                        "Video Files (*.mp4 *.mov)",
                                                        options=options)
                if files:
                    print(files)

                    perfect_file_name = files[0]

            else:
                perfect_file_name = 0
        else:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "", "Video Files (*.mp4 *.mov)",
                                                    options=options)
            if files:
                print(files)

                perfect_file_name = files[0]

    def style_choise(self, text):
        global  color_marker_top
        color_marker_top = (color_dict[text])


    def style_choise1(self, text):
        global color_marker_physical
        if text=='None':
            color_marker_physical=None
        else:

            color_marker_physical = (color_dict[text])

    def style_choise2(self, text):
        global width
        width = text

    def style_choise4(self, text):
        list_weights[0]=int(text)
        print(list_weights)

    def style_choise5(self, text):
        list_weights[1]=int(text)
        print(list_weights)


    def style_choise6(self, text):
        list_weights[2]=int(text)
        print(list_weights)


    def style_choise7(self, text):
        list_weights[3]=int(text)
        print(list_weights)


    def style_choise8(self, text):
        list_weights[4]=int(text)
        print(list_weights)




    def folder_path(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        if directory:
            print(directory)
            global videos_folder_path
            videos_folder_path = directory
            videos_folder_path = (str(videos_folder_path) + "/")
            print(videos_folder_path)

    def close_app(self):
        print("EXIT")
        sys.exit()
    def start_execute(self):
        if perfect_file_name != None:
            if videos_folder_path != None:
                if color_marker_top==color_marker_physical:
                    choice = QMessageBox.critical(self, 'Error',
                                                  "Color Markers cannot have the same Color!", QMessageBox.Ok,
                                                  QMessageBox.Ok)
                elif sum(list_weights)!=100:
                    choice = QMessageBox.critical(self, 'Error',
                                                  "The sum has to be 100!", QMessageBox.Ok,
                                                  QMessageBox.Ok)

                else:
                    # print("new width "+ str(width))
                    # print("color marker top " + str(color_marker_top))
                    # print("color marker phys " + str(color_marker_physical))
                    # print(list_weights)

                    csv_file_name = standard(perfect_file_name,width)

                    mainplot(csv_file_name,videos_folder_path,color_marker_top,color_marker_physical)

                    evaluate(csv_file_name)



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