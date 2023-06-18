"""
项目名称：音频事件检测-数据集标注工具
开发时间：2023/6/18
项目作者：李化顺
联系方式：2357872806@qq.com(邮箱)、17719333702(微信)
项目简介：
    类似目标检测数据集标注工具Labelme
    AudioEventLabeller用于音频事件数据集标注
基本功能：
    1.支持单文件导入，也支持文件夹导入
    2.视听结合的方式，动动鼠标即可完成标注
    3.标注文件默认保存在./annotations文件夹
注意事项：
    由于不同的数据集类别不同，请参考categories.json准备类别文件
"""
import traceback
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QPushButton, QVBoxLayout, QWidget, QLabel, \
    QTextEdit, QFileDialog, QMessageBox, QSlider, QSplitter, QMenuBar, QMenu, QComboBox
from PyQt5.QtCore import Qt, pyqtSlot
import sys
import sounddevice as sd
import threading
import librosa
import os
import json
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QBrush, QColor


class AudioLabelTool(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a menu bar
        menubar = QMenuBar()
        self.setMenuBar(menubar)

        # Create a file menu
        file_menu = QMenu("文件", self)
        menubar.addMenu(file_menu)
        menubar.setStyleSheet("background-color: #F9F8E9")

        # Create file menu actions
        open_file_action = file_menu.addAction("打开文件")
        open_file_action.triggered.connect(self.select_files)

        open_folder_action = file_menu.addAction("打开文件夹")
        open_folder_action.triggered.connect(self.select_folder)

        self.file_list = QListWidget()

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.hide()

        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.play_audio)

        self.add_button = QPushButton("添加")
        self.add_button.clicked.connect(self.add_annotation)

        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.save_annotations)

        self.current_audio = None
        self.player = QMediaPlayer()

        self.start_slider = QSlider(Qt.Horizontal)
        self.start_slider.valueChanged.connect(self.update_start)

        self.end_slider = QSlider(Qt.Horizontal)
        self.end_slider.valueChanged.connect(self.update_end)

        self.text_input = QTextEdit()

        self.play_selected_button = QPushButton("播放选中区域")
        self.play_selected_button.clicked.connect(self.play_selected_audio)

        layout_right = QVBoxLayout()
        layout_right.addWidget(self.toolbar)
        layout_right.addWidget(QLabel("音频波形图："))
        layout_right.addWidget(self.canvas)
        layout_right.addWidget(QLabel("开始："))
        layout_right.addWidget(self.start_slider)
        layout_right.addWidget(QLabel("结束："))
        layout_right.addWidget(self.end_slider)
        layout_right.addWidget(self.play_button)
        layout_right.addWidget(self.play_selected_button)
        self.category_combo_box = QComboBox()
        self.load_categories('categories.json')  # assuming categories.json is your categories file
        layout_right.addWidget(QLabel("类别："))
        layout_right.addWidget(self.category_combo_box)
        layout_right.addWidget(QLabel("音频事件时间范围："))
        layout_right.addWidget(self.text_input)
        layout_right.addWidget(self.add_button)
        layout_right.addWidget(self.save_button)

        layout_left = QVBoxLayout()
        layout_left.addWidget(QLabel("音频列表："))
        layout_left.addWidget(self.file_list)

        right_panel = QWidget()
        right_panel.setLayout(layout_right)

        left_panel = QWidget()
        left_panel.setLayout(layout_left)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        self.setCentralWidget(splitter)
        # self.setStyleSheet("background-color: #F8F8FF;")

        self.setWindowTitle("音频事件检测-数据集标注工具")
        self.show()

        self.file_list.itemClicked.connect(self.select_audio)

    def play_audio_on_new_thread(self, audio, sample_rate):
        sd.play(audio, sample_rate)

    # 加载类别的函数
    def load_categories(self, category_file):
        with open(category_file, 'r') as f:
            categories = json.load(f)
        self.category_combo_box.addItems(categories)

    @pyqtSlot()
    def play_selected_audio(self):
        start_index = self.start_slider.value()
        end_index = self.end_slider.value()
        selected_audio = self.y[start_index:end_index]
        threading.Thread(target=self.play_audio_on_new_thread,
                         args=(selected_audio, self.sr)).start()  # play selected audio on a new thread

    @pyqtSlot()
    def add_annotation(self):
        start_time = self.start_slider.value() / self.sr  # convert from samples to seconds
        end_time = self.end_slider.value() / self.sr  # convert from samples to seconds
        category = self.category_combo_box.currentText()  # get the current selected category
        self.text_input.append(f"{start_time}-{end_time}, {category}")  # add the range and category to the text box

    @pyqtSlot()
    def update_waveform(self):
        if hasattr(self, 'y') and hasattr(self, 'sr'):
            start_index = self.start_slider.value()
            end_index = self.end_slider.value()
            # Check if start_index and end_index are valid
            if start_index < end_index and end_index <= len(self.y):
                self.ax.clear()
                self.ax.plot(self.y, alpha=0.5)  # original waveform in light grey
                self.ax.plot(range(start_index, end_index), self.y[start_index:end_index],
                             color='red')  # selected part of the waveform in red
                self.canvas.draw()

    def display_waveform(self, audio_path):
        try:
            self.y, self.sr = librosa.load(audio_path)
            print(f"Loaded audio file with sample rate: {self.sr}")  # for debugging
            self.start_slider.setMaximum(len(self.y) - 1)
            self.start_slider.setValue(0)
            self.end_slider.setMaximum(len(self.y) - 1)
            self.end_slider.setValue(len(self.y) - 1)
            self.update_waveform()
            return True
        except Exception as e:
            print(f"Error loading waveform: {e}")  # print error message if something goes wrong
            return False

    @pyqtSlot()
    def select_audio(self):
        # Reset the text_input and player state when selecting new audio
        self.text_input.clear()

        # If the audio is playing, stop it
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.stop()
            while self.player.state() != QMediaPlayer.StoppedState:
                QApplication.processEvents()

        # Reset the audio data
        self.y = None
        self.sr = None

        audio_path = self.file_list.currentItem().text()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(audio_path)))
        if not self.display_waveform(audio_path):  # check for errors in display_waveform
            return  # exit if there were any errors

    @pyqtSlot()
    def play_audio(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    @pyqtSlot()
    def save_annotations(self):
        annotation_text = self.text_input.toPlainText()
        annotations = annotation_text.split('\n')
        event_list = []
        for annotation in annotations:
            if '-' in annotation:
                start, end, category = annotation.split('-')[0], annotation.split('-')[1].split(',')[0], \
                                       annotation.split(',')[1]  # split the annotation into start, end, and category
                start = float(start)
                end = float(end)
                duration = librosa.get_duration(y=self.y, sr=self.sr)  # total duration in seconds
                start /= duration  # start time as proportion of total duration
                end /= duration  # end time as proportion of total duration
                event_list.append({'start': start, 'end': end, 'category': category.strip()})

        # get the current audio file's name
        audio_path = self.file_list.currentItem().text()
        filename = os.path.basename(audio_path)
        filename_no_ext = os.path.splitext(filename)[0]  # remove the file extension
        annotation_filename = filename_no_ext + ".json"  # create the annotation file name

        # specify the directory where you want to save the annotations
        save_directory = "./annotations"
        if not os.path.exists(save_directory):  # create the directory if it does not exist
            os.makedirs(save_directory)
        save_path = os.path.join(save_directory, annotation_filename)

        with open(save_path, 'w') as f:
            json.dump(event_list, f)

        QMessageBox.information(self, "保存成功", f"标注已成功保存为 {annotation_filename} 文件。")
        # Change the background color of the current item to grey
        current_item = self.file_list.currentItem()
        current_item.setBackground(QBrush(QColor(230, 230, 250)))

        # QMessageBox.information(self, "保存成功", f"标注已成功保存为 {annotation_filename} 文件。")

    @pyqtSlot()
    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择音频文件", "", "Audio Files (*.wav *.mp3)")
        self.file_list.addItems(files)

    @pyqtSlot()
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择音频文件夹")
        if folder:
            for file_name in os.listdir(folder):
                if file_name.endswith('.wav') or file_name.endswith('.mp3'):
                    self.file_list.addItem(os.path.join(folder, file_name))

    @pyqtSlot()
    def update_start(self):
        if self.start_slider.value() > self.end_slider.value():
            self.end_slider.setValue(self.start_slider.value())
        self.update_waveform()

    @pyqtSlot()
    def update_end(self):
        if self.end_slider.value() < self.start_slider.value():
            self.start_slider.setValue(self.end_slider.value())
        self.update_waveform()


def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions by printing the traceback."""
    if issubclass(exc_type, KeyboardInterrupt):
        # KeyboardInterrupt is a special case,
        # we don't want the program to exit in an unusual way.
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    else:
        print("Unhandled Exception:", exc_type, exc_value)
        print("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

if __name__ == '__main__':
    sys.excepthook = handle_exception
    app = QApplication(sys.argv)
    window = AudioLabelTool()
    sys.exit(app.exec_())
