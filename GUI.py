import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QLineEdit, QTextEdit, QFileDialog
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QLineEdit, QTextEdit, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
# from pipeline import Pipeline

class JsgGui(QWidget):
    def __init__(self):
        super().__init__()
        # self.ppl = Pipeline()
        main_layout = QVBoxLayout()

        h_layout = QHBoxLayout()

        input_layout = QVBoxLayout()
        self.calibPathEdit = QLineEdit()
        self.calibPathBtn = QPushButton('浏览校准文件夹')
        self.calibPathBtn.clicked.connect(self.browse_calib_file)
        input_layout.addWidget(QLabel('校准文件路径:'))
        input_layout.addWidget(self.calibPathEdit)
        input_layout.addWidget(self.calibPathBtn)

        self.captureBtn = QPushButton('开始捕捉图像')
        self.captureBtn.clicked.connect(self.capture_image)
        input_layout.addWidget(self.captureBtn)

        self.runPipelineBtn = QPushButton('运行管道')
        self.runPipelineBtn.clicked.connect(self.run_pipeline)
        input_layout.addWidget(self.runPipelineBtn)

        self.logText = QTextEdit()
        input_layout.addWidget(self.logText)

        self.exitBtn = QPushButton('退出')
        self.exitBtn.clicked.connect(self.close)
        input_layout.addWidget(self.exitBtn)

        h_layout.addLayout(input_layout)

        self.imageLabel = QLabel('图像预览')
        self.imageLabel.setMinimumSize(640, 480)  # 设置最小尺寸
        h_layout.addWidget(self.imageLabel)

        main_layout.addLayout(h_layout)

        self.setLayout(main_layout)

    def browse_calib_file(self):
        pass
        # options = QFileDialog.Options()
        # directory = QFileDialog.getExistingDirectory(self, "选择标定文件夹", "", options=options)
        # if directory:
        #     self.calibPathEdit.setText(directory)
        #     self.ppl.calib_path = directory

    def capture_image(self):
        pass

    def run_pipeline(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = JsgGui()
    gui.show()
    sys.exit(app.exec_())
