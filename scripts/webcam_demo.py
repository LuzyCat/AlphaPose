import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from alphapose.utils.config import update_config
from alphapose.models import builder
from alphapose.utils.transforms import get_func_heatmap_to_coord
import torch

class AlphaPoseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.config = self.initAlphaPose()
        self.model = self.loadModel()

    def initUI(self):
        self.setWindowTitle('AlphaPose with PyQt')
        self.image_label = QLabel(self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)
        self.show()

    def initAlphaPose(self):
        config = update_config('configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml')
        return config

    def loadModel(self):
        model = builder.build_sppe(self.config.MODEL, preset_cfg=self.config.DATA_PRESET)
        model.load_state_dict(torch.load('pretrained_models/fast_res50_256x192.pth', map_location='cpu'))
        model.eval()
        return model

    def updateFrame(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # AlphaPose processing code here

            # Convert frame to QImage
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(640, 480, aspectRatioMode=Qt.KeepAspectRatio)
            self.image_label.setPixmap(QPixmap.fromImage(p))
            QApplication.processEvents()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AlphaPoseApp()
    sys.exit(app.exec_())
