import sys
import os
import time
import platform
import threading

import cv2
import argparse
import torch
import numpy as np
from easydict import EasyDict as edict
from pathlib import Path
from tqdm import tqdm

from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QTextEdit,
                               QToolBar, QFileDialog, QMessageBox, QStyle, QMenu,
                               QTabWidget, QHBoxLayout, QVBoxLayout, QWidget)
from PySide6.QtGui import QImage, QPixmap, QFontDatabase, QFont, QIcon, QAction, QKeySequence
from PySide6.QtMultimedia import QAudioOutput, QMediaFormat, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import Qt, QThread, Signal

from qtmodern import windows, styles

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        help='experiment configure file name',
                        default='configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml')
    parser.add_argument('--checkpoint', type=str,
                        help='checkpoint file name',
                        default='pretrained_models/fast_res50_256x192.pth')
    parser.add_argument('--sp', default=False, action='store_true',
                        help='Use single process for pytorch')
    parser.add_argument('--detector', dest='detector',
                        help='detector name', default="yolo")
    parser.add_argument('--detfile', dest='detfile',
                        help='detection result file', default="")
    parser.add_argument('--indir', dest='inputpath',
                        help='image-directory', default="")
    parser.add_argument('--list', dest='inputlist',
                        help='image-list', default="")
    parser.add_argument('--image', dest='inputimg',
                        help='image-name', default="")
    parser.add_argument('--outdir', dest='outputpath',
                        help='output-directory', default="examples/res/")
    parser.add_argument('--save_img', default=False, action='store_true',
                        help='save result as image')
    parser.add_argument('--vis', default=False, action='store_true',
                        help='visualize image')
    parser.add_argument('--showbox', default=False, action='store_true',
                        help='visualize human bbox')
    parser.add_argument('--profile', default=False, action='store_true',
                        help='add speed profiling at screen output')
    parser.add_argument('--format', type=str,
                        help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
    parser.add_argument('--min_box_area', type=int, default=0,
                        help='min box area to filter out')
    parser.add_argument('--detbatch', type=int, default=5,
                        help='detection batch size PER GPU')
    parser.add_argument('--posebatch', type=int, default=64,
                        help='pose estimation maximum batch size PER GPU')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                        help='save the result json as coco format, using image index(int) instead of image name(str)')
    parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                        help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
    parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                        help='the length of result buffer, where reducing it will lower requirement of cpu memory')
    parser.add_argument('--flip', default=False, action='store_true',
                        help='enable flip testing')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='print detail information')
    """----------------------------- Video options -----------------------------"""
    parser.add_argument('--video', dest='video',
                        help='video-name', default="")
    parser.add_argument('--webcam', dest='webcam', type=int,
                        help='webcam number', default=0)
    parser.add_argument('--save_video', dest='save_video',
                        help='whether to save rendered video', default=False, action='store_true')
    parser.add_argument('--vis_fast', dest='vis_fast',
                        help='use fast rendering', action='store_true', default=False)
    parser.add_argument('--no_qt', dest='show_qt',
                        help='not show in qt program', default=True, action='store_false')
    """----------------------------- Tracking options -----------------------------"""
    parser.add_argument('--pose_flow', dest='pose_flow',
                        help='track humans in video with PoseFlow', action='store_true', default=False)
    parser.add_argument('--pose_track', dest='pose_track',
                        help='track humans in video with reid', action='store_true', default=False)

    return parser.parse_args()

class PoseTrackingApp(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.opt = args
        self.__check__options()

        self.__initUI()

        # self.setWindowTitle('Pose Tracking Application')
        # self.setGeometry(100, 100, 800, 600)

        # # GUI layout
        # self.layout = QVBoxLayout()
        # self.video_label = QLabel()
        # self.video_label.setAlignment(Qt.AlignCenter)
        # self.layout.addWidget(self.video_label)

        # self.connect_button = QPushButton('Connect')
        # self.connect_button.clicked.connect(self.start_webcam)
        # self.layout.addWidget(self.connect_button)

        # self.container = QWidget()
        # self.container.setLayout(self.layout)
        # self.setCentralWidget(self.container)

        # Variables for pose tracking
        self.running = False
        self.thread = None
        self.det_loader = None
        self.pose_model = None
        self.writer = None
    
    def __initUI(self):
        self.setWindowTitle("Pose Tracking Application")
        self.setGeometry(100, 100, 1000, 800)

        # menu & toolbar
        tool_bar = QToolBar()
        self.addToolBar(tool_bar)
        style = self.style()

        file_menu = self.menuBar().addMenu("&File")
        # icon = QIcon.fromTheme("document-open.png",
        #                        style.standardIcon(QStyle.SP_DirOpenIcon))
        # open_action = QAction(icon, "&Open...", self,
        #                       shortcut=QKeySequence.Open, triggered=self.open)
        # file_menu.addAction(open_action)
        # tool_bar.addAction(open_action)
        icon = QIcon(str(ROOT) + '\\icon\\webcam.png')
        webcam_action = QAction(icon, "&Webcam", self,
                              shortcut="Ctrl+W", triggered=self.capture) # connect: capture
        file_menu.addAction(webcam_action)
        tool_bar.addAction(webcam_action)
        icon = QIcon.fromTheme("application-exit.png",
                               style.standardIcon(QStyle.SP_TitleBarCloseButton))
        exit_action = QAction(icon, "E&xit", self,
                              shortcut="Ctrl+Q", triggered=self.close)
        file_menu.addAction(exit_action)

        # play_menu = self.menuBar().addMenu("&Play")
        # icon = QIcon.fromTheme("media-playback-start.png",
        #                        style.standardIcon(QStyle.SP_MediaPlay))
        # self._play_action = tool_bar.addAction(icon, "Play")
        # self._play_action.triggered.connect(self._player.play)
        # play_menu.addAction(self._play_action)

        # icon = QIcon.fromTheme("media-skip-backward-symbolic.svg",
        #                        style.standardIcon(QStyle.SP_MediaSkipBackward))
        # self._previous_action = tool_bar.addAction(icon, "Previous")
        # self._previous_action.triggered.connect(self.previous_clicked)
        # play_menu.addAction(self._previous_action)

        # icon = QIcon.fromTheme("media-playback-pause.png",
        #                        style.standardIcon(QStyle.SP_MediaPause))
        # self._pause_action = tool_bar.addAction(icon, "Pause")
        # self._pause_action.triggered.connect(self._player.pause)
        # play_menu.addAction(self._pause_action)

        # icon = QIcon.fromTheme("media-skip-forward-symbolic.svg",
        #                        style.standardIcon(QStyle.SP_MediaSkipForward))
        # self._next_action = tool_bar.addAction(icon, "Next")
        # self._next_action.triggered.connect(self.next_clicked)
        # play_menu.addAction(self._next_action)

        # icon = QIcon.fromTheme("media-playback-stop.png",
        #                        style.standardIcon(QStyle.SP_MediaStop))
        # self._stop_action = tool_bar.addAction(icon, "Stop")
        # self._stop_action.triggered.connect(self._ensure_stopped)
        # play_menu.addAction(self._stop_action)

        # main
        main_layout = QHBoxLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        # videoplayer
        self._video_widget = QVideoWidget()
        self._video_widget.setMinimumWidth(800)
        self._video_widget.setMinimumHeight(600)
        # self._player.playbackStateChanged.connect(self.update_buttons)
        # self._player.setVideoOutput(self._video_widget)
        # # self._player.mediaStatusChanged.connect(self.on_media_status_changed)
        # # self._player.stateChanged.connect(self.on_media_status_changed)
        # self._player.setLoops(QMediaPlayer.Loops.Infinite)
        # self.update_buttons(self._player.playbackState())

        self._webcam_widget = QLabel()

        ## Webcam Tab
        webcam_tab_layout = QVBoxLayout()
        webcam_tab_layout.addWidget(self._webcam_widget)
        webcam_tab = QWidget()
        webcam_tab.setLayout(webcam_tab_layout)
        ## Video Tab
        video_tab_layout = QVBoxLayout()
        video_tab_layout.addWidget(self._video_widget)
        video_tab = QWidget()
        video_tab.setLayout(video_tab_layout)

        tab_widget = QTabWidget(self)
        tab_widget.addTab(webcam_tab, "Webcam")
        tab_widget.addTab(video_tab, "Video")

        main_layout.addWidget(tab_widget)

        # # result
        # right_layout = QVBoxLayout()
        # self.rightWidget = QWidget()
        # self.rightWidget.setMinimumWidth(500)
        # # self.rightWidget.setMaximumHeight(500)
        # self.rightWidget.setLayout(right_layout)

        # self.process_button = QPushButton("Process")
        # self.process_button.clicked.connect(self.process_video)
        # right_layout.addWidget(self.process_button)

        # self.dropdown_menu = QMenu(self.process_button)
        # self.process_button.setMenu(self.dropdown_menu)

        # action1 = QAction("Video", self)
        # action1.triggered.connect(self.process_video) # connect: process_video
        # self.dropdown_menu.addAction(action1)

        # action2 = QAction("Webcam", self)
        # action2.triggered.connect(self.process_webcam) # connect: process_webcam
        # self.dropdown_menu.addAction(action2)

        # self.result_panel = QTextEdit(self)
        # self.result_panel.setMinimumHeight(480)
        # self.result_panel.setMaximumHeight(480)
        # self.result_panel.setText('Result')
        # right_layout.addWidget(self.result_panel)

        # self.result_label = QLabel(self)
        # self.result_label.setMaximumHeight(100)
        # self.result_label.setText("-")
        # self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.result_label.setFont(QFont("Noto Sans KR", 20, QFont.Bold))
        # right_layout.addWidget(self.result_label)
        # right_layout.setAlignment(Qt.AlignTop)

        # main_layout.addWidget(self.rightWidget)
    
    def __check__options(self):
        if platform.system() == 'Windows':
            self.opt.sp = True
        self.opt.gpus = [int(i) for i in self.opt.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
        self.opt.device = torch.device("cuda:" + str(self.opt.gpus[0]) if self.opt.gpus[0] >= 0 else "cpu")
        self.opt.detbatch = self.opt.detbatch * len(self.opt.gpus)
        self.opt.posebatch = self.opt.posebatch * len(self.opt.gpus)
        self.opt.tracking = self.opt.pose_track or self.opt.pose_flow or self.opt.detector=='tracker'

        if not self.opt.sp:
            torch.multiprocessing.set_start_method('forkserver', force=True)
            torch.multiprocessing.set_sharing_strategy('file_system')

    def check_input(self):  
        # for webcam
        if self.opt.webcam != -1:
            self.opt.detbatch = 1
            return 'webcam', int(self.opt.webcam)
        # for video
        if len(self.opt.video):
            if os.path.isfile(self.opt.video):
                self.videofile = self.opt.video
                return 'video', self.videofile
            else:
                raise IOError('Error: --video must refer to a video file, not directory.')
        else:
            raise NotImplementedError

    def print_finish_info(self):
        print('===========================> Finish Model Running.')
        if (self.opt.save_img or self.opt.save_video) and not self.opt.vis_fast:
            print('===========================> Rendering remaining images in the queue...')
            print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')

        
    def capture(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run_pose_tracking)
            self.thread.start()

    def run_pose_tracking(self):

        if not os.path.exists(self.opt.outputpath):
            os.makedirs(self.opt.outputpath)

        mode, input_source = self.check_input()
        
        # Update config
        self.cfg = update_config(self.opt.cfg)

        # Load detection loader
        if mode == 'webcam':
            self.det_loader = WebCamDetectionLoader(input_source, get_detector(self.opt), self.cfg, self.opt)
            self.det_worker = self.det_loader.start()
        else:
            self.det_loader = DetectionLoader(input_source, get_detector(self.opt), self.cfg, self.opt)
            self.det_worker = self.det_loader.start()

        # Load pose model
        self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
        print('Loading pose model from %s...' % (self.opt.checkpoint,))

        self.pose_model.load_state_dict(torch.load(self.opt.checkpoint, map_location=self.opt.device))
        self.pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        if self.opt.pose_track:
            self.tracker = Tracker(tcfg, self.opt)
        if len(self.opt.gpus) > 1:
            self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=self.opt.gpus).to(self.opt.device)
        else:
            self.pose_model.to(self.opt.device)
        self.pose_model.eval()

        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }

        # init data writer
        queueSize = 2 if mode == 'webcam' else self.opt.qsize
        if self.opt.save_video:
            from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
            if mode == 'video':
                video_save_opt['savepath'] = os.path.join(self.opt.outputpath, 'AlphaPose_' + os.path.basename(input_source))
            elif mode == 'webcam':
                video_save_opt['savepath'] = os.path.join(self.opt.outputpath, 'AlphaPose_webcam_'  + str(input_source) + '.mp4')
            video_save_opt.update(self.det_loader.videoinfo)
            self.writer = DataWriter(self.cfg, self.opt, save_video=True, video_save_opt=video_save_opt, queueSize=queueSize).start()
        else:
            self.writer = DataWriter(self.cfg, self.opt, save_video=False, queueSize=queueSize, image_callback=self.display_image).start()

        if mode == 'webcam':
            print('Starting webcam demo, press Ctrl + C to terminate...')
            sys.stdout.flush()
            im_names_desc = tqdm(loop())
        else:
            data_len = self.det_loader.length
            im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

        batchSize = self.opt.posebatch
        if self.opt.flip:
            batchSize = int(batchSize / 2)
        
        try:
            for i in im_names_desc:
                start_time = getTime()
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = self.det_loader.read()
                    if orig_img is None:
                        break
                    if boxes is None or boxes.nelement() == 0:
                        self.writer.save(None, None, None, None, None, orig_img, im_name)
                        continue
                    if self.opt.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    # Pose Estimation
                    inps = inps.to(self.opt.device)
                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % batchSize:
                        leftover = 1
                    num_batches = datalen // batchSize + leftover
                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                        if self.opt.flip:
                            inps_j = torch.cat((inps_j, flip(inps_j)))
                        hm_j = self.pose_model(inps_j)
                        if self.opt.flip:
                            hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], self.pose_dataset.joint_pairs, shift=True)
                            hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    if self.opt.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    if self.opt.pose_track:
                        boxes,scores,ids,hm,cropped_boxes = track(self.tracker,self.opt,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
                    hm = hm.cpu()
                    self.writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                    if self.opt.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)
                    self.display_image(orig_img)

                if self.opt.profile:
                    # TQDM
                    im_names_desc.set_description(
                        'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                            dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                    )
            self.print_finish_info()
            while(self.writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(self.writer.count()) + ' images in the queue...', end='\r')
            self.writer.stop()
            self.det_loader.stop()
        except Exception as e:
            print(repr(e))
            print('An error as above occurs when processing the images, please check it')
            pass
        except KeyboardInterrupt:
            self.print_finish_info()
            # Thread won't be killed when press Ctrl+C
            if self.opt.sp:
                self.det_loader.terminate()
                while(self.writer.running()):
                    time.sleep(1)
                    print('===========================> Rendering remaining ' + str(self.writer.count()) + ' images in the queue...', end='\r')
                self.writer.stop()
            else:
                # subprocesses are killed, manually clear queues

                self.det_loader.terminate()
                self.writer.terminate()
                self.writer.clear_queues()
                self.det_loader.clear_queues()


    def display_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self._webcam_widget.setPixmap(pixmap)
        self._webcam_widget.setAlignment(Qt.AlignCenter)

    def closeEvent(self, event):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        event.accept()

def loop():
    n = 0
    while True:
        yield n
        n += 1

if __name__ == "__main__":
    args = parse_opt()
    app = QApplication()
    window = PoseTrackingApp(args)
    styles.dark(app)
    fontDB = QFontDatabase()
    fontDB.addApplicationFont('demo/font/Supreme-Medium.otf')
    fontDB.addApplicationFont('./font/NotoSansKR-Regular.otf')
    app.setFont(QFont('Noto Sans KR'))
    mw = windows.ModernWindow(window)
    mw.show()
    sys.exit(app.exec())
