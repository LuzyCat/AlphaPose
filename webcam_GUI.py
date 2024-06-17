import sys
import os
import time
import datetime
import platform

import argparse
import cv2
import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
from queue import Queue
from threading import Thread
# # debug
# import debugpy

from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QTextEdit,
                               QToolBar, QFileDialog, QMessageBox, QStyle, QMenu,
                               QSpinBox, QComboBox, QGroupBox,
                               QTabWidget, QHBoxLayout, QVBoxLayout, QWidget)
from PySide6.QtGui import QImage, QPixmap, QFontDatabase, QFont, QIcon, QAction, QKeySequence, QTextCursor
from PySide6.QtMultimedia import QAudioOutput, QMediaFormat, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QObject

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms, write_json
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader


parser = argparse.ArgumentParser(description='Pose Tracking Application')
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
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)
args = parser.parse_args()
cfg = update_config(args.cfg)

# args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
# args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
# args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

# class DetectionLoader():
#     def __init__(self, detector, cfg, opt):
#         self.cfg = cfg
#         self.opt = opt
#         self.device = opt.device
#         self.detector = detector

#         self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
#         self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

#         self._sigma = cfg.DATA_PRESET.SIGMA

#         if cfg.DATA_PRESET.TYPE == 'simple':
#             pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
#             self.transformation = SimpleTransform(
#                 pose_dataset, scale_factor=0,
#                 input_size=self._input_size,
#                 output_size=self._output_size,
#                 rot=0, sigma=self._sigma,
#                 train=False, add_dpg=False, gpu_device=self.device)
#         elif cfg.DATA_PRESET.TYPE == 'simple_smpl':
#             # TODO: new features
#             from easydict import EasyDict as edict
#             dummpy_set = edict({
#                 'joint_pairs_17': None,
#                 'joint_pairs_24': None,
#                 'joint_pairs_29': None,
#                 'bbox_3d_shape': (2.2, 2.2, 2.2)
#             })
#             self.transformation = SimpleTransform3DSMPL(
#                 dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
#                 color_factor=cfg.DATASET.COLOR_FACTOR,
#                 occlusion=cfg.DATASET.OCCLUSION,
#                 input_size=cfg.MODEL.IMAGE_SIZE,
#                 output_size=cfg.MODEL.HEATMAP_SIZE,
#                 depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
#                 bbox_3d_shape=(2.2, 2,2, 2.2),
#                 rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
#                 train=False, add_dpg=False, gpu_device=self.device,
#                 loss_type=cfg.LOSS['TYPE'])

#         self.image = (None, None, None, None)
#         self.det = (None, None, None, None, None, None, None)
#         self.pose = (None, None, None, None, None, None, None)

#     def process(self, im_name, image):
#         # start to pre process images for object detection
#         self.image_preprocess(im_name, image)
#         # start to detect human in images
#         self.image_detection()
#         # start to post process cropped human image for pose estimation
#         self.image_postprocess()
#         return self

#     def image_preprocess(self, im_name, image):
#         # expected image shape like (1,3,h,w) or (3,h,w)
#         img = self.detector.image_preprocess(image)
#         if isinstance(img, np.ndarray):
#             img = torch.from_numpy(img)
#         # add one dimension at the front for batch if image shape (3,h,w)
#         if img.dim() == 3:
#             img = img.unsqueeze(0)
#         orig_img = image # scipy.misc.imread(im_name_k, mode='RGB') is depreciated
#         im_dim = orig_img.shape[1], orig_img.shape[0]

#         im_name = os.path.basename(im_name)

#         with torch.no_grad():
#             im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

#         self.image = (img, orig_img, im_name, im_dim)

#     def image_detection(self):
#         imgs, orig_imgs, im_names, im_dim_list = self.image
#         if imgs is None:
#             self.det = (None, None, None, None, None, None, None)
#             return

#         with torch.no_grad():
#             dets = self.detector.images_detection(imgs, im_dim_list)
#             if isinstance(dets, int) or dets.shape[0] == 0:
#                 self.det = (orig_imgs, im_names, None, None, None, None, None)
#                 return
#             if isinstance(dets, np.ndarray):
#                 dets = torch.from_numpy(dets)
#             dets = dets.cpu()
#             boxes = dets[:, 1:5]
#             scores = dets[:, 5:6]
#             ids = torch.zeros(scores.shape)

#         boxes = boxes[dets[:, 0] == 0]
#         if isinstance(boxes, int) or boxes.shape[0] == 0:
#             self.det = (orig_imgs, im_names, None, None, None, None, None)
#             return
#         inps = torch.zeros(boxes.size(0), 3, *self._input_size)
#         cropped_boxes = torch.zeros(boxes.size(0), 4)

#         self.det = (orig_imgs, im_names, boxes, scores[dets[:, 0] == 0], ids[dets[:, 0] == 0], inps, cropped_boxes)

#     def image_postprocess(self):
#         with torch.no_grad():
#             (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes) = self.det
#             if orig_img is None:
#                 self.pose = (None, None, None, None, None, None, None)
#                 return
#             if boxes is None or boxes.nelement() == 0:
#                 self.pose = (None, orig_img, im_name, boxes, scores, ids, None)
#                 return

#             for i, box in enumerate(boxes):
#                 inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
#                 cropped_boxes[i] = torch.FloatTensor(cropped_box)

#             self.pose = (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes)

#     def read(self):
#         return self.pose

class QTextEditLogger(QObject):
    write_signal = Signal(str)

    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.write_signal.connect(self.write_to_text_edit)

    def write(self, message):
        self.write_signal.emit(message)

    def flush(self):
        pass

    def write_to_text_edit(self, message):
        self.text_edit.moveCursor(QTextCursor.End)
        self.text_edit.insertPlainText(message)
        self.text_edit.moveCursor(QTextCursor.End)

class DataWriter():
    def __init__(self, cfg, opt, queueSize=1024):
        self.cfg = cfg
        self.opt = opt

        self.eval_joints = list(range(cfg.DATA_PRESET.NUM_JOINTS))
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        # self.item = (None, None, None, None, None, None, None)
        
        if opt.sp:
            self.result_queue = Queue(maxsize=queueSize)
        else:
            self.result_queue = mp.Queue(maxsize=queueSize)
        
        if opt.pose_flow:
            from trackers.PoseFlow.poseflow_infer import PoseFlowWrapper
            self.pose_flow_wrapper = PoseFlowWrapper(save_path=os.path.join(opt.outputpath, 'poseflow'))

        loss_type = self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss')
        num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
        if loss_type == 'MSELoss':
            self.vis_thres = [0.4] * num_joints
        elif 'JointRegression' in loss_type:
            self.vis_thres = [0.05] * num_joints
        elif loss_type == 'Combined':
            if num_joints == 68:
                hand_face_num = 42
            else:
                hand_face_num = 110
            self.vis_thres = [0.4] * (num_joints - hand_face_num) + [0.05] * hand_face_num

        self.use_heatmap_loss = (self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss')
        self.final_result = []

    def start(self):
        # start to read pose estimation results
        return self.update()

    def update(self):
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE

        # ensure the queue is not empty and get item
        (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.wait_and_get(self.result_queue)
        if orig_img is None:
            return None
        # image channel RGB->BGR
        orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
        self.orig_img = orig_img
        if boxes is None or len(boxes) == 0:
            return None
        else:
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            assert hm_data.dim() == 4

            face_hand_num = 110
            if hm_data.size()[1] == 136:
                self.eval_joints = [*range(0,136)]
            elif hm_data.size()[1] == 26:
                self.eval_joints = [*range(0,26)]
            elif hm_data.size()[1] == 133:
                self.eval_joints = [*range(0,133)]
            elif hm_data.size()[1] == 68:
                face_hand_num = 42
                self.eval_joints = [*range(0,68)]
            elif hm_data.size()[1] == 21:
                self.eval_joints = [*range(0,21)]
            pose_coords = []
            pose_scores = []
            for i in range(hm_data.shape[0]):
                bbox = cropped_boxes[i].tolist()
                if isinstance(self.heatmap_to_coord, list):
                    pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                        hm_data[i][self.eval_joints[:-face_hand_num]], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                        hm_data[i][self.eval_joints[-face_hand_num:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                    pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
                else:
                    pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)
            if not self.opt.pose_track:
                boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                    pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area, use_heatmap_loss=self.use_heatmap_loss)

            _result = []
            for k in range(len(scores)):
                _result.append(
                    {
                        'keypoints':preds_img[k],
                        'kp_score':preds_scores[k],
                        'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx':ids[k],
                        'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                    }
                )

            result = {
                'imgname': im_name,
                'result': _result
            }

            if self.opt.pose_flow:
                poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
                for i in range(len(poseflow_result)):
                    result['result'][i]['idx'] = poseflow_result[i]['idx']

            self.final_result.append(result)

            if hm_data.size()[1] == 49:
                from alphapose.utils.vis import vis_frame_dense as vis_frame
            elif self.opt.vis_fast:
                from alphapose.utils.vis import vis_frame_fast as vis_frame
            else:
                from alphapose.utils.vis import vis_frame
            self.vis_frame = vis_frame

        return result

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        # save next frame in the queue
        self.wait_and_put(self.result_queue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name))

    def running(self):
        # indicate that the thread is still running
        return not self.result_queue.empty()

    def count(self):
        # indicate the remaining images
        return self.result_queue.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None)

    def clear_queues(self):
        self.clear(self.result_queue)
        
    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def results(self):
        # return final result
        print(self.final_result)
        return self.final_result

    def write_json(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.opt.outputpath, timestamp)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        write_json(self.final_result, output_dir, form=self.opt.format, for_eval=self.opt.eval)
        print(f"Results have been written to {output_dir}")


class MainWindow(QMainWindow):
    def __init__(self, args, cfg):
        super().__init__()
        self.args = args
        self.cfg = cfg
       
        self.mode = None
        self.input_source = None
        self.mode, self.input_source = self.check_input()
        self.inference_fps = 29.95

        self.__check__options()
        self.__initUI()
        self.__initialize()
        
        self.is_running = False
        self.cap = None
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_result)

        # QTextEditLogger 설정
        self.logger = QTextEditLogger(self.log_text_edit)
        sys.stdout = self.logger


    def __initUI(self):
        self.setWindowTitle("Pose Tracking Application")
        self.setGeometry(100, 100, 1100, 800)

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

        # main
        main_layout = QHBoxLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        # videoplayer
        self._video_widget = QVideoWidget()
        self._video_widget.setMinimumWidth(800)
        self._video_widget.setMinimumHeight(600)

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
    
        # control panel layout
        control_panel_layout = QVBoxLayout()

        # Create a group box to contain status, frame, persons, model
        self.info_group_box = QGroupBox("Information")
        info_layout = QVBoxLayout()
        
        # Pose tracking status display
        self.status_label = QLabel("Status: Ready")
        info_layout.addWidget(self.status_label)

        # frame number/time stamp display
        self.frame_label = QLabel("Frame: 0")
        info_layout.addWidget(self.frame_label)

        # 인식된 사람 수
        self.person_count_label = QLabel("Persons: 0")  
        info_layout.addWidget(self.person_count_label)

        # Model
        self.model_label = QLabel("Model:")
        info_layout.addWidget(self.model_label)

        self.info_group_box.setLayout(info_layout)
        control_panel_layout.addWidget(self.info_group_box)

        # FPS control
        self.fps_label = QLabel("FPS: 30")
        control_panel_layout.addWidget(self.fps_label)
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.valueChanged.connect(self.set_fps)
        control_panel_layout.addWidget(self.fps_spinbox)


        # log
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setStyleSheet("background-color: #2E2E2E; color: white;")
        control_panel_layout.addWidget(self.log_text_edit)

        main_layout.addLayout(control_panel_layout)

        """AlphaPose"""

    def __check__options(self):
        if platform.system() == 'Windows':
            self.args.sp = True
        self.args.gpus = [int(i) for i in self.args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
        self.args.device = torch.device("cuda:" + str(self.args.gpus[0]) if self.args.gpus[0] >= 0 else "cpu")
        self.args.detbatch = self.args.detbatch * len(self.args.gpus)
        self.args.posebatch = self.args.posebatch * len(self.args.gpus)
        self.args.tracking = self.args.pose_track or self.args.pose_flow or self.args.detector=='tracker'

        if not self.args.sp:
            torch.multiprocessing.set_start_method('forkserver', force=True)
            torch.multiprocessing.set_sharing_strategy('file_system')

    def __initialize(self):
        # Load pose model
        self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)

        self.model_label.setText("Model: " + self.args.checkpoint)
        print('Loading pose model from %s...' % (self.args.checkpoint,))
        self.pose_model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device))
        self.pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        if self.args.pose_track:
            self.tracker = Tracker(tcfg, self.args)
        if len(self.args.gpus) > 1:
            self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=self.args.gpus).to(self.args.device)
        else:
            self.pose_model.to(self.args.device)
        self.pose_model.eval()

        # Init data writer
        self.writer = DataWriter(self.cfg, self.args)

    def check_input(self):  
        # for webcam
        if self.args.webcam != -1:
            self.args.detbatch = 1
            return 'webcam', int(self.args.webcam)
        # for video
        if len(self.args.video):
            if os.path.isfile(self.args.video):
                self.videofile = self.args.video
                return 'video', self.videofile
            else:
                raise IOError('Error: --video must refer to a video file, not directory.')
        else:
            raise NotImplementedError

    def closeEvent(self, event):
        self.update_timer.stop()
        self.det_loader.terminate()
        self.writer.stop()
        event.accept()
    
    @Slot()
    def set_fps(self, fps):
        if not self.is_running:
            self.inference_fps = fps
            self.fps_label.setText("FPS: " + fps)

    @Slot()
    def capture(self):
        if not self.is_running:
            self.is_running = True
            self.status_label.setText("Status: Tracking...")
            self.status_label.setStyleSheet("color: red;")
            # Load detection loader
            self.det_loader = WebCamDetectionLoader(self.input_source, get_detector(self.args), self.cfg, self.args)
            self.det_worker = self.det_loader.start()
            print("Load Camera...")
            self.update_timer.start(self.inference_fps)
        else:
            self.is_running = False
            print("=== Finish Model Running ===")
            self.status_label.setText("Status: Ready")
            self.status_label.setStyleSheet("color: black;")
            self.update_timer.stop()
            self.writer.write_json()
            if self.args.sp:
                self.det_loader.terminate()
                self.writer.stop()
            else:
                self.det_loader.terminate()
                self.writer.stop()
                self.writer.clear_queues()
                self.det_loader.clear_queues()
    
    def update_webcam(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape
        q_image = QImage(frame.data, w, h, c * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self._webcam_widget.setPixmap(pixmap)
        self._webcam_widget.setAlignment(Qt.AlignCenter)
    
    def update_result(self):
        pose = self.process()
        img = self.getImg()
        img = self.vis(img, pose)
        self.update_webcam(img)
        # result = [pose]
        # self.writeJson(result, self.args.outputpath, form=self.args.format, for_eval=self.args.eval)

    def process(self):
        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }
        batchSize = self.args.posebatch
        if self.args.flip:
            batchSize = int(batchSize / 2)
        pose = None
        try:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = self.det_loader.read()
                if orig_img is None:
                    raise Exception("no image is given")
                if boxes is None or boxes.nelement() == 0:
                    if self.args.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    self.writer.save(None, None, None, None, None, orig_img, im_name)
                    if self.args.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    pose = self.writer.start()
                    if self.args.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)
                else:
                    if self.args.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    # Pose Estimation
                    inps = inps.to(self.args.device)
                    datalen = inps.size(0)
                    leftwover = 0
                    if (datalen) % batchSize:
                        leftover = 1
                    num_batches = datalen // batchSize + leftover
                    hm = []

                    # if self.args.flip:
                    #     inps = torch.cat((inps, flip(inps)))
                    # hm = self.pose_model(inps)
                    # if self.args.flip:
                    #     hm_flip = flip_heatmap(hm[int(len(hm) / 2):], self.pose_dataset.joint_pairs, shift=True)
                    #     hm = (hm[0:int(len(hm) / 2)] + hm_flip) / 2
                    for j in range(num_batches):
                        inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    if args.flip:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = self.pose_model(inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], self.pose_dataset.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)

                    hm = torch.cat(hm)
                    if self.args.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    if args.pose_track:
                        boxes,scores,ids,hm,cropped_boxes = track(self.tracker,self.args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
                    hm = hm.cpu()
                    self.writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                    pose = self.writer.start()
                    if self.args.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)

            if self.args.profile:
                print(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )

            # Increment frame label
            current_frame = int(self.frame_label.text().split(": ")[1])  # Extract the current frame number
            self.frame_label.setText(f"Frame: {current_frame + 1}")  # Update the frame label

            # Update person count
            person_count = boxes.size(0) if boxes is not None else 0
            self.person_count_label.setText(f"Persons: {person_count}")

        except Exception as e:
            print(repr(e))
            print('An error as above occurs when processing the images, please check it')
            pass
        except KeyboardInterrupt:
            print('===========================> Finish Model Running.')
            if self.args.sp:
                self.det_loader.terminate()
                self.writer.stop()
            else:
                self.det_loader.terminate()
                self.writer.stop()
                self.writer.clear_queues()
                self.det_loader.clear_queues()

        return pose

    def getImg(self):
        return self.writer.orig_img

    def vis(self, image, pose):
        if pose is not None:
            image = self.writer.vis_frame(image, pose, self.writer.opt, self.writer.vis_thres)
        return image

    # def writeJson(self, final_result, outputpath, form='coco', for_eval=False):
    #     from alphapose.utils.pPose_nms import write_json
    #     write_json(final_result, outputpath, form=form, for_eval=for_eval)
    #     print("Results have been written to json.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    QFontDatabase.addApplicationFont('demo/font/Supreme-Medium.otf')
    QFontDatabase.addApplicationFont('./font/NotoSansKR-Regular.otf')
    app.setFont(QFont('Noto Sans KR', 12))
    main_win = MainWindow(args, cfg)
    available_geometry = main_win.screen().availableGeometry()
    main_win.resize(available_geometry.width() / 3,
                    available_geometry.height() / 2)
    main_win.show()
    sys.exit(app.exec())