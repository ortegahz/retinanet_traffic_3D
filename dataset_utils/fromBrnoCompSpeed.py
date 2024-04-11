import json
import os
import pickle
import sys

import cv2
import numpy as np

# Script which creates training data from the BrnoCompSpeed dataset for the Transform3D and Transform2D variants from
# the paper

# Requires https://github.com/matterport/Mask_RCNN

if os.name == 'nt':
    sys.path[0:0] = [os.path.join(sys.path[0], '../../Mask_RCNN')]
else:
    sys.path[0:0] = ['/home/Huangzhe/workspace/Mask_RCNN']

import coco as coco
import model as modellib

import os

# from dataset_utils.warper import get_transform_matrix, intersection, line
if os.name == 'nt':
    from dataset_utils.geometry import line, intersection, computeCameraCalibration

    COCO_MODEL_PATH = os.path.join('D:/Skola/PhD/code/Mask_RCNN', "mask_rcnn_coco.h5")
else:
    from dataset_utils.warper import get_transform_matrix_with_criterion
    from dataset_utils.geometry import line, intersection, computeCameraCalibration

    COCO_MODEL_PATH = os.path.join('/home/k/kocur15/code/Mask_RCNN', "mask_rcnn_coco.h5")
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# COCO_MODEL_PATH = os.path.join('D:\Skola\PhD\code\MASK_RCNN', "mask_rcnn_coco.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    DETECTION_MIN_CONFIDENCE = 0.5


class BCS_boxer(object):
    def __init__(self, model, vid_list, calib_list, pkl_path, images_path, im_w, im_h, lastvid=0, lastpos=0,
                 save_often=False, n=0):
        # self.vehicles = [3, 6, 8]
        self.model = model
        self.vehicles = [3, 4, 6, 8]
        self.vid_list = vid_list
        self.calib_list = calib_list
        self.vid = lastvid
        self.pos = lastpos
        self.pkl_path = pkl_path
        if not os.path.exists(os.path.dirname(self.pkl_path)):
            os.makedirs(os.path.dirname(self.pkl_path))
        self.images_path = images_path
        if not os.path.exists(os.path.dirname(self.images_path)):
            os.makedirs(os.path.dirname(self.images_path))
        self.im_w = im_w
        self.im_h = im_h
        self.save_often = save_often
        self.n = n
        if self.vid != 0 or self.pos != 0:
            with open(self.pkl_path, "rb") as f:
                self.entries = pickle.load(f, encoding='latin-1', fix_imports=True)
        else:
            self.entries = []

    def process(self):
        N = len(self.vid_list)
        for v in range(self.vid, N):
            with open(self.calib_list[v], 'r+') as file:
                # with open(os.path.join(os.path.dirname(json_path), 'system_retinanet_first.json'), 'r+') as file:
                structure = json.load(file)
                camera_calibration = structure['camera_calibration']

            vp1, vp2, vp3, _, _, _ = computeCameraCalibration(camera_calibration["vp1"], camera_calibration["vp2"],
                                                              camera_calibration["pp"])

            vp1 = vp1[:-1] / vp1[-1]
            vp2 = vp2[:-1] / vp2[-1]
            vp3 = vp3[:-1] / vp3[-1]

            self.process_video(self.vid_list[v], vp1, vp2, vp3)
            self.pos = 0
            self.vid += 1

    def id(self):
        return self.pos * 1000 + self.vid

    def filename(self):
        return "{:02d}_{:08d}.png".format(self.vid, self.pos)

    def blob_boxer(self, image, roi, vp0_t, M):
        image = cv2.warpPerspective(np.array(200 * image), M, (self.im_w, self.im_h), borderMode=cv2.BORDER_CONSTANT)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # x_min = roi[1]/(1920/self.im_w)
        # x_max = roi[3]/(1920/self.im_w)
        # y_min = roi[0]/(1080/self.im_h)
        # y_max = roi[2]/(1080/self.im_h)

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            return None
        cnt = contours[0]
        x_min, y_min, w, h = cv2.boundingRect(cnt)
        x_max = x_min + w
        y_max = y_min + h

        if x_max < vp0_t[0]:
            # box vlavo
            cls = 1
        elif x_min > vp0_t[0]:
            # box vpravo
            cls = 3
        else:
            # box vstrede
            cls = 2

        hull = cv2.convexHull(cnt)
        V = [p[0].tolist() for p in hull]

        rt, lt = self.tangent_point_poly(vp0_t, V)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.line(image, tuple(rt), tuple(vp0_t), (0, 255, 0))
        image = cv2.line(image, tuple(lt), tuple(vp0_t), (0, 0, 255))
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        if cls == 1:
            cy1 = intersection(line([x_min, y_min], [x_min, y_max]), line(vp0_t, lt))
            if vp0_t[1] < 0:
                cx = intersection(line([x_min, y_max], [x_max, y_max]), line(vp0_t, rt))
                cy2 = intersection(line(cx, [cx[0], cx[1] + 1]), line(vp0_t, [x_max, y_min]))
            else:
                cx = intersection(line([x_min, y_min], [x_max, y_min]), line(vp0_t, rt))
                cy2 = intersection(line(cx, [cx[0], cx[1] + 1]), line(vp0_t, [x_max, y_max]))

        if cls == 3:
            cy1 = intersection(line([x_max, y_min], [x_max, y_max]), line(vp0_t, rt))
            if vp0_t[1] < 0:
                cx = intersection(line([x_min, y_max], [x_max, y_max]), line(vp0_t, lt))
                cy2 = intersection(line(cx, [cx[0], cx[1] + 1]), line(vp0_t, [x_min, y_min]))
            else:
                cx = intersection(line([x_min, y_min], [x_max, y_min]), line(vp0_t, lt))
                cy2 = intersection(line(cx, [cx[0], cx[1] + 1]), line(vp0_t, [x_min, y_max]))

        if cls == 2:
            cy1 = intersection(line([x_max, y_min], [x_max, y_max]), line(vp0_t, rt))
            cy2 = intersection(line([x_min, y_min], [x_min, y_max]), line(vp0_t, lt))

        image = cv2.circle(image, tuple(cy1), 2, (0, 255, 0))
        image = cv2.circle(image, tuple(cy2), 2, (0, 0, 255))

        # cv2.imshow("Detects", image)
        # cv2.waitKey(0)

        if vp0_t[1] < 0:
            cy = min(cy1[1], cy2[1])
        else:
            cy = max(cy1[1], cy2[1])

        centery = (cy - y_min) / (y_max - y_min)

        if centery < 0:
            centery = 0
        elif centery > 1:
            centery = 1

        # cv2.imshow("Debug", image)
        # cv2.waitKey(0)

        box = {'class_id': cls,
               'x_min': x_min,
               'x_max': x_max,
               'y_min': y_min,
               'y_max': y_max,
               'centery': centery}
        return box

    # je P nalavo od usecky A,B
    def isLeft(self, A, B, P):
        ret = (P[0] - A[0]) * (B[1] - A[1]) - (P[1] - A[1]) * (B[0] - A[0])
        return ret < 0

    def tangent_point_poly(self, p, V):
        left_idx = 0
        right_idx = 0
        p = [np.float64(x) for x in p]
        n = len(V)
        for i in range(1, n):
            if self.isLeft(p, V[left_idx], V[i]):
                left_idx = i
            if not self.isLeft(p, V[right_idx], V[i]):
                right_idx = i
        if p[1] > self.im_h:
            return V[left_idx], V[right_idx]
        return V[right_idx], V[left_idx]

    def process_video(self, vid_path, vp1, vp2, vp3):
        mask = cv2.imread(os.path.join(os.path.dirname(vid_path), 'video_mask.png'), 0)
        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.pos)

        ret, frame = cap.read()
        if mask is None:
            mask = 255 * np.ones(frame.shape[:2], dtype=np.uint8)

        if pair == '12':
            M, IM = get_transform_matrix_with_criterion(vp1, vp2, mask, self.im_w, self.im_h)
            vp0_t = np.array([vp3], dtype="float32")
        elif pair == '13':
            M, IM = get_transform_matrix_with_criterion(vp1, vp3, mask, self.im_w, self.im_h)
            vp0_t = np.array([vp2], dtype="float32")
        else:
            M, IM = get_transform_matrix_with_criterion(vp3, vp2, mask, self.im_w, self.im_h)
            vp0_t = np.array([vp1], dtype="float32")

        vp0_t = np.array([vp0_t])
        vp0_t = cv2.perspectiveTransform(vp0_t, M)
        vp0_t = vp0_t[0][0]

        while ret:
            frames = []
            t_images = []
            for _ in range(InferenceConfig.IMAGES_PER_GPU):
                for _ in range(self.n):
                    ret, frame = cap.read()
                    if not ret:
                        break
                if not ret:
                    break

                frame = cv2.bitwise_and(frame, frame, mask=mask)
                t_image = cv2.warpPerspective(frame, M, (self.im_w, self.im_h), borderMode=cv2.BORDER_CONSTANT)
                frames.append(frame)
                t_images.append(t_image)
            if not ret:
                break

            # cv2.imshow('Original', frame)
            # cv2.imshow('Warped',t_image)
            # cv2.waitKey(0)

            results = self.model.detect(frames)

            for r, t_image in zip(results, t_images):
                boxes = []
                for idx in range(len(r['class_ids'])):
                    if r['class_ids'][idx] in self.vehicles:
                        box = self.blob_boxer(r['masks'][:, :, idx], r['rois'][idx], vp0_t, M)
                        if box is not None:
                            boxes.append(box)

                entry = {'id': self.id(), 'filename': self.filename(), 'labels': boxes}
                self.entries.append(entry)

                targetpath = os.path.join(self.images_path, entry['filename'])
                if not os.path.exists(os.path.dirname(targetpath)):
                    os.makedirs(os.path.dirname(targetpath))
                cv2.imwrite(targetpath, t_image)

                if self.save_often and self.pos % (1000 * self.n) == 0:
                    with open(self.pkl_path, "wb") as f:
                        pickle.dump(self.entries, f)
                        print("Saving, vid:{}, pos:{}".format(self.vid, self.pos))

                self.pos += self.n

        with open(self.pkl_path, "wb") as f:
            pickle.dump(self.entries, f)
            print("Saving, vid:{}, pos:{}".format(self.vid, self.pos))

        cap.release()
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = InferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    pair = '23'

    vid_lists = []
    calib_lists = []

    if os.name == 'nt':
        vid_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/dataset'
        ds_path = 'D:/Skola/PhD/data/BCS_boxed_12/'
        results_path = 'D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/results/'
    else:
        vid_path = '/home/Huangzhe/data/2016-ITS-BrnoCompSpeed/dataset/'
        results_path = '/home/Huangzhe/data/2016-ITS-BrnoCompSpeed/results/'
        ds_path = '/home/Huangzhe/data/BCS_boxed_rot12/'

    for i in range(0, 7):
        dir_list = ['session{}_center'.format(i), 'session{}_left'.format(i), 'session{}_right'.format(i)]
        vid_list = [os.path.join(vid_path, d, 'video.avi') for d in dir_list]
        calib_list = [os.path.join(results_path, d, 'system_dubska_optimal_calib.json') for d in dir_list]
        # pkl_path = os.path.join(ds_path, 'dataset_9.pkl')
        # image_path = os.path.join(ds_path, 'images_9')
        vid_lists.append(vid_list)
        calib_lists.append(calib_list)

    # for i in range(4):
    #     boxer = BCS_boxer(model, vid_lists[i], calib_lists[i],
    #                       pkl_paths[i], image_paths[i], 960, 540, save_often=True, n=25)
    #     boxer.process()

    # if os.name == 'nt':
    #     vid_path = 'D:/Skola/PhD/data/LuvizonDataset/dataset/'
    #     results_path = 'D:/Skola/PhD/data/LuvizonDataset/results/'
    #     ds_path = 'D:/Skola/PhD/data/BCS_boxed23/'
    # else:
    #     vid_path = '/home/k/kocur15/data/luvizon/dataset/'
    #     results_path = '/home/k/kocur15/data/luvizon/results/'
    #     ds_path = '/home/k/kocur15/data/BCS_boxed23/'
    #
    # # # sample vid_dict
    # # vid_dict = {1: [1], 2: [1], 3: [1], 4: [1], 5: [1]}
    # #
    # # # full vid dict
    # # vid_dict = {1: [1, 2, 3, 4], 2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 3: [1, 2], 4: [1, 2], 5: [1]}
    #
    # # train dict
    # vid_dict = {1: [3, 4], 2: [7, 8, 9, 10, 11], 3: [2], 4: [2]}
    #
    # vid_list = []
    # for i in vid_dict.keys():
    #     vid_list.extend(
    #         [os.path.join(vid_path, 'subset{:02d}'.format(i), 'video{:02d}'.format(j), 'video.h264') for j in
    #          vid_dict[i]])
    #
    # calib_path = os.path.join(results_path, 'subset01', 'video01', 'calib.json')
    # calib_list = [calib_path for _ in vid_list]
    #
    # pkl_path = os.path.join(ds_path, 'dataset_9.pkl')
    # image_path = os.path.join(ds_path, 'images_9')
    #
    # boxer = BCS_boxer(model, vid_list, calib_list, pkl_path, image_path, 960, 540, save_often=True, n=15)
    # boxer.process()
