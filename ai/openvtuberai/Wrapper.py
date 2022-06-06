from queue import Queue
from threading import Thread

from ai.openvtuberai import *


class Wrapper:
    def __init__(self, width, height, buffer_size):
        self.fd = UltraLightFaceDetecion("weights/RFB-320.tflite",
                                         conf_threshold=0.98)
        self.fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")
        self.hp = HeadPoseEstimator("weights/head_pose_object_points.npy",
                                    width, height)
        self.gs = IrisLocalizationModel("weights/iris_localization.tflite")

        self.QUEUE_BUFFER_SIZE = buffer_size

        self.frame_queue = Queue(maxsize=self.QUEUE_BUFFER_SIZE)
        self.box_queue = Queue(maxsize=self.QUEUE_BUFFER_SIZE)
        self.landmark_queue = Queue(maxsize=self.QUEUE_BUFFER_SIZE)
        self.upstream_queue = Queue(maxsize=self.QUEUE_BUFFER_SIZE)
        self.out_queue = Queue(maxsize=self.QUEUE_BUFFER_SIZE)

        face_detection_thread = Thread(target=self.face_detection)
        face_detection_thread.start()

        iris_thread = Thread(target=self.iris_localization)
        iris_thread.start()

        alignment_thread = Thread(target=self.face_alignment)
        alignment_thread.start()

    # ======================================================

    def face_detection(self):
        while True:
            ret, frame = self.frame_queue.get()
            if not ret:
                break

            face_boxes, _ = self.fd.inference(frame)
            self.box_queue.put((frame, face_boxes))

    def face_alignment(self):
        while True:
            frame, boxes = self.box_queue.get()
            landmarks = self.fa.get_landmarks(frame, boxes)
            if landmarks is None:
                continue
            self.landmark_queue.put((frame, landmarks))

    def iris_localization(self, YAW_THD=45):
        while True:
            frame, landmarks = self.landmark_queue.get()
            # calculate head pose
            euler_angle = self.hp.get_head_pose(landmarks).flatten()
            pitch, yaw, roll = euler_angle

            eye_starts = landmarks[[35, 89]]
            eye_ends = landmarks[[39, 93]]
            eye_centers = landmarks[[34, 88]]
            eye_lengths = (eye_ends - eye_starts)[:, 0]

            pupils = eye_centers.copy()

            if yaw > -YAW_THD:
                iris_left = self.gs.get_mesh(frame, eye_lengths[0], eye_centers[0])
                pupils[0] = iris_left[0]

            if yaw < YAW_THD:
                iris_right = self.gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
                pupils[1] = iris_right[0]

            poi = eye_starts, eye_ends, pupils, eye_centers

            theta, pha, _ = self.gs.calculate_3d_gaze(poi)
            mouth_open_percent = (
                                         landmarks[60, 1] - landmarks[62, 1]) / (
                                         landmarks[53, 1] - landmarks[71, 1])
            left_eye_status = (
                                      landmarks[33, 1] - landmarks[40, 1]) / eye_lengths[0]
            right_eye_status = (
                                       landmarks[87, 1] - landmarks[94, 1]) / eye_lengths[1]
            result_string = {'euler': (pitch, -yaw, -roll),
                             'eye': (theta.mean(), pha.mean()),
                             'mouth': mouth_open_percent,
                             'blink': (left_eye_status, right_eye_status)}
            self.out_queue.put(result_string)
            self.upstream_queue.put((frame, landmarks, euler_angle))

    def get_string(self):
        return self.out_queue.get()

    def get_upstream(self):
        return self.upstream_queue.get()
