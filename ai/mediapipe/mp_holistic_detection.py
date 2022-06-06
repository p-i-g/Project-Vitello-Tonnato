import threading
import time

import cv2
import mediapipe as mp


# Define the thread that will continuously pull frames from the camera
class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self._stopped = False
        self.start()

    def run(self):
        while not self._stopped:
            ret, self.last_frame = self.camera.read()
        self.camera.release()

    def stop(self):
        self._stopped = True


class Wrapper:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1)

    def detect(self, img):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img.flags.writeable = False
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)
        return results

    def draw(self, img, detections):
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 1. Face
        self.mp_drawing.draw_landmarks(img, detections.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                       self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                       self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                       )

        # 2. Right hand
        self.mp_drawing.draw_landmarks(img, detections.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                       )

        # 3. Left Hand
        self.mp_drawing.draw_landmarks(img, detections.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                       )

        # 4. Pose Detections
        self.mp_drawing.draw_landmarks(img, detections.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                       )


# For webcam input:
def main():
    # used to record the time when we processed last frame
    prev_frame_time = 0

    cap = cv2.VideoCapture(0)
    cam_cleaner = CameraBufferCleanerThread(cap)
    wrapper = Wrapper()

    while cap.isOpened():
        image = cam_cleaner.last_frame
        if image is None:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        results = wrapper.detect(image)

        wrapper.draw(image, results)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame
        new_frame_time = time.time()

        # Calculating the fps

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        # putting the FPS count on the frame
        cv2.putText(image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cam_cleaner.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
