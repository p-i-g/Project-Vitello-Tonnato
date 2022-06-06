# coding: utf-8

from threading import Thread

import cv2
import numpy as np
import socketio

from ai.openvtuberai import Wrapper

QUEUE_BUFFER_SIZE = 18


def main():
    cap = cv2.VideoCapture(0)

    ai_wrapper = Wrapper(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), QUEUE_BUFFER_SIZE)
    read_thread = Thread(target=read_frame, args=(cap, ai_wrapper))
    read_thread.start()

    write_thread = Thread(target=write_socket, args=(ai_wrapper,))
    write_thread.start()

    draw(ai_wrapper)
    cap.release()
    cv2.destroyAllWindows()


def read_frame(cap: cv2.VideoCapture, ai_wrapper: Wrapper):
    while True:
        ret, frame = cap.read()

        if ret:
            ai_wrapper.frame_queue.put((ret, frame))
        else:
            print("Skipping empty frame")


def write_socket(ai_wrapper: Wrapper):
    sio = socketio.Client()

    sio.connect("http://127.0.0.1:6789", namespaces='/kizuna')

    while True:
        result_string = ai_wrapper.get_string()
        sio.emit('result_data', result_string, namespace='/kizuna')


def draw(ai_wrapper: Wrapper, color=(125, 255, 0), thickness=2):
    while True:
        frame, landmarks, euler_angle = ai_wrapper.get_upstream()

        for p in np.round(landmarks).astype(np.int):
            cv2.circle(frame, tuple(p), 1, color, thickness, cv2.LINE_AA)

        face_center = np.mean(landmarks, axis=0)
        ai_wrapper.hp.draw_axis(frame, euler_angle, face_center)

        frame = cv2.resize(frame, (960, 720))

        cv2.imshow('result', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
