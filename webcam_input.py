import cv2
import torch
import numpy as np
from models.simple_classify import Classifier3d, Classifier2dLSTM

import threading, time

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SSRContext:
    '''
    Start-Stop-Release Context abstract class.
    Calls start at enter, stop and release at exit.
    '''
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        self.release()
        return False


class WebcamFrame(SSRContext):
    '''
    Webcam Frame updating with multi-threading.
    Referred to https://github.com/PyImageSearch/imutils/blob/master/imutils/video/webcamvideostream.py
    '''
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.release = self.capture.release

    def start(self):
        self.stopped = False
        ret, frame = self.capture.read()
        if ret:
            self.frame = frame
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while True:
            if self.stopped:
                return
            ret, frame = self.capture.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get(self) -> np.ndarray:
        with self.lock:
            return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()


class FrameBuffer:
    def __init__(self, size=1):
        self.size = size
        self.frames = np.zeros((size, 3, 240, 320))
        self.pos = 0
        self.lock = threading.Lock()

    def put(self, frame):
        processed = np.ascontiguousarray(frame.transpose(2, 0, 1)[np.newaxis])
        with self.lock:
            self.frames[self.pos % self.size] = processed
            self.pos += 1

    def to_array(self):
        with self.lock:
            if self.pos < self.size:
                return self.frames[:self.pos]
            else:
                idx = self.pos % self.size
                return np.concatenate((self.frames[idx:], self.frames[:idx]))


class BufferedWebcamFrames(SSRContext):
    '''
    Thread-safe buffered webcam frames. 
    Using WebcamFrame, Framebuffer, and an additional thread, a buffer is kept being updated.
    '''

    def __init__(self, fps, src=0, size=1):
        self.webcam_frame = WebcamFrame(src)
        self.buffer = FrameBuffer(size)
        self.stopped = False
        self.start_time = None
        self.interval = 1/fps
        self.release = self.webcam_frame.release

    def start(self):
        self.stopped = False
        self.webcam_frame.start()
        self.buffer.put(self.webcam_frame.get())
        self.start_time = time.time()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while True:
            if self.stopped:
                return
            start = time.time()
            self.buffer.put(self.webcam_frame.get())
            if time.time() - start > self.interval / 1.5:
                print("[WARNING] Frames are processed slowly")
            time.sleep(self.interval - ((time.time() - self.start_time) % self.interval))

    def get(self) -> np.ndarray:
        '''
        Returns numpy array of shape (T, C, H, W).
        '''
        return self.buffer.to_array()

    def now(self) -> np.ndarray:
        '''
        Returns the most recently captured image as numpy array.
        The shape is (H, W, C).
        '''
        return self.webcam_frame.get().copy()

    def stop(self):
        self.stopped = True
        self.thread.join()


@torch.no_grad()
def main_2dlstm():
    '''
    Main routine for 2D CNN + LSTM network.
    Referred to:

    - cv2 window management: https://stackoverflow.com/a/45564409
    '''

    net = Classifier2dLSTM().to(device)

    net.load_state_dict(torch.load('Classifier2dLSTM.pt', map_location=device))
    net.eval()

    win_table = ["misc", "paper", "rock", "scissors"]

    with BufferedWebcamFrames(fps=30, size=1) as f:
        state = None
        while True:
            frame = f.get().squeeze()
            features = net.get_features(frame, device)
            output, state = net.next_output(features, state)
            idx = int(output[-1].argmax())
            result = win_table[idx]

            screen = f.now()
            cv2.putText(screen, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Result', screen)
            cv2.waitKey(1)

            if cv2.getWindowProperty('Result', cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()


def main():
    main_2dlstm()

if __name__ == '__main__':
    main()
