import cv2
import torch
import numpy as np
from queue import Queue
from models.simple_classify import Classifier3d, Classifier2dLSTM

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():

    # net = Classifier3d(
    #         channel_nums=[16, 16, 32, 64, 128],
    #         block_nums=[1, 1, 1, 1],
    #     ).to(device)
    net = Classifier2dLSTM().to(device)

    net.load_state_dict(torch.load('Classifier2dLSTM.pt', map_location=device))
    net.eval()

    video_b, video_g, video_r = Queue(128), Queue(128), Queue(128)
    # table = ["misc", "rock", "scissors", "paper"]
    win_table = ["misc", "paper", "rock", "scissors"]

    capture = cv2.VideoCapture(0)

    def read_and_put():
        ret, color_img = capture.read()
        if not ret: raise()

        b, g, r = cv2.split(color_img)
        b = cv2.resize(b, (320, 240))
        g = cv2.resize(g, (320, 240))
        r = cv2.resize(r, (320, 240))

        video_b.put(b)
        video_g.put(g)
        video_r.put(r)

        return b, g, r

    for i in range(128):
        b, g, r = read_and_put()
    
    while capture.isOpened():
        video_b.get()
        video_g.get()
        video_r.get()

        b, g, r = read_and_put()

        video = np.array([np.array((list(video_b.queue), list(video_g.queue), list(video_r.queue)))])
        video_capture = torch.from_numpy(video).to(device).float().div(255)
        video_capture.sub_(0.5).div_(0.5)

        result = net(video_capture)[0]
        result = result[-1].argmax()
        result = win_table[int(result)]

        frame = np.dstack([b, g, r])
        cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Result", frame)
        cv2.waitKey(0)

        print(result)

    capture.release()
    cv2.destroyAllWindows()
