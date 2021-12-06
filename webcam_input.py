import cv2
import torch
import numpy as np
from models.simple_classify import Classifier3d

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():

    net = Classifier3d(
        channel_nums=[16, 16, 32, 64, 128],
        block_nums=[1, 1, 1, 1],
    )

    net.load_state_dict(torch.load('classifier.pt'))
    net.to(device)

    capture = cv2.VideoCapture(0)
    video_b, video_g, video_r = [], [], []
    while capture.isOpened():
        ret, color_img = capture.read()
        if not ret: break
        b, g, r = cv2.split(color_img)
        b = cv2.resize(b, (320, 240))
        g = cv2.resize(g, (320, 240))
        r = cv2.resize(r, (320, 240))

        video_b.append(b)
        video_g.append(g)
        video_r.append(r)

    capture.release()
    cv2.destroyAllWindows()

    video = np.array((video_b, video_g, video_r))
    video_capture = torch.from_numpy(video).to(device)

    result = net(video_capture)
    print(result)
