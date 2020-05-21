from __future__ import print_function
from PIL import Image, ImageTk
import Tkinter as tki
import threading
import imutils
import cv2
import time
import imageio


class videoPlayer():
    def __init__(self):
        self.frame = None
        self.thread = None
        self.stopEvent = None

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

    def videoLoop(self):
        # keep looping over frames until we are instructed to stop
        video_path = '/home/wnj/projects/videos/003.avi'
        video = imageio.get_reader(video_path, 'ffmpeg')
        for frame in video:
            self.frame = imutils.resize(frame, width=300)
            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            # if the panel is not None, we need to initialize it
            if self.panel is None:
                self.panel = tki.Label(image=image)
                self.panel.image = image
                self.panel.pack(side="left", padx=10, pady=10)
                # otherwise, simply update the panel
            else:
                self.panel.configure(image=image)
                self.panel.image = image

            time.sleep(0.02)


myVideoPLayer = videoPlayer()
myVideoPLayer.root.mainloop()
