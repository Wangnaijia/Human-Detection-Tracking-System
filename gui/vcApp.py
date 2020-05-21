# -*- coding:utf-8

import time
import Tkinter as tk
from Tkinter import *
import cv2
from PIL import Image, ImageTk
import multiprocessing

window_width = 1280
window_height = 480
image_width = int(window_width * 0.5)
image_height = int(window_height)
imagepos_x = 0
imagepos_y = 0
butpos_x = 450
butpos_y = 450
vc1 = cv2.VideoCapture(0)  # 读取视频


# 图像转换，用于在画布中显示
def tkImage(vc):
    ref, frame = vc.read()
    cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(cvimage)
    pilImage = pilImage.resize((image_width, image_height), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=pilImage)
    return tkImage


# 图像的显示与更新
def video():
    def video_loop():
        try:
            while True:
                picture1 = tkImage(vc1)
                canvas1.create_image(0, 0, anchor='nw', image=picture1)
                canvas2.create_image(0, 0, anchor='nw', image=picture1)
                win.update_idletasks()  # 最重要的更新是靠这两句来实现
                win.update()
        except:
            pass

    video_loop()
    win.mainloop()
    vc1.release()
    cv2.destroyAllWindows()


'''布局'''
win = tk.Tk()
win.geometry(str(window_width) + 'x' + str(window_height))
canvas1 = Canvas(win, bg='white', width=image_width, height=image_height)
canvas1.place(x=imagepos_x, y=imagepos_y)
canvas2 = Canvas(win, bg='white', width=image_width, height=image_height)
canvas2.place(x=480, y=0)


if __name__ == '__main__':
    app = video()
    app.start()