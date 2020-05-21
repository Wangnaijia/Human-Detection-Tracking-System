# -*- coding:utf-8 -*-
from Tkinter import *
from PIL import Image, ImageTk
import threading
import imageio
import logging
import time
import cv2

try:
    import tkinter as tk  # Python 3.x
    import tkinter.scrolledtext as ScrolledText
except ImportError:
    import Tkinter as tk  # Python 2.x
    import ScrolledText

window_width = 1280
window_height = 480
image_width = int(window_width * 0.5)
image_height = int(window_height)
imagepos_x = 0
imagepos_y = 0
butpos_x = 450
butpos_y = 450
pilImage=None
tkImage=None
vc1 = cv2.VideoCapture('/home/wnj/projects/videos/003.avi')  # 读取视频

class Application(Frame):
    # 基准页面
    def __init__(self, master=None):
        Frame.__init__(self, master, bg='black')
        self.pack(expand=YES, fill=BOTH)
        self.window_init()
        self.createWidgets()

    def window_init(self):
        self.master.title('welcome to human detecting and tracking system')
        self.master.bg = 'black'
        width, height = self.master.maxsize()
        self.master.geometry("1280x760")

    def createWidgets(self):
        self.video_var=0
        # fm1
        self.fm1 = Frame(self, bg='black')
        self.titleLabel = Label(self.fm1, text="human detecting and tracking system", font=('Dialog', 24), fg="white",
                                bg='black')
        self.titleLabel.pack()
        self.fm1.pack(side=TOP, fill='x', pady=2)

        # fm2
        self.fm2 = Frame(self, bg='black')
        self.stereoButton = Button(self.fm2, text='stereo camera', bg='#22C9C9', fg='white',
                                   font=('Dialog', 16), width='16',command=self.stereo_cam)
        self.stereoButton.pack(side=LEFT, padx=5)
        self.videoButton = Button(self.fm2, text='video tracking', bg='#22C9C9', fg='white',
                                  font=('Dialog', 16), width='16',command=self.video_win)
        self.videoButton.pack(side=LEFT, padx=5)
        self.fm2.pack(side=TOP, expand=YES, fill="x", pady=20)

        self.fm3 = Frame(self, bg='black')
        if self.video_var:
            logging.info('play')
            self.start_video_thread()
        self.fm3.pack(side=TOP, expand=YES, fill="y")

        # fm4 显示日志信息
        self.fm4 = Frame(self, bg='black')
        self.fm4_left = Frame(self.fm4, bg='black')
        self.fm4_right = Frame(self.fm4, bg='black')
        self.build_logger()
        self.fm4_left.pack(side=LEFT, expand=YES, fill="x", pady=2)
        self.result_output()
        self.fm4_right.pack(side=RIGHT, expand=YES, fill="x", pady=2)
        self.fm4.pack(side=TOP, expand=YES, fill="x", pady=2)

    def build_logger(self):
        # 添加文本窗口显示日志信息
        st = ScrolledText.ScrolledText(self.fm4_left, width=70,height=13,state='disabled')
        st.configure(font='TkFixedFont')
        st.pack(side=TOP, expand=YES, fill="x", pady=2)
        # Create textLogger
        text_handler = TextHandler(st)
        # Logging configuration
        logging.basicConfig(filename='test.log',
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # logger句柄
        logger = logging.getLogger()
        logger.addHandler(text_handler)

    def result_output(self):
        result = Label(self.fm4_right, width=40,height=13,text="Help!")
        result.configure(font='TkFixedFont')
        result.pack(side=TOP, expand=YES, fill="x", pady=2)

    def video_win(self):
        self.video_var=1

    def stereo_cam(self):
        cap = cv2.VideoCapture(0)
        i = 0
        while (True):
            ret, frame = cap.read()
            cv2.imshow('Camera', frame)
            k = cv2.waitKey(1)
            i += 1
            if k == ord('s'):
                cv2.imwrite(str(i) + '.jpg', frame)
                msg = 'image saved'
                logging.info(msg)
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def start_video_thread(self):
        self.th=threading.Thread(target=self.video_loop(),args=())
        self.th.setDaemon(True)#守护线程
        self.th.start()

    def video_loop(self):
        canvas1 = Canvas(self.fm3, bg='white', width=640, height=480)
        canvas1.pack()
        try:
            while True:
                pic1 = self.tkImage(vc1)
                canvas1.creat_image(0,0,anchor='nw',image=pic1)
                self.fm3.update_idletasks()
                self.fm3.update()
        except:
            pass

    def tkImage(self,vc):
        ref, frame = vc.read()
        cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(cvimage)
        pilImage = pilImage.resize((image_width, image_height), Image.ANTIALIAS)
        tkImage = ImageTk.PhotoImage(image=pilImage)
        return tkImage


class TextHandler(logging.Handler):
    # 在ScrolledText widget窗口输出日志

    def __init__(self, text):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Store a reference to the Text it will log to
        self.text = text

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text.configure(state='normal')
            self.text.insert(tk.END, msg + '\n')
            self.text.configure(state='disabled')
            # Autoscroll to the bottom
            self.text.yview(tk.END)

        # This is necessary because we can't modify the Text from other threads
        self.text.after(0, append)




def worker():
    # 在另一线程运行
    while True:
        # Report time / date at 2-second intervals
        time.sleep(2)
        timeStr = time.asctime()
        msg = 'Current time: ' + timeStr
        logging.info(msg)


if __name__ == '__main__':
    app = Application()


    app.mainloop()

    vc1.release()
    cv2.destroyAllWindows()
