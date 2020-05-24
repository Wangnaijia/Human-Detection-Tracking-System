#-*- coding:utf-8 -*-
from Tkinter import *
import tkFileDialog
from tkMessageBox import *
import cv2
from detect_for_tracker import detect_with_rects
from stereo_measure import runStereoCam
from PIL import Image,ImageTk
from threading import Thread
import time
import numpy as np
from kill_thread import stop_thread
from trackers.eco import ECOTracker
from runECO import ecotrack
from imutils.object_detection import non_max_suppression


def encode_region(region):
    region = (region[0], region[1],
            region[2] - region[0], region[3] - region[1])
    # output as integer
    return '{:.0f} {:.0f} {:.0f} {:.0f}'.format(region[0], region[1], region[2], region[3])



class App:
    def __init__(self):
        self.t1 = ''
        self.t2 = ''
        self.ts = []
        self.path = ''
        self.win = Tk()
        self.win.title("human detect and tracking on PC")
        self.win.geometry('1200x800+200+100')
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas = ''
        self.detectFlag = 1
        self.trackFlag = 0

        # 菜单栏
        self.menu = Menu(self.win)
        self.win.config(menu=self.menu)
        self.fileabout = Menu(self.menu)
        self.menu.add_cascade(label="About",menu=self.fileabout)
        self.fileabout.add_command(label="Help",command=self.hlp)
        self.fileabout.add_command(label="Exit",command=self.ex)

        # 功能选择
        self.detect_btn = Button(self.win, text='检测任务', font='宋体 16 bold', command=self.detect)
        self.detect_btn.place(x=30, y=50, width=150, height=50)

        self.track_btn = Button(self.win, text='跟踪任务', font='宋体 16 bold', command=self.track)
        self.track_btn.place(x=30, y=130, width=150, height=50)
        # 相机按钮
        self.camera_btn = Button(self.win, text='相机', font='宋体 16 bold', command=self.camera)
        self.camera_btn.place(x=30, y=210, width=150, height=50)

        # 路径输入框
        # self.path_entry = Text(self.win, font='宋体 16')
        # self.path_entry.place(x=20, y=110, width=150, height=80)

        # 加载视频，这个视频的路径从上面输入框中获取
        self.video_btn = Button(self.win, text='加载视频', font='宋体 16 bold', command=self.video)
        self.video_btn.place(x=30, y=290, width=150, height=50)

        # 双目相机抓图
        self.stereo_btn = Button(self.win, text='双目画面',font='宋体 16 bold',command=self.stereo_cam)
        self.stereo_btn.place(x=30, y=370, width=150, height=50)


        self.fps_l = Label(self.win, text="帧率", font='宋体 16 bold')
        self.fps_l.place(x=30, y=450)
        # "FPS:%f" % (fps)

        self.label = Label(self.win, text="目标位置", font='宋体 16 bold')
        self.label.place(x=30, y=500)

        self.pick_l = Label(self.win, text="", anchor=NW)
        self.pick_l.place(x=30, y=550, width=150, height=100)
        # 自定义logger模块方便输出
        self.label = Label(self.win, text="输出日志", font='宋体 16 bold')
        self.label.place(x=30, y=650)
        self.logger = Label(self.win, text="", anchor='w', wraplength=150)
        self.logger.place(x=30, y=680, width=150, height=150)
        self.logger['text'] = "welcome!"


        self.run()

    def camera(self):
        try:
            self.capture2.release()
            cv2.destroyAllWindows()
        except:
            pass
        try:
            while self.ts:
                stop_thread(self.ts.pop())
        except:
            pass
        self.capture1 = cv2.VideoCapture(0)
        if self.detectFlag:
            self.t2 = Thread(target=self.open_detector, args=(self.capture1,))
            self.ts.append(self.t2)
            self.t2.start()
        else:
            self.t2 = Thread(target=self.open_tracker, args=(self.capture1,))
            self.ts.append(self.t2)
            self.t2.start()

    def video(self):
        try:
            self.capture1.release()
            cv2.destroyAllWindows()
        except:
            pass
        try:
            while self.ts:
                stop_thread(self.ts.pop())
        except:
            pass
        self.path = tkFileDialog.askopenfilename(title="打开单个文件", filetypes=[("全部文件", "*.*")], initialdir="E:\\")
        # self.path_entry.insert(0.0, self.path)

        # capture = cv.VideoCapture('007.avi')  # 打开摄像头，0代表的是设备id，如果有多个摄像头，可以设置其他数值
        self.capture2 = cv2.VideoCapture(self.path)

        if self.detectFlag:
            self.t1 = Thread(target=self.open_detector, args=(self.capture2,))
            self.ts.append(self.t1)
            self.t1.start()
        else:
            self.t1 = Thread(target=self.open_tracker, args=(self.capture2,))
            self.ts.append(self.t1)
            self.t1.start()

    def detect(self):
        self.logger['text'] = "choose to detect people in a video"
        self.detectFlag = 1


    def track(self):

        self.logger['text'] = "choose to track people in a video"
        self.detectFlag = 0

    def open_detector(self, capture):
        if self.canvas != '':
            self.canvas.delete(ALL)
        # 画框，用于显视用户打开的图像
        self.canvas = Canvas(self.win, background="gray", width=1000, height=760, bg="white")
        self.canvas.place(x=200, y=10)
        # 帧率统计
        fps = 0
        fps_result = []
        # 检测人数统计
        detect_num = []
        # 帧数
        frames = 0

        while True:
            t1 = time.time()
            ret, frame = capture.read()  # 读取摄像头,它能返回两个参数，第一个参数是bool型的ret，其值为True或False，代表有没有读到图片；第二个参数是frame，是当前截取一帧的图片
            if ret==0:
                break
            pick, img, num = detect_with_rects(frame, cv2)

            fps = (fps + (1. / (time.time() - t1))) / 2
            fps_result.append(fps)
            detect_num.append(num)
            frames += 1
            try:
                current_image = Image.fromarray(img)  # 将图像转换成Image对象
            except Exception as e:
                self.logger['text'] = e
            width, height = current_image.size
            # print(width, height)
            ratio = width / height
            if ratio < 1:
                height2 = 730
                width2 = int(730/height*width)
            else:
                width2 = 970
                height2 = int(970/width*height)
            current_image = current_image.resize((width2, height2), Image.ANTIALIAS)
            try:
                imgtk = ImageTk.PhotoImage(image=current_image)
            except Exception as e:
                break
            self.canvas.create_image((1000-width2)/2, (760-height2)/2, anchor=NW, image=imgtk)
            self.pick_l['text'] = pick
            self.fps_l['text'] = "FPS:%f" % (fps)
            # 键盘输入q退出窗口，不按q点击关闭会一直关不掉 也可以设置成其他键。
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()

    def open_tracker(self, capture):
        if self.canvas != '':
            self.canvas.delete(ALL)
        # 画框，用于显示用户打开的图像
        self.canvas = Canvas(self.win, background="gray", width=1000, height=760, bg="white")
        self.canvas.place(x=200, y=10)
        # 目标框
        bbox = []
        ok, frame = capture.read()

        imheight, imwidth = frame.shape[:2]
        center = imwidth
        if len(frame.shape) == 3:
            is_color = True
        else:
            is_color = False
            frame = frame[:, :, np.newaxis]
        # starting tracking
        tracker = ECOTracker(is_color)
        #
        while (True):
            ok, frame = capture.read()
            pick = self.hog_detect(frame)
            # check the first frame, if no object, do the next frame
            for (x, y, w, h) in pick:
                if abs(x + w / 2 - imwidth / 2) + abs(y + h / 2 - imheight / 2) < center:
                    center = abs(x + w / 2 - imwidth / 2) + abs(y + h / 2 - imheight / 2)
                    bbox = (x, y, w, h)
            if bbox:
                # initialize the tracker with frame[0] and bbox
                tracker.init(frame, bbox)
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            current_image = Image.fromarray(frame)  # 将图像转换成Image对象
            width, height = current_image.size
            # print(width, height)
            ratio = width / height
            if ratio < 1:
                height2 = 730
                width2 = int(730 / height * width)
            else:
                width2 = 970
                height2 = int(970 / width * height)
            current_image = current_image.resize((width2, height2), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=current_image)
            self.canvas.create_image((1000 - width2) / 2, (760 - height2) / 2, anchor=NW, image=imgtk)

        idx = 0
        fps_result = []

        while True:

            ret, frame = capture.read()  # 读取摄像头,它能返回两个参数，第一个参数是bool型的ret，其值为True或False，代表有没有读到图片；第二个参数是frame，是当前截取一帧的图片
            if ret==0:
                break
            frame, bbox, fps = ecotrack(frame, idx, tracker, imheight, imwidth)
            fps_result.append(fps)
            idx += 1
            try:
                current_image = Image.fromarray(frame)  # 将图像转换成Image对象
            except Exception as e:
                self.logger['text'] = e
            width, height = current_image.size
            # print(width, height)
            ratio = width / height
            if ratio < 1:
                height2 = 730
                width2 = int(730/height*width)
            else:
                width2 = 970
                height2 = int(970/width*height)
            current_image = current_image.resize((width2, height2), Image.ANTIALIAS)

            try:
                imgtk = ImageTk.PhotoImage(image=current_image)
            except Exception as e:
                break
            self.canvas.create_image((1000-width2)/2, (760-height2)/2, anchor=NW, image=imgtk)
            self.pick_l['text'] = encode_region(bbox)
            self.fps_l['text'] = "FPS:%f" % (fps)

        tracker.quit()
        capture.release()

    def hog_detect(self, frame):
        '''
        :param frame: 初始帧
        :param model_path: 用于HOGDescriptor的SVM检测器
        :return: 检测到的最中心的人体目标bbox
        '''

        # opencv提取hog特征
        hog = cv2.HOGDescriptor()
        # opencv自带的训练好了的分类器
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # 构造了一个尺度scale=1.05的图像金字塔
        rects, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        return pick


    def ex(self):
        if askokcancel("Quit", "Do you really want to quit?"):
            while self.ts:
                stop_thread(self.ts.pop())
            try:
                self.capture1.release()
            except:
                pass
            try:
                self.capture2.release()
            except:
                pass
            self.win.destroy()
            cv2.destroyAllWindows()

    def hlp(self):
        showinfo("HELP", "First::You can click either to detect or to track, if no choice, detect in the video.\n\nSecond::camera and video file is accessable using button.\n\nThird::a module to capture stereo cam picture is designed(attention:only for photo capture)")


    def stereo_cam(self):
        width = 640
        height = 480
        usb_camera = True
        width *= 2  # 双目摄像头，总宽度
        cap = cv2.VideoCapture(int(usb_camera))  # VideoCapture()中参数是1，表示打开外接usb摄像头
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置摄像头的分辨率，宽
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 设置摄像头的分辨率，高
        cam_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cam_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cam_fps = cap.get(cv2.CAP_PROP_FPS)
        ret,_ = cap.read()
        if ret==0:
            self.logger['text']="Check the if the camera is connected!"
        else:
            self.logger['text'] = "Run stereo camera,command 'esc' used to leave.\n\n摄像头信息：\n宽：{}\n高：{}\n帧率：{}\n".format(cam_w,cam_h,cam_fps)
            time.sleep(0.02)
            runStereoCam(cap,cv2)


    def close(self):
        while self.ts:
            stop_thread(self.ts.pop())
        try:
            self.capture1.release()
        except:
            pass
        try:
            self.capture2.release()
        except:
            pass
        self.win.destroy()
        cv2.destroyAllWindows()

    def run(self):
        self.win.mainloop()


if __name__ == '__main__':
    app = App()
