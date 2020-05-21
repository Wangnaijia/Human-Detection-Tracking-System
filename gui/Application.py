# -*- coding:utf-8 -*-
from Tkinter import *
from PIL import Image, ImageTk
import threading
import imageio
import logging
import time
import cv2


class Application(Frame):
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
        # fm1
        self.fm1 = Frame(self, bg='black')
        self.titleLabel = Label(self.fm1, text="human detecting and tracking system", font=('Dialog', 24), fg="white",
                                bg='black')
        self.titleLabel.pack()
        self.fm1.pack(side=TOP, fill='x', pady=2)

        # fm2
        self.fm2 = Frame(self, bg='green')
        self.fm2_left = Frame(self.fm2, bg='black')
        self.fm2_right = Frame(self.fm2, bg='black')
        self.fm2_left_top = Frame(self.fm2_left, bg='black')
        self.fm2_left_bottom = Frame(self.fm2_left, bg='black')

        self.predictEntry = Entry(self.fm2_left_top, font=('Dialog', 16), width='72', fg='#FF4081')
        self.predictButton = Button(self.fm2_left_top, text='predict sentence', bg='#FF4081', fg='white',
                                    font=('Dialog', 16), width='16', command=self.output_predict_sentence)
        self.predictButton.pack(side=LEFT)
        self.predictEntry.pack(side=LEFT,  padx=5)
        self.fm2_left_top.pack(side=TOP, fill='x')

        self.truthEntry = Entry(self.fm2_left_bottom, font=('Dialog', 16), width='72', fg='#22C9C9')
        self.truthButton = Button(self.fm2_left_bottom, text='ground truth', bg='#22C9C9', fg='white',
                                  font=('Dialog', 16), width='16', command=self.output_ground_truth)
        self.truthButton.pack(side=LEFT)
        self.truthEntry.pack(side=LEFT, padx=5)
        self.fm2_left_bottom.pack(side=TOP, fill='x', pady=5)

        self.fm2_left.pack(side=LEFT, padx=5, pady=5, expand=YES, fill='x')
        load=Image.open('./detector/expr/airport1.jpeg')
        self.nextVideoImg = ImageTk.PhotoImage(load)
        self.nextVideoButton = Button(self.fm2_right, image=self.nextVideoImg, text='next video', bg='black',
                                      command=self.start_play_video_thread)
        self.nextVideoButton.pack(expand=YES, fill=BOTH)
        self.fm2_right.pack(side=RIGHT, padx=5)
        self.fm2.pack(side=TOP, expand=YES, fill="x",pady=2)

        #fm3
        self.fm3=Frame(self,bg='black')
        load=Image.open('./detector/expr/airport1.jpeg')
        initImage=ImageTk.PhotoImage(load)
        self.panel=Label(self.fm3,image=initImage)
        self.panel.image=initImage
        self.panel.pack()
        self.fm3.pack(side=TOP,expand=YES,fill=BOTH,pady=10)

    def output_predict_sentence(self):
        predicted_sentence_str='hello'
        self.predictEntry.delete(0,END)
        self.predictEntry.insert(0,predicted_sentence_str)

    def output_ground_truth(self):
        ground_truth='this is ground truth'
        self.truthEntry.delete(0,END)
        self.truthEntry.insert(0,ground_truth)

    def start_play_video_thread(self):
        self.thread=threading.Thread(target=self.play_next_video,args=())
        self.thread.start()

    def play_next_video(self):
        self.predictEntry.delete(0,END)
        self.truthEntry.delete(0,END)

        #to play video
        self.video_path='/home/wnj/projects/videos/007.avi'
        self.video=imageio.get_reader(self.video_path,'ffmpeg')
        for self.videoFrame in self.video:
            self.image=cv2.cvtColor(self.videoFrame,cv2.COLOR_BGR2RGB)
            self.image=Image.fromarray(self.image)
            self.image=ImageTk.PhotoImage(self.image)

            self.panel.configure(image=self.image)
            self.panel.image=self.image
            time.sleep(0.02)

if __name__=='__main__':
    app=Application()
    app.mainloop()
