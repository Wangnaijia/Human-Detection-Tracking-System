# -*- coding:utf-8 -*-
# Tested on windows machines

import cv2, webbrowser
from Tkinter import *
from tkMessageBox import *
import tkFileDialog
from io import StringIO
from PIL import Image, ImageTk

root = Tk()
root.title("human detecting and tracking sys")
root.geometry('640x480')
bcolor = 'black'
fcolor = 'white'

fm1=Frame(root)
fm2=Frame(root)
fm3=Frame(root)

def selectPath():
    global path_
    path_ = tkFileDialog.askopenfilename()
    path.set(path_)

path = StringVar()
root.configure(bg=bcolor)
root.resizable(0, 0)
root.option_add('*Dialog.msg', 'Perpetua 12')
write_video_flag=0
# 初始化图窗口
panel=Label(root)
panel.pack(padx=10,pady=10)

def ex():
    if askokcancel("Quit", "Do you really want to quit?"):
        root.destroy()


root.protocol('WM_DELETE_WINDOW', ex)


def hlp():
    showinfo("HELP",
             "Step1::You must enter file name (else:: noname file saved)\n\nStep2::Command : 's' Used for take shot\n\n Step3::Command : 'esc' Used for leave\n\n Notification ::: \n\n       Video Camera will be automaticaly record while start video cam.")



def cam():
    p1 = imname.get()
    cap = cv2.VideoCapture(0)
    i = 0
    while (True):

        ret, frame = cap.read()
        cv2.imshow('Camera', frame)
        k = cv2.waitKey(1)
        if k == ord('s'):
            i += 1
            cv2.imwrite(p1 + str(i) + '.jpg', frame)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def video_loop():
    # 视频来源
    cap = cv2.VideoCapture(0)
    success,img=cap.read()
    if success:
        cvimage=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)
        current_image = Image.fromarray(cvimage)
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk=imgtk
        panel.config(image=imgtk)
        root.after(1,video_loop)

        if write_video_flag:
            p1 = imname.get()
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(p1 + '.mp4', fourcc, 20.0, (640, 480))

            while (cap.isOpened()):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                out.write(frame)
                cv2.imshow('Video', frame)
                k = cv2.waitKey(1)
                if k == 27:
                    break

            out.release()
    cap.release()
    cv2.destroyAllWindows()


menu = Menu(root)
root.config(menu=menu)
filemenu = Menu(menu)
fileabout = Menu(menu)

menu.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="Video_Camera", command=video_loop)
filemenu.add_command(label="Camera", command=cam)
filemenu.add_command(label="Exit", command=ex)

menu.add_cascade(label="About", menu=fileabout)
fileabout.add_command(label="Help", command=hlp)

Label(root, text='Welcome', font='Magneto 25 bold ', bg=bcolor, fg=fcolor).place(x=180, y=0)  # 34383C, Stencil

Label(root, text="Save_As: ", bg=bcolor, font='System 15 bold', fg=fcolor).place(x=120, y=90)  # , bg='gray'
imname = Entry(root, font=(10), borderwidth=0, fg='#34383C')
imname.place(x=200, y=90)

vidbt = Button(root, text="Video_Cam", borderwidth=1, bg=bcolor, font='System 15 bold', fg=fcolor, command=video_loop)
vidbt.place(x=160, y=140)
cmbt = Button(root, text="Camera", borderwidth=1, bg=bcolor, font='System 15 bold', fg=fcolor, command=cam)
cmbt.place(x=260, y=140)
exbt = Button(root, text="Exit", borderwidth=1, bg=bcolor, font='System 15 bold', fg=fcolor, command=ex)
exbt.place(x=335, y=140)

root.mainloop()
cv2.destroyAllWindows()