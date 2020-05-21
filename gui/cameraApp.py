# This program develope by <<<---Thamizhan_PS--->>>
# Tested on windows machines

import cv2, webbrowser
from Tkinter import *
from tkMessageBox import *
from tkFileDialog import *
from io import StringIO
from PIL import ImageTk

camera = Tk()
camera.title("TPS_Camera")
camera.geometry('500x250+280+200')
bcolor = 'black'
fcolor = 'red'
camera.configure(bg=bcolor)
camera.resizable(0, 0)
camera.option_add('*Dialog.msg.font', 'Perpetua 12')


def ex():
    if askokcancel("Quit", "Do you really want to quit?"):
        camera.destroy()


#camera.protocol('WM_DELETE_WINDOW', ex)


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


def vidcam():
    p1 = imname.get()
    cap = cv2.VideoCapture(0)
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
    cap.release()
    out.release()
    cv2.destroyAllWindows()


menu = Menu(camera)
camera.config(menu=menu)
filemenu = Menu(menu)
fileabout = Menu(menu)

menu.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="Video_Camera", command=vidcam)
filemenu.add_command(label="Camera", command=cam)
filemenu.add_command(label="Exit", command=ex)

menu.add_cascade(label="About", menu=fileabout)
fileabout.add_command(label="Help", command=hlp)

Label(camera, text='Welcome', font='Magneto 25 bold ', bg=bcolor, fg=fcolor).place(x=180, y=0)  # 34383C, Stencil

Label(camera, text="Save_As: ", bg=bcolor, font='System 15 bold', fg=fcolor).place(x=120, y=90)  # , bg='gray'
imname = Entry(camera, font=(10), borderwidth=0, fg='#34383C')
imname.place(x=200, y=90)

vidbt = Button(camera, text="Video_Cam", borderwidth=1, bg=bcolor, font='System 15 bold', fg=fcolor, command=vidcam)
vidbt.place(x=160, y=140)
cmbt = Button(camera, text="Camera", borderwidth=1, bg=bcolor, font='System 15 bold', fg=fcolor, command=cam)
cmbt.place(x=260, y=140)
exbt = Button(camera, text="Exit", borderwidth=1, bg=bcolor, font='System 15 bold', fg=fcolor, command=ex)
exbt.place(x=335, y=140)
Label(camera, text='Created By-> #Thamizhan_PS', font='Perpetua 10', bg=bcolor, fg=fcolor).place(x=0, y=210)
Label(camera, text='Version : 1.0.1', font='Perpetua 10', bg=bcolor, fg=fcolor).place(x=420, y=210)

camera.mainloop()