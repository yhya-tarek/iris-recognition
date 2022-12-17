from tkinter import *
from tkinter import filedialog
# Import required libraries
from tkinter import *
from PIL import ImageTk, Image
import cv2
from IrisSegmentation import IrisSeg 
import tkinter
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from tensorflow import keras

root = Tk()
root.title("IRIS RECOGNITION")
l = Label(root, text = "IRIS RECOGINTION SYSTEM",fg='black')
l.config(font =("Courier", 22))

l.pack()
root.geometry("1000x800")
#set window color
root['bg']='gold'

def open():
    global my_image
    global my_image_label
    root.filename = filedialog.askopenfilename(
        initialdir="/", title="Select a File", filetypes=(("png files", ".png"), ("all files", ".*")))
    my_label = Label(root, text='Input Image').pack()
    my_image = PhotoImage(file=root.filename).subsample(1)
    my_image_label = Label(image=my_image).place(x=180, y=60).pack()


def featureExtraction():
    result, img=IrisSeg(root.filename)
    result = cv2.resize(result,  (320,240))
    img = cv2.resize(img,  (320,240))
    test = ImageTk.PhotoImage(image=Image.fromarray(result))
    
    label1 = tkinter.Label(image=test)
    label1.image = test

  # Position image
    label1.place(x=100, y=360)

    test1 = ImageTk.PhotoImage(image=Image.fromarray(img))
    label2 = tkinter.Label(image=test1)
    
    label2.image = test1

  # Position image
    label2.place(x=500, y=360)

def predict():
    X, _ = IrisSeg(root.filename)

    X = X.flatten()
    X = np.array(X)
    X = X.reshape(1, 10000, -1)
    model = keras.models.load_model('iris-recognition\wh.model')
    prediction = model.predict(X)
    l = Label(root, text = f"the result : {np.argmax(prediction)}",fg='black')
    l.config(font =("Courier", 22))
    l.pack()
    l.place(x = 300,y = 700)


my_btn = Button(root, text="Browse Image", command=open, width=20).place(x=20,y=30)

my_btn1 = Button(root, text='Feature Extraction',command=featureExtraction, width=20).place(x=20, y=320)
my_btn1 = Button(root, text='show Result', width=20, command = predict).place(x=20, y=720)
root.mainloop()

