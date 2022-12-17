import os
import pickle
import numpy as np
from IrisSegmentation import IrisSeg
import random

labels = ["aeva","fiona","hock","kelvin","lec","liujw","loke","lowyf","lpj","mahsk","maran","bryan","mas","mazwan","mimi","mingli","ngkokwhy","nkl","noraza","norsuhaidah","ongbl","pcl","chingyc","philip","rosli","sala","sarina","siti","suzaili","tanwn","thomas","tick","tingcy","tonghl","vimala","weecm","win","yann","zaridah","zulaikah","chongpk","christine","chuals","eugeneho","fatma"]
directory = 'C:/Users/yhya2/Desktop/Iris-project/MMU-Iris-Database'
training_data = []

for files in os.listdir(directory):
        for side in os.listdir(f"{directory}/{files}"):
            for imgName in os.listdir(f"{directory}/{files}/{side}"):
                if imgName.endswith(".bmp"):
                        class_num = labels.index(imgName[:-6])
                        path = os.path.join(f"{directory}/{files}/{side}/{imgName}")
                        img, _ = IrisSeg(path)
                        img = img.flatten()
                        training_data.append([img, class_num])

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 10000, 1)
y = np.array(y)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close
