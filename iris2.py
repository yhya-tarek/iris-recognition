import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from IrisSegmentation import IrisSeg

labels = ["aeva","fiona","hock","kelvin","liujw","loke","lowyf","lpj","mahsk","maran","bryan","mas","mazwan","mimi","mingli","ngkokwhy","nkl","noraza","norsuhaidah","ongbl","pcl","chingyc","philip","rosli","sala","sarina","siti","suzaili","tanwn","thomas","tick","tingcy","tonghl","vimala","win","yann","zaridah","chongpk","christine","chuals","eugeneho","fatma"]
directory = 'C:/Users/RAHEEM/Downloads/archive/MMU-Iris-Database'

img_dataset = pd.DataFrame()
pixels = []
imgNo = 1
id = 0

for files in os.listdir(directory):
        for side in os.listdir(f"{directory}/{files}"):
            limimg = 0
            for imgName in os.listdir(f"{directory}/{files}/{side}"):
                if imgName.endswith(".bmp"):
                    path = os.path.join(f"{directory}/{files}/{side}/{imgName}")
                    img = IrisSeg(path)
                    img=cv2.imread(path)
                    cv2.imwrite(f"ImageNo{imgName}.png", img)
                    img = img.flatten()
                    img = np.append(img, id)
                    os.remove(f"{path}")
                    df = pd.DataFrame({f"ImageNo{imgNo}" : img})
                    img_dataset.insert(len(img_dataset.columns), f"ImageNo{imgNo}", img)
                    newPath=path.split('.bmp')
                    cv2.imwrite(f"{newPath[0]}.png",img)
                    imgNo += 1
                    limimg += 1
          
        id+=1

img_dataset.to_csv('imagees.csv', index= False, header= False, mode="wb", sep=",")
