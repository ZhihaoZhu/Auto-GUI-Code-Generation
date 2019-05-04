#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2
import shutil

from model.classes.model.Config import *


## Defined Functions
def content_transform(string):
    bl_token = []
    tl_token = []
    stack = 0
    tokens = string.split(" ")
    bl_token.append(tokens[0])
    for i in range(1,len(tokens)-1):
        if stack==0:
            if tokens[i] != "{":
                bl_token.append(tokens[i])
                continue
            elif tokens[i] == "{":
                stack += 1
        else:
            if tokens[i] == "}":
                stack -= 1
                if stack != 0:
                    tl_token.append(tokens[i])
            elif tokens[i] == "{":
                stack += 1
                tl_token.append(tokens[i])
            else:
                if tokens[i][-1] == ",":
                    tl_token.append(tokens[i][:-1])
                else:
                    tl_token.append(tokens[i])
    return bl_token, tl_token


# arg = sys.argv[1:]
# platform = arg[0]
# distribution = arg[1]
platform = "web"
distribution = 6

if not os.path.isdir("../datasets/{}/train_data".format(platform)):
    os.mkdir("../datasets/{}/train_data".format(platform))

if not os.path.isdir("./datasets/{}/val_data".format(platform)):
    os.mkdir("../datasets/{}/val_data".format(platform))

if not os.path.isdir("../datasets/{}/test_data".format(platform)):
    os.mkdir("../datasets/{}/test_data".format(platform))

DISTRIBUTION = 6 if not distribution else distribution

path = []
for f in os.listdir("../datasets/{}/all_data".format(platform)):

    if f.find(".gui") != -1:
        file_name = f[:f.find(".gui")]
        path.append(file_name)

        ##----------Split the document and tokenize each word----------------##
        with open("../datasets/{}/all_data/{}".format(platform,f),'r') as file:
            char = ""
            for line in file:
                char += line
            char = char.replace("\n", " ")
            bl_token, tl_token = content_transform(char)

        image_path = "../datasets/{}/all_data/{}.png".format(platform,file_name)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (256, 256))
        img = img.astype('float32')
        img /= 255

        ##-----------Save the resized images and tokens into npz-----------------##
        np.savez_compressed("../datasets/{}/all_data/{}".format(platform, file_name),
                            bl_token=bl_token, tl_token=tl_token, img = img)

##-----------split the data-----------------##
val_size = len(path)//(DISTRIBUTION+1)
np.random.shuffle(path)
val_data = path[:val_size]
test_data = path[val_size:val_size*2]
train_data = path[2*val_size:]
search_index = {0:"val_data", 1:"test_data", 2:"train_data"}

##-----------copy the data into corresponding folders-----------------##
for index, set in enumerate([val_data, test_data, train_data]):
    for file_name in set:
        shutil.copyfile("../datasets/{}/all_data/{}.npz".format(platform, file_name),
                    "../datasets/{}/{}/{}.npz".format(platform, search_index[index], file_name))

