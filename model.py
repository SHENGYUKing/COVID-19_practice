# -*- coding: utf-8 -*-

# Copyright 2020. SHENGYUKing.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The dataset comes from https://github.com/ieee8023/covid-chestxray-dataset

import os
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageEnhance

LIST_PATH = "./covid-chestxray-dataset/metadata.csv"
FILE_PATH = "./covid-chestxray-dataset/images/"
SAVE_PATH = "./data/images_ori/"
DATA_PATH = "./data/images_pro/"


def preprocess(data, mod='one'):
    if mod is 'none':
        return data
    if mod is 'max':
        return data / 255.0


def read_database(listpath=LIST_PATH, filepath=FILE_PATH, savepath=SAVE_PATH):
    print("Begin to extract the samples...")
    table = pd.read_csv(listpath)
    table = table[~table['modality'].isin(['CT'])]
    table = table[~table['view'].isin(['L'])]
    files = table['filename'].tolist()
    labels = table['finding'].tolist()
    for i in range(0, len(files)):
        img = Image.open(filepath + files[i])
        img_grey = img.convert('L')
        new = savepath + str(i + 1) + '_' + labels[i] + '.jpg'
        img_grey.save(new)
    print("Original samples have been extracted.")


def extract_database(filepath=SAVE_PATH, newfile=DATA_PATH, enhance=True):
    print("Begin to crop the samples...")
    if enhance:
        print("Data-enhance switch is ON.")
    file_names = os.listdir(filepath)

    for file in file_names:
        file_path = os.path.join(SAVE_PATH, file)
        img = Image.open(file_path)
        n = 500.0 / min(img.size[0], img.size[1])
        x, y = round(img.size[0] * n), round(img.size[1] * n)
        new_size = (x, y)
        crop_size = (x/2 - 250, 0, x/2 + 250, 500)
        img_new = img.resize(new_size).crop(crop_size)
        img_new.save(newfile + file)

        if enhance:
            img_bri_down = ImageEnhance.Brightness(img_new).enhance(0.8)
            img_bri_up = ImageEnhance.Brightness(img_new).enhance(1.2)
            img_shp_down = ImageEnhance.Sharpness(img_new).enhance(0.8)
            img_shp_up = ImageEnhance.Sharpness(img_new).enhance(1.2)
            img_flip = img_new.transpose(Image.FLIP_LEFT_RIGHT)

            img_bri_down.save(newfile + 'bri_down_' + file)
            img_bri_up.save(newfile + 'bri_up_' + file)
            img_shp_down.save(newfile + 'shp_down_' + file)
            img_shp_up.save(newfile + 'shp_up_' + file)
            img_flip.save(newfile + 'flip_' + file)
    print("Samples have been cropped.")
    print("Done.")


def load_database(filepath, pretreatment='max'):
    print("Begin to load the database...")
    file_names = os.listdir(filepath)
    database = np.zeros((1, 1 + 500 * 500))
    for file in file_names:
        file_path = os.path.join(filepath, file)
        img = Image.open(file_path)
        data = preprocess(np.asarray(img), pretreatment).reshape(1, -1)
        if 'COVID-19' in file:
            label = np.array([[1]])
        else:
            label = np.array([[0]])
        unit = np.concatenate((label, data), axis=1)
        database = np.concatenate((database, unit), axis=0)
    print("Database has been generated.")
    database_out = np.delete(database, 0, 0)
    # print(database_out[0:4])

    return database_out
