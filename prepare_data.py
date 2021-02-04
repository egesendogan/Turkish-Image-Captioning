#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:07:36 2021

@author: ege
"""


import collections
import os
import json

annotation_folder = '/_annotations/'
annotation_file = os.path.abspath('.')+'/_annotations/captions_train2014_tr.json'

def prepare_data():

    image_folder = '/archive/Images/'
    PATH = os.path.abspath('.') + image_folder

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    for annotation in annotations:
        img_name = annotation['file_path'][-31:]
        new_path = img_name
        annotation['file_path'] = new_path
    
    image_path_to_caption = collections.defaultdict(list)
    for annotation in annotations:
        for cap in annotation['captions']:
            caption = f"<start> {cap} <end>"

            img_path = PATH + annotation['file_path']
            if(os.path.exists(img_path)):
                image_path_to_caption[img_path].append(caption)

    train_image_paths = list(image_path_to_caption.keys())
    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))
        
    return img_name_vector, train_captions