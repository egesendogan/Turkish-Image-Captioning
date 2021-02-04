#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 18:45:50 2021

@author: ege
"""

import tensorflow as tf

import collections
import random
import numpy as np
import time
from prepare_data import prepare_data
from preprocess_data import extract_features, preprocess_captions, load_model
from models import CNN_Encoder, RNN_Decoder
from train_utils import create_ckpt_manager, train_step, plt_loss, plot_image_and_caption

def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap

if __name__ == '__main__':
    img_name_vector, train_captions = prepare_data()
    
    extract_features(img_name_vector)
    top_k = 5000
    cap_vector, max_length, tokenizer = preprocess_captions(train_captions,top_k)
    
    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)
    
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys)*0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]
    
    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])
        
    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])
    
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    vocab_size = top_k + 1
    num_steps = len(img_name_train) // BATCH_SIZE

    features_shape = 2048
    attention_features_shape = 64
    
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    ckpt, ckpt_manager = create_ckpt_manager(encoder,decoder,optimizer)
    
    start_epoch = 0
    '''if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)'''
    
    loss_plot = []
    
    EPOCHS = 25

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target, encoder, decoder, 
                                        optimizer, tokenizer, loss_object)
        total_loss += t_loss

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
        ckpt_manager.save()

    print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
    plt_loss(loss_plot)
    
    plot_image_and_caption(img_name_val,cap_val,encoder,decoder,tokenizer,max_length,attention_features_shape)
    