# -*- coding: utf-8 -*-

import PIL
import time
import numpy as np
import tensorflow as tf

inImgPath = "data/Lenna.png"
outImgPath = "data/Lenna2.png"
logPath = "log"

# Define the TF-graph of image process flow
def imageProcessFlow(img):
    img = tf.image.convert_image_dtype(img, tf.float32, saturate=True)
    img = tf.image.central_crop(img, 0.5)
    img = tf.image.random_brightness(img, 0.5)
    img = tf.image.resize_images(img, [299, 299])
    img = tf.image.convert_image_dtype(img, tf.uint8, saturate=True)
    return img

# Process image (R/W by binary string)
def imageTest1(inImgPath, outImgPath):
    with open(inImgPath, "rb") as file:
        imgstr = file.read()
    img = tf.image.decode_png(imgstr) # Decode image from binary string
    img = imageProcessFlow(img)
    img = tf.image.encode_png(img) # Encode image to binary string
    with tf.Session() as sess:
        #tf.summary.merge_all()
        #writer = tf.summary.FileWriter(logPath, sess.graph)
        #writer.close()
        imgstr = sess.run(img)
    with open(outImgPath, "wb") as file:
        file.write(imgstr)

# Process image (R/W by numpy array)
def imageTest2(inImgPath, outImgPath):
    img = np.asarray(PIL.Image.open(inImgPath))
    with tf.Session() as sess:
        img2 = sess.run(imageProcessFlow(img))
    PIL.Image.fromarray(img2).save(outImgPath)

if __name__ == "__main__":
    start_time = time.time()
    imageTest1(inImgPath, outImgPath)
    print("--- %.2f seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    imageTest2(inImgPath, outImgPath)
    print("--- %.2f seconds ---" % (time.time() - start_time))
