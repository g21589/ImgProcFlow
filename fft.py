# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

refImgPath = "data/stp2.png"
defImgPath = "data/stp2_d1.png"

def genMagnitudeSpectrum(img, pkg='cv'):
    return {
        'np': genMagnitudeSpectrum_np(img),
        'cv': genMagnitudeSpectrum_cv(img),
        'tf': genMagnitudeSpectrum_tf(img)
    }[pkg]

def genMagnitudeSpectrum_np(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    result = 20 * np.log(np.abs(fshift))
    return result

def genMagnitudeSpectrum_cv(img):
    f = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(f)
    result = 20 * np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]))
    return result

def genMagnitudeSpectrum_tf(img):
    data = tf.placeholder(np.uint8, name='input')
    real = tf.cast(data, tf.float32, name='cast_to_float32')
    imag = tf.constant(0.0, dtype=tf.float32)
    cast = tf.complex(real, imag, name='cast_to_complex')
    fftOp = tf.fft(cast, name='fft')
    with tf.Session() as sess:
        f = sess.run(fftOp, feed_dict={data: img})
    fshift = np.fft.fftshift(f)
    result = 20 * np.log(np.abs(fshift))
    return result

if __name__ == "__main__":
    img = cv2.imread(refImgPath, 0)
    magnitude_spectrum = genMagnitudeSpectrum(img, pkg='tf')
    
    img2 = cv2.imread(defImgPath, 0)
    magnitude_spectrum2 = genMagnitudeSpectrum(img2, pkg='cv')
    
    plt.subplot(221), plt.imshow(img, cmap = 'gray')
    plt.title('Image A'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('FFT Image A'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img2, cmap = 'gray')
    plt.title('Image B'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(magnitude_spectrum2, cmap = 'gray')
    plt.title('FFT Image B'), plt.xticks([]), plt.yticks([])
    plt.show()
