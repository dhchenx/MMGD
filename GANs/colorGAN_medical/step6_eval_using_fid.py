#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import fid
from imageio  import imread
import tensorflow as tf

# Paths
image_path = 'generated/0' # set path to some generated images
stats_path = 'fid_stats/fid_stats_0.npz' # training set statistics
inception_path = fid.check_or_download_inception(None) # download inception network

# loads all images into memory (this might require a lot of RAM!)
image_list = glob.glob(os.path.join(image_path, '*.jpg'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

# load precalculated training set statistics
f = np.load(stats_path)
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
f.close()

fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)

fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
print("FID: %s" % fid_value)