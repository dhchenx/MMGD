# example of loading the generator model and generating images
from tensorflow.keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
import imageio
from keras.preprocessing import image
import glob
import imageio
import fid
import os
import cv2
import numpy as np
import tensorflow as tf
from imageio import imread
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


def get_FID(label_id):
    # Paths
    image_path = 'generated/'+str(label_id)  # set path to some generated images
    stats_path = 'generated/fid_stats_'+str(label_id)+'.npz'  # training set statistics
    inception_path = fid.check_or_download_inception(None)  # download inception network

    # loads all images into memory (this might require a lot of RAM!)
    image_list = glob.glob(os.path.join(image_path, '*.jpg'))
    images = np.array([imread(str(fn),as_gray=False,pilmode="RGB").astype(np.float32) for fn in image_list])

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
    return fid_value


labels=['hand','lung','cxt','heart']
flist=[]
for idx,lb in enumerate(labels):
    FID_value=get_FID(idx)
    flist.append(round(FID_value,2))
print('----result-----')
for f in flist:
    print(f)