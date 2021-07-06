import matplotlib.pyplot as plt
from matplotlib import pyplot
import imageio
import fid
from imageio import imread
import os
import numpy as np
import glob
import tensorflow as tf

# begin config
label_ids=["0","1","2","3"]
concepts=["lung","heart","cxr","hand"]

current_concept='lung'
iteration_no=2150

label_id=label_ids[concepts.index(current_concept)]
# end config

def get_FID(label_id):
    # Paths
    image_path = 'generated/'+label_id  # set path to some generated images
    stats_path = 'fid_stats/fid_stats_'+label_id+'.npz'  # training set statistics
    inception_path = fid.check_or_download_inception(None)  # download inception network

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
    return fid_value

# analyze training logs for models
folder='outputs/image64_'+current_concept
n=10
for i in range(50,5050,50):
    print(i)
    sample_path=folder+"/generated_sample"+str(i)+".png"
    pyplot.subplot(n, n, 1 + (i/50-1))
    # turn off axis
    pyplot.axis('off')
    # plot raw pixel data
    image=imageio.imread(sample_path)
    pyplot.imshow(image, cmap='jet')
pyplot.show()

print("evaluating...")

# example of loading the generator model and generating images
from tensorflow.keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
import numpy as np
import cv2
from keras.preprocessing import image
from PIL import Image
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# create and save a plot of generated images (reversed grayscale)
def generate_images_from_current_model(examples, n,label_id):
	# plot images
	for i in range(n * n):
		img = image.array_to_img(examples[i] * 255., scale=False)
		img.save('generated/'+label_id+'/'+str(i+1)+'.jpg')

	pyplot.show()

list_FID=[]

label_id=label_ids[concepts.index(current_concept)]
f_out=open("generated/"+str(label_id)+"_fid.txt","w",encoding='utf-8')
for i in range(50,5050,50):
    print("processing iter. no. "+str(i))
    model = load_model(
        'models/model64_' + current_concept + '/GAN_model_64_' + current_concept + '_' + str(i) + '.h5')
    # model.summary()
    # generate images
    latent_points = generate_latent_points(100, 100)
    # noise = np.random.normal(size=(64, 100))
    # generate images
    X = model.predict(latent_points)
    # plot the result
    generate_images_from_current_model(X, 10, label_id)

    FID=get_FID(label_id)
    print("FID for "+str(i)+": "+str(FID))
    list_FID.append(FID)
    f_out.write(str(FID)+"\n")
    f_out.flush()

f_out.close()
print("the FID result is:\n")
for f in list_FID:
    print(f)








