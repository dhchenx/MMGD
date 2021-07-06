# example of loading the generator model and generating images
from tensorflow.keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
import numpy as np

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# create and save a plot of generated images (reversed grayscale)
def show_plot(examples, n):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i],cmap='jet')
	pyplot.show()

# load model
label_ids=["0","1","2","3"]
concepts=["lung","heart","cxr","hand"]

current_concept='heart'
iteration_no=1850

label_id=label_ids[concepts.index(current_concept)]
model = load_model('models/model64_'+current_concept+'/GAN_model_64_'+current_concept+'_'+str(iteration_no)+'.h5')
model.summary()
# generate images
latent_points = generate_latent_points(100, 100)
# noise = np.random.normal(size=(64, 100))
# generate images
X = model.predict(latent_points)
# plot the result
show_plot(X, 10)