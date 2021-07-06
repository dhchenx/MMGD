# example of loading the generator model and generating images
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.models import load_model
from matplotlib import pyplot
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# create and save a plot of generated images
def save_plot(examples, n_class,n_sample_num):
	# plot images
	for i in range(n_class * n_sample_num):
		# define subplot
		pyplot.subplot(n_class, n_sample_num, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0],cmap='jet')
	pyplot.show()

# load model
model = load_model('cgan_generator-64.h5')
# config parameters
N_DIM=100
N_CLASS=4
N_SAMPLE_PER_CLASS=10
# generate images
latent_points, labels = generate_latent_points(N_DIM, N_CLASS*N_SAMPLE_PER_CLASS,n_classes=N_CLASS)
# specify labels
labels = asarray([x for _ in range(N_SAMPLE_PER_CLASS) for x in range(N_CLASS)])
# generate images
X  = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
# X = (X + 1) / 2.0
# plot the result
save_plot(X, N_SAMPLE_PER_CLASS,N_CLASS)