#load the required libraries

from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed

# all the necessary imports required 

from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from matplotlib import pyplot
#
from tqdm import tqdm
import numpy as np
import tensorflow
import pathlib
import sys
import time
import gc
from matplotlib import pyplot
import multiprocessing
#
from kerassurgeon import Surgeon
import tensorflow_model_optimization as tfmot

retrain_pocs = 8 
prune_pocs = 8
prune_loops = 40
n_init_epochs = 50
bulk_samples_to_test = 32
Flip = False

def apply_pruning_w_params(layer):
        end_step = end_step = 1096*prune_pocs
        pruning_params = {
          'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
        														   final_sparsity=0.80,
        														   begin_step=0,
        														   end_step=end_step)
        }
        if isinstance(layer, tensorflow.keras.layers.Conv2D):                                                 ###########TO STOP PRUNING OF THE LAST LAYER##############
            print(layer.name+" Was Identified for pruning")
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

def apply_pruning(layer):
        if isinstance(layer, tensorflow.keras.layers.Conv2D):                                                 ###########TO STOP PRUNING OF THE LAST LAYER##############
            print(layer.name+" Was Identified for pruning")
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

# load all images from the directory into memory with appropriate preprocessing

def load_images(path, size=(256,512)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assuming all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + '\\' + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
		src_list.append(sat_img)
		tar_list.append(map_img)
	return [asarray(src_list), asarray(tar_list)]

# define the discriminator model

def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model



# define an encoder block

def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block

def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the standalone generator model

def define_generator(image_shape=(256,256,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	
# decoder model

	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model


def define_generator_edit(block_sizes,image_shape=(256,256,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, block_sizes[0], batchnorm=False)
	e2 = define_encoder_block(e1, block_sizes[1])
	e3 = define_encoder_block(e2, block_sizes[2])
	e4 = define_encoder_block(e3, block_sizes[3])
	e5 = define_encoder_block(e4, block_sizes[4])
	e6 = define_encoder_block(e5, block_sizes[5])
	e7 = define_encoder_block(e6, block_sizes[6])
	# bottleneck, no batch norm and relu
	b = Conv2D(block_sizes[7], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	
# decoder model

	d1 = decoder_block(b, e7, block_sizes[8])
	d2 = decoder_block(d1, e6, block_sizes[9])
	d3 = decoder_block(d2, e5, block_sizes[10])
	d4 = decoder_block(d3, e4, block_sizes[11], dropout=False)
	d5 = decoder_block(d4, e3, block_sizes[12], dropout=False)
	d6 = decoder_block(d5, e2, block_sizes[13], dropout=False)
	d7 = decoder_block(d6, e1, block_sizes[14], dropout=False)
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model





# define the combined generator and discriminator model, for updating the generator

def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

def define_pruned_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# setup pruning 
	#g_model = tensorflow.keras.models.clone_model(g_model,clone_function=apply_pruning,)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model



# load image array saved in previous step and prepare training images 

def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]



# select a batch of random samples, returns images and target

def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

def generate_real_samples_sequen(dataset, n_samples, patch_shape, last_n):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = range(n_samples)
	ix = [last_n+x for x in ix]
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y, n_samples + last_n


# generate a batch of images, returns images and targets

def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y



# generate samples and save as a plot and save the model

def summarize_performance(step, g_model, d_model, dataset, n_samples=3):
	#clone and remove pruning components
	#g_model = tensorflow.keras.models.clone_model(g_p_model)
	#g_model = tfmot.sparsity.keras.strip_pruning(g_p_model)
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'gmodel_%06d.h5' % (step+1)
	filename3 = 'dmodel_%06d.h5' % (step+1)
	g_model.save(filename2)
	d_model.save(filename3)
	print('>Saved: %s and %s and %s' % (filename1, filename2, filename3))

def plot_compare(step, g_model, g_model_2, dataset, n_samples=3):
	#clone and remove pruning components
	#g_model = tensorflow.keras.models.clone_model(g_p_model)
	#g_model = tfmot.sparsity.keras.strip_pruning(g_p_model)
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	X_fakeB_2, _ = generate_fake_samples(g_model_2, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	X_fakeB_2 = (X_fakeB_2 + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(4, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(4, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	for i in range(n_samples):
		pyplot.subplot(4, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB_2[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(4, n_samples, 1 + n_samples*3 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()

def plot_single(step, g_model, dataset, n_samples=3):
	#clone and remove pruning components
	#g_model = tensorflow.keras.models.clone_model(g_p_model)
	#g_model = tfmot.sparsity.keras.strip_pruning(g_p_model)
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()


# pruning stuff



# train pix2pix model
def train(d_model, g_model, gan_model, dataset, testset, n_epochs=50, n_batch=1):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in tqdm(range(n_steps)):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real,)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		#print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if (i+1) % (bat_per_epo * int(n_epochs/10)) == 0:
			summarize_performance(i, g_model, d_model, testset)

def train_wo_save(d_model, g_model, gan_model, dataset, testset, n_epochs=50, n_batch=1):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in tqdm(range(n_steps)):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real,)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		#print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance

# setup smaller prune.
def add_prune_to_gen(g_model,prune_pocs):
	end_step = 1096*prune_pocs
	pruning_params = {
	  'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
															   final_sparsity=0.80,
															   begin_step=0,
															   end_step=end_step)
	}
	g_model = tensorflow.keras.models.clone_model(g_model,clone_function=apply_pruning,)
	opt = Adam(lr=0.0002, beta_1=0.5)
	g_model.compile(loss=['mae'], optimizer=opt)
	return g_model

def find_nodes(weights,block_sizes,batches,n_samples,conn):
	###### Splitting Model #######
	gan = define_generator_edit(block_sizes)
	gan.set_weights(weights)
	submodels = []
	for layer in gan.layers:
	    if isinstance(layer, tensorflow.keras.layers.Conv2D):
	        if gan.layers[-1].name not in layer.name:
	            main_model = tensorflow.keras.models.clone_model(gan)
	            time.sleep(1)
	            while True:
	                if main_model.layers[-1].name == layer.name:
	                    submodels.append(Model(inputs=main_model.input, outputs=main_model.layers[-1].output))
	                    print("adding " + layer.name + " to probe")
	                    break
	                popped_layer = main_model._layers.pop()
	
	########### VIZ OP ###########
	
	#plt.figure()
	#plt.imshow(decoder((batch)[0]))
	#pred = gan.predict(batch[..., [0]])
	#colour_pred = np.concatenate([batch[..., [0]], pred], -1)
	#plt.figure()
	#plt.imshow(decoder((colour_pred)[0]))
	#plt.show()
	
	######## Gather multi batch predictions###########
	

	total_set_size,_,_,_ = batches[0].shape

	if total_set_size%n_samples == 0:
		batches_to_probe = total_set_size/n_samples
	else:
		to_slice_1 = batches[0]
		to_slice_2 = batches[1]
		slice_ammount = total_set_size%n_samples
		batches[0] = to_slice_1[:-slice_ammount,:,:,:]
		batches[1] = to_slice_2[:-slice_ammount,:,:,:]
		total_set_size,_,_,_ = batches[0].shape
		batches_to_probe = total_set_size/n_samples

	all_submodel_predictons = []
	last_n = 0
	for i in tqdm(range(int(batches_to_probe))):
		[X_realA, X_realB], _, last_n = generate_real_samples_sequen(batches, n_samples, 1, last_n)
		if i == 0:
			submodel_predictons = []
			for sub_model in submodels:
				submodel_predictons.append(abs(sub_model.predict(X_realA)))
		else:
			j = 0
			for sub_model in submodels:
				submodel_predictons[j]+=abs(sub_model.predict(X_realA))
				j+=1
		#batch = next(batches) 

		oki =1
	
	del submodels
	
	######### average multi batches ##############
	
	for i in range(len(submodel_predictons)):
	    submodel_predictons[i]=submodel_predictons[i]/batches_to_probe
	
	########## sum across batch ####### 
	
	everything_summed_averged = []
	j=0
	for submodel_batch in submodel_predictons:
	    for i in range(submodel_batch.shape[0]):
	        if i == 0:
	            everything_summed_averged.append(submodel_batch[i,:,:,:])
	        else:
	            everything_summed_averged[j] += submodel_batch[i,:,:,:]
	    everything_summed_averged[j] = everything_summed_averged[j]/submodel_batch.shape[0]
	    j+=1
	
	#####   Find remove points ###########
	
	total_blocks = 0
	for size in block_sizes:
		total_blocks+=size
	mulitplier = 0.1
	nothing_removed = True
	counter = 0
	while nothing_removed:
		channels_removed = 0
		
		remove_points = []
		for pred in everything_summed_averged:
			j = 0
			nodes_to_remove = []
			avg_for_pred = np.average(pred)
			avg_for_pred = avg_for_pred * mulitplier
			for chanloc in range(pred.shape[2]):
				spechan = pred[:,:,chanloc]
				max_chan_val = np.max(spechan)
				if  max_chan_val < avg_for_pred:
					nodes_to_remove.append(j)
				j+=1
			remove_points.append(nodes_to_remove)
		for i in remove_points:
			if i != []:
				nothing_removed = False
				channels_removed += len(i)
		if channels_removed < total_blocks*0.01:
			nothing_removed = True
		mulitplier += 0.01
		counter += 1
		if counter % 5 == 0:
			print(mulitplier)
	conn.send([remove_points])
	conn.close()


def op_on_model(block_sizes,g_model_weights,remove_points,conn):
	g_model = define_generator_edit(block_sizes)
	g_model.set_weights(g_model_weights)
	surgeon = Surgeon(g_model)
	i = 0
	
	for layer1 in g_model.layers:
		if isinstance(layer1, tensorflow.keras.layers.Conv2D):
			if g_model.layers[-1].name not in layer1.name:
				if remove_points[i] != []:
					j = 0
					for layer in g_model.layers:
						if layer1.name == layer.name: #and not("transpose" in layer.name):
							surgeon.add_job('delete_channels', g_model.layers[j],channels=remove_points[i])
							block_sizes[i] -= len(remove_points[i])
							break
						j+=1 
				i+=1
	
	new_model = surgeon.operate()
	pruned_weights = new_model.get_weights()

	conn.send([block_sizes,pruned_weights])
	conn.close()

def op_on_model_NMP(block_sizes,g_model_weights,remove_points):
	g_model = define_generator_edit(block_sizes)
	g_model.set_weights(g_model_weights)
	surgeon = Surgeon(g_model)
	i = 0
	
	for layer1 in g_model.layers:
		if isinstance(layer1, tensorflow.keras.layers.Conv2D):
			if g_model.layers[-1].name not in layer1.name:
				if remove_points[i] != []:
					j = 0
					for layer in g_model.layers:
						if layer1.name == layer.name: #and not("transpose" in layer.name):
							surgeon.add_job('delete_channels', g_model.layers[j],channels=remove_points[i])
							block_sizes[i] -= len(remove_points[i])
							break
						j+=1 
				i+=1
	
	new_model = surgeon.operate()
	pruned_weights = new_model.get_weights()

def retrain_n_test(block_sizes,generator_weights,generator_weights_old,descriminator_weights,image_shape,dataset, testset, retrain_pocs, prune_pocs, batch_size, prune_it, conn):

	###
	new_gan = define_generator_edit(block_sizes)
	new_gan.set_weights(generator_weights)

	d_model = define_discriminator(image_shape)
	###
	old_gen = define_generator()
	old_gen.set_weights(generator_weights_old)
	###

	gan_model = define_pruned_gan(new_gan, d_model, image_shape)

	train_wo_save(d_model, new_gan, gan_model, dataset, testset, retrain_pocs ,n_batch=1)

	g_model = new_gan

	g_model.save("pruned_pre"+str(prune_it)+".h5")

	plot_compare(prune_it+20000, g_model, old_gen, dataset)

	g_model = tensorflow.keras.models.clone_model(g_model,clone_function=apply_pruning_w_params,)

	opt = Adam(lr=0.0002, beta_1=0.5)
	g_model.compile(loss=['mae'], optimizer=opt)

	callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

	g_model.fit(dataset[0],dataset[1],callbacks=callbacks,epochs=prune_pocs,batch_size=batch_size)

	g_model = tfmot.sparsity.keras.strip_pruning(g_model)
	g_model.save("pruned_post"+str(prune_it)+".h5")
	
	textfile = open("pruned_block_sizes"+str(prune_it)+".txt", "w")
	for element in block_sizes:
		textfile.write(str(element) + "\n")
	textfile.close()
	
	plot_compare(prune_it, g_model, old_gen, dataset)

	gweights = g_model.get_weights()
	dweights = d_model.get_weights()
	conn.send([gweights,dweights])
	conn.close()


def retrain_n_test_no_mp(block_sizes,generator_weights,generator_weights_old,descriminator_weights,image_shape,dataset, testset, retrain_pocs, prune_pocs, batch_size, prune_it):

	###
	new_gan = define_generator_edit(block_sizes)
	new_gan.set_weights(generator_weights)

	d_model = define_discriminator(image_shape)
	###
	old_gen = define_generator()
	old_gen.set_weights(generator_weights_old)
	###

	gan_model = define_pruned_gan(new_gan, d_model, image_shape)

	train_wo_save(d_model, new_gan, gan_model, dataset, testset, retrain_pocs ,n_batch=1)

	g_model = new_gan

	g_model.save("pruned_pre"+str(prune_it)+".h5")

	plot_compare(prune_it+20000, g_model, old_gen, dataset)

	g_model = tensorflow.keras.models.clone_model(g_model,clone_function=apply_pruning_w_params,)

	opt = Adam(lr=0.0002, beta_1=0.5)
	g_model.compile(loss=['mae'], optimizer=opt)

	callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

	g_model.fit(dataset[0],dataset[1],callbacks=callbacks,epochs=prune_pocs,batch_size=batch_size)

	g_model = tfmot.sparsity.keras.strip_pruning(g_model)
	g_model.save("pruned_post"+str(prune_it)+".h5")
	
	textfile = open("pruned_block_sizes"+str(prune_it)+".txt", "w")
	for element in block_sizes:
		textfile.write(str(element) + "\n")
	textfile.close()
	
	plot_compare(prune_it, g_model, old_gen, dataset)

	gweights = g_model.get_weights()
	dweights = d_model.get_weights()


if __name__ == "__main__":


	# load image data
	dataset = load_real_samples('maps_256.npz')
	testset = load_real_samples('maps_test_set_256.npz')



	prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
	
	if Flip:
		# for generating from facades
		one = dataset[0]
		two = dataset[1]
		dataset = two,one

		# for generating from facades
		one = testset[0]
		two = testset[1]
		testset = two,one

	print('Loaded', dataset[0].shape, dataset[1].shape)

	# define input shape based on the loaded dataset
	image_shape = dataset[0].shape[1:]

	# define the models
	d_model = define_discriminator(image_shape)
	g_model = define_generator(image_shape)

	####### Inital Train #####

	gan_model = define_pruned_gan(g_model, d_model, image_shape)

	train_wo_save(d_model, g_model, gan_model, dataset, testset, n_epochs=n_init_epochs, n_batch=1)

	old_weights = g_model.get_weights()

	g_model.save("UnPrunedModel.h5")

	###### Initial Prune ######

	end_step = end_step = 1096*n_init_epochs
	pruning_params = {
	  'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
															   final_sparsity=0.80,
															   begin_step=0,
															   end_step=end_step)
	}

	callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

	opt = Adam(lr=0.0002, beta_1=0.5)
	g_model.compile(loss=['mae'], optimizer=opt)
	g_model.fit(dataset[0],dataset[1],callbacks=callbacks,epochs=n_init_epochs)
	g_model = tfmot.sparsity.keras.strip_pruning(g_model)
	g_model.save("PrunedModel.h5")
	
	###########################
	
	block_sizes = [64,128,256,512,512,512,512,512,512,512,512,512,256,128,64]

	remove_pointers= {'arr': [],'weigh': [],'blocks':[]}

	
	weights = g_model.get_weights()
	d_model_weights = d_model.get_weights()
	

	for prune_it in range(prune_loops):

				parent_conn, child_conn = multiprocessing.Pipe()

				reader_process  = multiprocessing.Process(target=find_nodes, args=(weights,block_sizes,testset,bulk_samples_to_test,child_conn))

				reader_process.start()
				
				remove_points_returned = parent_conn.recv()
				
				remove_points = remove_points_returned[0]

				reader_process.join()

				###############

				parent_conn, child_conn = multiprocessing.Pipe()

				reader_process = multiprocessing.Process(target=op_on_model, args=(block_sizes,weights,remove_points,child_conn))

				reader_process.start()
				
				remove_points_returned = parent_conn.recv()
				
				[block_sizes,pruned_weights] = remove_points_returned

				reader_process.join()

				###############

				parent_conn, child_conn = multiprocessing.Pipe()

				retrain_batches = 32

				reader_process = multiprocessing.Process(target=retrain_n_test, args=(block_sizes,pruned_weights,old_weights,
																		  d_model_weights,image_shape,dataset, testset, retrain_pocs, 
																		  prune_pocs, retrain_batches, prune_it, child_conn))

				reader_process.start()
				
				remove_points_returned = parent_conn.recv()
				
				[weights,d_model_weights] = remove_points_returned

				reader_process.join()


