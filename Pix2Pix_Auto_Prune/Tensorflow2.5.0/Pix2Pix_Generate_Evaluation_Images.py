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
import matplotlib
from matplotlib import pyplot
import multiprocessing
import wget
from pathlib import Path
import tarfile
import os
from datetime import datetime
from sys import platform
from PIL import Image

from kerassurgeon import Surgeon

import tensorflow_model_optimization as tfmot



retrain_pocs = 50 
prune_pocs = 1
prune_loops = 100
n_init_epochs = 100
n_batch = 10
n_batch_fit = 32
bulk_samples_to_test = 32
retrain_batches = 10

Flip = True
d_select = 1

#[_,Flip,d_select] = (sys.argv)

Flip = bool(int(Flip))
d_select = int(d_select)

dataset_website = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/"

def apply_pruning_w_params(layer):
        end_step = end_step = 1096*prune_pocs
        pruning_params = {
          'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
        														   final_sparsity=0.80,
        														   begin_step=0,
        														   end_step=end_step)
        }
        if isinstance(layer, tensorflow.keras.layers.Conv2D) and 'no_prune' not in layer.name:                                                 ###########TO STOP PRUNING OF THE LAST LAYER##############
            print(layer.name+" Was Identified for pruning")
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

def apply_pruning(layer):
        if isinstance(layer, tensorflow.keras.layers.Conv2D) and 'no_prune' not in layer.name:                                                 ###########TO STOP PRUNING OF THE LAST LAYER##############
            print(layer.name+" Was Identified for pruning")
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

# load all images from the directory into memory with appropriate preprocessing
def choose_load_images(Selector):
	if Selector:
		def load_images(path, size=(256,512)):
			src_list, tar_list = list(), list()
			# enumerate filenames in directory, assuming all are images
			print("Processing Images")
			pathlist = listdir(path)
			pathlist.sort(key=lambda fname: int(fname.split('.')[0]))
			for filename in tqdm(pathlist):
				# load and resize the image
				pixels = load_img(path + '/' + filename, target_size=size)
				# convert to numpy array
				pixels = img_to_array(pixels)
				pixels = pixels.astype(np.int16)
				# split into satellite and map
				sat_img, map_img = pixels[:, :256], pixels[:, 256:]
				src_list.append(sat_img)
				tar_list.append(map_img)
			return [asarray(src_list), asarray(tar_list)]
	else:
		def load_images(path, size=(256,512)):
			src_list, tar_list = list(), list()
			# enumerate filenames in directory, assuming all are images
			print("Processing Images")
			for filename in tqdm(listdir(path)):
				# load and resize the image
				pixels = load_img(path + '/' + filename, target_size=size)
				# convert to numpy array
				pixels = img_to_array(pixels)
				# split into satellite and map
				sat_img, map_img = pixels[:, :256], pixels[:, 256:]
				src_list.append(sat_img)
				tar_list.append(map_img)
			return [asarray(src_list), asarray(tar_list)]
	return load_images

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
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same',activation=LeakyReLU(alpha=0.2), kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	#g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block

def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2),activation='relu', padding='same', kernel_initializer=init)(layer_in)
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
	b = Conv2D(512, (4,4), strides=(2,2),activation='relu', padding='same', kernel_initializer=init)(e7)
	#b = Activation('relu')(b)
	
# decoder model

	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	out_image = Conv2DTranspose(3, (4,4), strides=(2,2),activation='tanh', padding='same', name='no_prune', kernel_initializer=init)(d7)
	#out_image = Activation('tanh')(g)
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
	b = Conv2D(block_sizes[7], (4,4), strides=(2,2),activation='relu' ,padding='same', kernel_initializer=init)(e7)
	#b = Activation('relu')(b)
	
# decoder model

	d1 = decoder_block(b, e7, block_sizes[8])
	d2 = decoder_block(d1, e6, block_sizes[9])
	d3 = decoder_block(d2, e5, block_sizes[10])
	d4 = decoder_block(d3, e4, block_sizes[11], dropout=False)
	d5 = decoder_block(d4, e3, block_sizes[12], dropout=False)
	d6 = decoder_block(d5, e2, block_sizes[13], dropout=False)
	d7 = decoder_block(d6, e1, block_sizes[14], dropout=False)
	# output
	out_image = Conv2DTranspose(3, (4,4), strides=(2,2),activation='tanh', padding='same', name='no_prune', kernel_initializer=init)(d7)
	#out_image = Activation('tanh')(g)
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

def Generate_Test_Images(generator_weights,testset_file_loc,image_output_location, conn):
	vars = 0
	#load generator
	First = True
	g_model = tensorflow.keras.models.load_model(generator_weights)
	
	#load testset
	testset = load_real_samples(testset_file_loc)
	
	if Flip:
		three = testset[0]
		four = testset[1]
		testset = [four,three]
	
	
	X, Y = testset	
	X = g_model.predict(X,verbose='1')
	X = (X + 1) / 2.0

	x=0
	print("making_outputs_for" + testset_file_loc)

	mypath = "F:\\Data4NN\\Pix2Pix\\leftImg8bit\\val\\frankfurt\\"

	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	for array in tqdm(X):
		matplotlib.image.imsave(image_output_location + "\\" + onlyfiles[x], array)
		x+=1

		if x >= len(onlyfiles) and First:
			mypath = "F:\\Data4NN\\Pix2Pix\\leftImg8bit\\val\\lindau\\"
			onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
			x=0
			First = False
		elif  x >= len(onlyfiles):
			mypath = "F:\\Data4NN\\Pix2Pix\\leftImg8bit\\val\\munster\\"
			onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
			x=0
	
	
	
	conn.send([vars])
	conn.close()
from os.path import isfile, join
def Generate_GTS(testset_file_loc,image_output_location,A):
	First = True
	#load testset
	testset = load_real_samples(testset_file_loc)	
	
	X, Y = testset	

	if A:
		X=X
	else:
		X=Y

	X = (X + 1) / 2.0

	x=0
	print("making_outputs_for" + testset_file_loc)

	mypath = "F:\\Data4NN\\Pix2Pix\\leftImg8bit\\val\\frankfurt\\"

	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	for array in tqdm(X):
		matplotlib.image.imsave(image_output_location + onlyfiles[x], array)
		x+=1

		if x >= len(onlyfiles) and First:
			mypath = "F:\\Data4NN\\Pix2Pix\\leftImg8bit\\val\\lindau\\"
			onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
			x=0
			First = False
		elif  x >= len(onlyfiles):
			mypath = "F:\\Data4NN\\Pix2Pix\\leftImg8bit\\val\\munster\\"
			onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
			x=0


if __name__ == "__main__":
	tensorflow.keras.backend.clear_session

	if platform == "linux" or platform == "linux2":
		print("Linux Detected")
		try:
			multiprocessing.set_start_method('spawn')
		except RuntimeError:
			pass
	elif platform == "darwin":
		sys.exit("OSx not supported")
	elif platform == "win32":
		print("Windows Detected")

	# make dataset directory 
	
	if not os.path.exists('datasets'):
		os.makedirs('datasets')

	if not os.path.exists('experiments'):
		os.makedirs('experiments')

	# select dataset
	# load_images = choose_load_images(False)
	load_images = choose_load_images(True)
	if d_select == 1:
		dataset_prefix = "cityscapes.tar.gz"
		dataset_folder = "cityscapes"
	elif d_select == 2:
		dataset_prefix = "edges2handbags.tar.gz"
		dataset_folder = "edges2handbags"
	elif d_select == 3:
		dataset_prefix = "edges2shoes.tar.gz"
		dataset_folder = "edges2shoes"
	elif d_select == 4:
		dataset_prefix = "facades.tar.gz"
		dataset_folder = "facades"
	elif d_select == 5:
		dataset_prefix = "maps.tar.gz"
		dataset_folder = "maps"
	elif d_select == 6:
		dataset_prefix = "night2day.tar.gz"
		dataset_folder = "night2day"

	if not os.path.exists("datasets\\"+dataset_folder):
		os.makedirs("datasets\\"+dataset_folder)
		os.makedirs("experiments\\"+dataset_folder)

	print("Selected " + dataset_folder)

	### download dataset
	if not os.path.exists("datasets\\"+dataset_folder+"\\"+dataset_prefix):
		print("downloading")
		download = dataset_website + dataset_prefix
		data_folder = os.getcwd()+"\\datasets\\"+dataset_folder+"\\"
		wget.download(download,out = data_folder)

	to_check = "datasets\\"+dataset_folder+"\\"+dataset_folder+"\\"+"train"
	if not os.path.exists(to_check):
		### extract files
		print("Extracting")
		Tar_file = os.getcwd()+"\\datasets\\"+dataset_folder+"\\"+dataset_prefix
		tar = tarfile.open(Tar_file, 'r')
		for item in tar:
			tar.extract(item, os.getcwd()+"\\datasets\\"+dataset_folder+"\\")
			if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
				extract(item.name, ".\\" + item.name[:item.name.rfind("\\")])

		tar.close()

	if not os.path.exists(os.getcwd()+"\\datasets\\"+dataset_folder+"\\"+dataset_folder+"\\"+dataset_folder+'train.npz'):
		#
		## save trainset
		#
		save_loc = os.getcwd()+"\\datasets\\"+dataset_folder+"\\"+dataset_folder+"\\train"
		[src_images, tar_images] = load_images(save_loc)
		print('Loaded: ', src_images.shape, tar_images.shape)
		#
		## save as compressed numpy array
		#
		filename = dataset_folder+'train.npz'
		print('Compressing and saving data (can take some time... sorry no progress bar)')
		savez_compressed(os.getcwd()+"\\datasets\\"+dataset_folder+"\\"+dataset_folder+"\\"+filename, src_images, tar_images)
		print('Saved dataset: ', filename)
		del src_images, tar_images 
	if not os.path.exists(os.getcwd()+"\\datasets\\"+dataset_folder+"\\"+dataset_folder+"\\"+dataset_folder+'val.npz'):
		#
		## save valset
		#
		save_loc = os.getcwd()+"\\datasets\\"+dataset_folder+"\\"+dataset_folder+"\\val"
		[src_images, tar_images] = load_images(save_loc)
		print('Loaded: ', src_images.shape, tar_images.shape)
		#
		## save as compressed numpy array
		#
		filename = dataset_folder+'val.npz'
		print('Compressing and saving data (can take some time... sorry no progress bar)')
		savez_compressed(os.getcwd()+"\\datasets\\"+dataset_folder+"\\"+dataset_folder+"\\"+filename, src_images, tar_images)
		print('Saved dataset: ', filename)
		del src_images, tar_images
	# load image data
	print("Loading " + dataset_folder)
	dataset_file_loc = os.getcwd()+"\\datasets\\"+dataset_folder+"\\"+dataset_folder+"\\"+dataset_folder+"train.npz"
	testset_file_loc = os.getcwd()+"\\datasets\\"+dataset_folder+"\\"+dataset_folder+"\\"+dataset_folder+"val.npz"
	#dataset = load_real_samples(dataset_file_loc)
	#dataset = tensorflow.convert_to_tensor(dataset, dtype=tensorflow.float32)
	testset = load_real_samples(testset_file_loc)

	image_shape = testset[0].shape[1:]

	####
	experiment_to_test = "F:\\Data4NN\\Pix2Pix\\202208311633Flip\\"
	####

	def Locate_Models(directory_in_str,lookfor):
		directory = os.fsencode(directory_in_str)
		files = []
		for file in os.listdir(directory):
			filename = os.fsdecode(file)
			if lookfor in filename and "h5fin" not in filename and "post_d" not in filename: 
				print(os.path.join(directory, file))
				files.append(filename)
		return files

	

	image_save_location = experiment_to_test + "\\Ground_Truths"

	if not os.path.exists(image_save_location):
		os.makedirs(image_save_location)
	temp = True

	for generator_weights_location in ["A","B"]:

		gen_in_test = generator_weights_location

		generator_weights_location = experiment_to_test + generator_weights_location

		spec_save_location = image_save_location + "\\" + gen_in_test + "\\"

		spec_save_location = spec_save_location

		if not os.path.exists(spec_save_location):
			os.makedirs(spec_save_location)

		Generate_GTS(testset_file_loc,spec_save_location,temp)

	Gan_Init_Gen = Locate_Models(experiment_to_test,"gmodel")
	
	####### Generate Gan Training Images #########

	image_save_location = experiment_to_test + "\\Inital_train"

	if not os.path.exists(image_save_location):
		os.makedirs(image_save_location)

	for generator_weights_location in Gan_Init_Gen:

		gen_in_test = generator_weights_location

		generator_weights_location = experiment_to_test + generator_weights_location

		spec_save_location = image_save_location + "\\" + gen_in_test + "\\"

		spec_save_location = spec_save_location[:-4]

		if not os.path.exists(spec_save_location):
			os.makedirs(spec_save_location)

		parent_conn, child_conn = multiprocessing.Pipe()
	
		reader_process  = multiprocessing.Process(target=Generate_Test_Images, args=(generator_weights_location,testset_file_loc,spec_save_location, child_conn))
	
		reader_process.start()
	
		remove_points_returned = parent_conn.recv()
	
		[vars] = remove_points_returned
	
		reader_process.join()

	image_save_location = experiment_to_test + "\\Inital_models"

	if not os.path.exists(image_save_location):
		os.makedirs(image_save_location)

	for generator_weights_location in ["PrunedModelG.h5","UnPrunedModelG.h5"]:

		gen_in_test = generator_weights_location

		generator_weights_location = experiment_to_test + generator_weights_location

		spec_save_location = image_save_location + "\\" + gen_in_test + "\\"

		spec_save_location = spec_save_location[:-4]

		if not os.path.exists(spec_save_location):
			os.makedirs(spec_save_location)

		parent_conn, child_conn = multiprocessing.Pipe()
	
		reader_process  = multiprocessing.Process(target=Generate_Test_Images, args=(generator_weights_location,testset_file_loc,spec_save_location, child_conn))
	
		reader_process.start()
	
		remove_points_returned = parent_conn.recv()
	
		[vars] = remove_points_returned
	
		reader_process.join()

	Post_Pruned = Locate_Models(experiment_to_test,"post")

	image_save_location = experiment_to_test + "\\Post_pruned_models"

	if not os.path.exists(image_save_location):
		os.makedirs(image_save_location)

	for generator_weights_location in Post_Pruned:

		gen_in_test = generator_weights_location

		generator_weights_location = experiment_to_test + generator_weights_location

		spec_save_location = image_save_location + "\\" + gen_in_test + "\\"

		spec_save_location = spec_save_location[:-4]

		if not os.path.exists(spec_save_location):
			os.makedirs(spec_save_location)

		parent_conn, child_conn = multiprocessing.Pipe()
	
		reader_process  = multiprocessing.Process(target=Generate_Test_Images, args=(generator_weights_location,testset_file_loc,spec_save_location, child_conn))
	
		reader_process.start()
	
		remove_points_returned = parent_conn.recv()
	
		[vars] = remove_points_returned
	
		reader_process.join()

	Post_Pruned = Locate_Models(experiment_to_test,"pre")

	image_save_location = experiment_to_test + "\\Pre_pruned_models"

	if not os.path.exists(image_save_location):
		os.makedirs(image_save_location)

	for generator_weights_location in Post_Pruned:

		gen_in_test = generator_weights_location

		generator_weights_location = experiment_to_test + generator_weights_location

		spec_save_location = image_save_location + "\\" + gen_in_test + "\\"

		spec_save_location = spec_save_location[:-4]

		if not os.path.exists(spec_save_location):
			os.makedirs(spec_save_location)

		parent_conn, child_conn = multiprocessing.Pipe()
	
		reader_process  = multiprocessing.Process(target=Generate_Test_Images, args=(generator_weights_location,testset_file_loc,spec_save_location, child_conn))
	
		reader_process.start()
	
		remove_points_returned = parent_conn.recv()
	
		[vars] = remove_points_returned
	
		reader_process.join()

