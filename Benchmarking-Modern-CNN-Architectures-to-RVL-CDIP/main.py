import os
import tensorflow as tf

from utils import CyclicLR, WarmUpCosineDecayScheduler, cosine_decay_with_warmup, step_decay_schedule, LRFinder, WarmUpLearningRateScheduler, SWA, OneCycleLR
import tensorflow as tf
from base_models import *
import argparse
from efficientnet.keras import *
#
import csv
import cv2
import keras
from keras.layers import *
from keras import backend as K
from keras.optimizers import *
from keras.callbacks import *
from keras.applications.inception_v3 import preprocess_input

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, image
from keras.models import Model

from tensorflow.keras.callbacks import *
from keras.utils import multi_gpu_model
from keras.utils import to_categorical
from keras.utils import plot_model
import math
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Process
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import random
import subprocess
from subprocess import check_output
from scipy import ndimage
from skimage.io import imread
import sklearn.metrics as metrics
import tarfile
import time
from time import sleep
from tqdm import tqdm

from keras.preprocessing.image import save_img

from keras import activations
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import pickle
import time, os, fnmatch, shutil
from random_eraser import get_random_eraser
from MixupImageDataGenerator import MixupImageDataGenerator

t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)

print("runtime initiated at "+str(timestamp))

#/data/home/djonathan/sharedfiles/all/test

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', action = 'store', dest = 'dataset', help = "dataset", default='rvlcdip')

parser.add_argument('--train_dir', action = 'store', dest = 'train_dir', help = "train dir", default='/data/home/djonathan/datasets/rvlcdip/train')
parser.add_argument('--val_dir', action = 'store', dest = 'val_dir', help = "val dir", default='/data/home/djonathan/datasets/rvlcdip/val')
parser.add_argument('--test_dir', action = 'store', dest = 'test_dir', help = "test dir", default = '/data/home/djonathan/datasets/rvlcdip/test')
parser.add_argument('--num_class', action = 'store', dest = 'num_class', help = "number of classes",default = 16)

parser.add_argument("--num_epochs", type=int, default=1 ,dest = 'num_epochs', help="num_epoch")
parser.add_argument('--batch_size', action = 'store', dest = 'batch_size', help = "batch size", default = 8)
parser.add_argument("--num_train_steps", type=int, default= 10 ,dest = 'num_train_steps', help="steps per epoch")
parser.add_argument("--num_val_steps", type=int, default=100 ,dest = 'num_val_steps', help="steps per eval")
parser.add_argument("--num_test_steps", type=int, default=1000 ,dest = 'num_test_steps', help="steps per eval")
parser.add_argument('--input_dim_width', action = 'store', dest = 'input_dim_width', help = "The input dimension width", default = 224)
parser.add_argument('--input_dim_length', action = 'store', dest = 'input_dim_length', help = "The input dimension length", default = 224)
parser.add_argument('--load_model_path', action = 'store', dest = 'load_model_path', help = "path to load model from")
parser.add_argument('--save_model_path', action = 'store', dest = 'save_model_path', help = "path to save model to", default = './saved_models/')
parser.add_argument('--gpu_string', action = 'store',  default='2,3', dest = 'gpu_string', help='comma separated list of GPU(s) to use.')
parser.add_argument('--gpu_count', action = 'store',  default=2, dest = 'gpu_count', help='comma separated list of GPU(s) to use.')
parser.add_argument('--tensorboard_logdir', action = 'store', dest = 'tensorboard', help='tensorboard log directory', default = "default")
parser.add_argument("--model_name", type=str, default="ResNet18",dest = 'model_name',	help="name of pre-trained network to use")
parser.add_argument("--weights", action = 'store', default=None, dest = 'weights', help="name of pre-trained network to use")
parser.add_argument("--name", default='debug', type=str, help="name of the log")

#optional dense layer representations
parser.add_argument("--num_dense_layers", type=int, default=1 ,dest = 'num_dense_layers', help="dense in MLP")
parser.add_argument("--num_dense_nodes", type=int, default=4096 ,dest = 'num_dense_nodes', help="number of nodes in MLP steps per eval")
parser.add_argument("--dropout_pct", type=float, default=0.50 ,dest = 'dropout_pct', help="percentage of dropout in each layer")

#image augmentation
parser.add_argument("--featurewise_center", type=bool, default=False ,dest = 'featurewise_center')
parser.add_argument("--samplewise_center", type=bool, default=False ,dest = 'samplewise_center')
parser.add_argument("--featurewise_std_normalization", type=bool, default=False ,dest = 'featurewise_std_normalization')
parser.add_argument("--samplewise_std_normalization", type=bool, default=False ,dest = 'samplewise_std_normalization')
parser.add_argument("--zca_whitening", type=bool, default=False ,dest = 'zca_whitening')
parser.add_argument("--rotation_range", type=float, default=0 ,dest = 'rotation_range')
parser.add_argument("--width_shift_range", type=float, default=0 ,dest = 'width_shift_range')
parser.add_argument("--height_shift_range", type=float, default=0 ,dest = 'height_shift_range')
parser.add_argument("--horizontal_flip", action="store", default=False ,dest = 'horizontal_flip')
parser.add_argument("--vertical_flip", action="store", default=False ,dest = 'vertical_flip')
parser.add_argument("--zoom_range", type=float, default=0 ,dest = 'zoom_range')
parser.add_argument("--shear_range", type=float, default=0 ,dest = 'shear_range')
parser.add_argument("--course_dropout_probability", type=float, default=0.5 ,dest = 'course_dropout_probability')
parser.add_argument("--cut_out", action="store", default=False ,dest = 'cut_out')
parser.add_argument("--mix_alpha_init", action="store", default=0.6 ,dest = 'mix_alpha_init',type=float)
parser.add_argument("--mix_up", action="store", default=False ,dest = 'mix_up')

#optimizer stuff
parser.add_argument("--optimizer_choice", type=str, default='sgd' ,dest = 'optimizer_choice')
parser.add_argument("--decay_choice", type=float, default=1e-6 ,dest = 'decay_choice')
parser.add_argument("--initial_lr", type=float, default=0.001 ,dest = 'initial_lr', help="the intial learning rate in the cycle")
parser.add_argument("--final_lr", type=float, default=0.01 ,dest = 'final_lr', help="the final learning rate in the cycle")
parser.add_argument("--lr_schedule", type=str, default="one_cycle",dest = 'lr_schedule',   help="The learning rate schedule used. Options are Warmup Cosine Decay, One Trinagular Cycle, Step and Exponential Decay")
parser.add_argument("--cycle_period", type=int, default=0,dest = 'cycle_period',   help="number of steps to get to top of wave cycle")
parser.add_argument("--period_shape", type=str, default="triangular",dest = 'period_shape',   help="period shape")

parser.add_argument("--use_SWA", action="store", default=False ,dest = 'use_SWA')


args = parser.parse_args()
print(args)


#data augmentation schedule stuff
featurewise_center = False
samplewise_center = False
featurewise_std_normalization = False
samplewise_std_normalization = False
zca_whitening = args.zca_whitening
rotation_range = 0
width_shift_range = 0
height_shift_range = 0
shear_range = 0
horizontal_flip = False
vertical_flip = False
zoom_range = 0
cut_out = False
mix_up = False
mix_alpha_init = 0.6


#learning rate schedule stuff
initial_lr = 0.001
final_lr = 0.01
lr_schedule = "step_decay"
cycle_period = 0
period_shape = "triangular"

use_SWA = False

#optimizer stuff
optimizer_choice = "sgd"
decay_choice = 1e-6

#training stuff
num_dense_layers= 1
num_dense_nodes = 4096
dropout_pct = 0.5
model_name="InceptionResNetV2"
save_model_path = './saved_models'
num_class = 16
batch_size = 32
input_dim_width =  224
input_dim_length =  224
num_epochs = 1
num_cpu = 1
num_gpu = 0
gpu_string='2,3'
num_train_steps = 10
num_val_steps = 10
num_test_steps = 10
total_steps = int(num_epochs * num_train_steps)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
test_scores = 'NA' 



#config GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_string)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)




# dataset is rvlcdip
train_dir = os.path.abspath('../rvlcdip-dataset/train')
val_dir = os.path.abspath('../rvlcdip-dataset/val')
test_dir = os.path.abspath('../rvlcdip-dataset/test')
num_class = 16

if cut_out == True and mix_up == False:

    train_datagen = ImageDataGenerator(rescale=1./255,
        featurewise_center= featurewise_center,  # set input mean to 0 over the dataset
        samplewise_center= samplewise_center,  # set each sample mean to 0
        featurewise_std_normalization=featurewise_std_normalization,  # divide inputs by std of the dataset
        samplewise_std_normalization=samplewise_std_normalization,  # divide each input by its std
        zca_whitening=zca_whitening,  # apply ZCA whitening
        rotation_range=rotation_range,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=width_shift_range,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=height_shift_range,
        shear_range= shear_range,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=horizontal_flip,  # randomly flip images
        vertical_flip=vertical_flip,
        zoom_range=zoom_range,
        preprocessing_function= get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
                v_l=0, v_h=255, pixel_level=True))
        #preprocessing_function=additional_augmentation)
    training_set = train_datagen.flow_from_directory(
            train_dir,
            target_size = (input_dim_length, input_dim_width),
            batch_size = batch_size,
            class_mode = 'categorical',
            shuffle = True) 

elif cut_out == True and mix_up == True:
    
    train_datagen = ImageDataGenerator(rescale=1./255,
        featurewise_center= featurewise_center,  # set input mean to 0 over the dataset
        samplewise_center= samplewise_center,  # set each sample mean to 0
        featurewise_std_normalization=featurewise_std_normalization,  # divide inputs by std of the dataset
        samplewise_std_normalization=samplewise_std_normalization,  # divide each input by its std
        zca_whitening=zca_whitening,  # apply ZCA whitening
        rotation_range=rotation_range,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=width_shift_range,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=height_shift_range,
        shear_range= shear_range,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=horizontal_flip,  # randomly flip images
        vertical_flip=vertical_flip,
        zoom_range=zoom_range,
        preprocessing_function= get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
                v_l=0, v_h=255, pixel_level=True))
    
    training_set = MixupImageDataGenerator(generator=train_datagen,
                                        directory=train_dir,
                                        batch_size=batch_size,
                                        img_height=input_dim_length,
                                        img_width=input_dim_width,
                                        alpha=mix_alpha_init)

elif cut_out == False and mix_up == False:
    train_datagen = ImageDataGenerator(rescale=1./255,
        featurewise_center= featurewise_center,  # set input mean to 0 over the dataset
        samplewise_center= samplewise_center,  # set each sample mean to 0
        featurewise_std_normalization=featurewise_std_normalization,  # divide inputs by std of the dataset
        samplewise_std_normalization=samplewise_std_normalization,  # divide each input by its std
        zca_whitening=zca_whitening,  # apply ZCA whitening
        rotation_range=rotation_range,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=width_shift_range,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=height_shift_range,
        shear_range= shear_range,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=horizontal_flip,  # randomly flip images
        vertical_flip=vertical_flip,
        zoom_range=zoom_range)

    training_set = train_datagen.flow_from_directory(
            train_dir,
            target_size = (input_dim_length, input_dim_width),
            batch_size = batch_size,
            class_mode = 'categorical',
            shuffle = True) 

elif cut_out == False and mix_up == True:
    print("Using MixUp Augmentation")
    
    train_datagen = ImageDataGenerator(rescale=1./255,
        featurewise_center= featurewise_center,  # set input mean to 0 over the dataset
        samplewise_center= samplewise_center,  # set each sample mean to 0
        featurewise_std_normalization=featurewise_std_normalization,  # divide inputs by std of the dataset
        samplewise_std_normalization=samplewise_std_normalization,  # divide each input by its std
        zca_whitening=zca_whitening,  # apply ZCA whitening
        rotation_range=rotation_range,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=width_shift_range,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=height_shift_range,
        shear_range= shear_range,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=horizontal_flip,  # randomly flip images
        vertical_flip=vertical_flip,
        zoom_range=zoom_range)

    training_set = MixupImageDataGenerator(generator=train_datagen,
                                        directory=train_dir,
                                        batch_size=batch_size,
                                        img_height=input_dim_length,
                                        img_width=input_dim_width,
                                        alpha=mix_alpha_init)


val_datagen = ImageDataGenerator(rescale=1./255)
val_set = val_datagen.flow_from_directory(
    val_dir,
    target_size=(input_dim_length, input_dim_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = True)


#Don't shuffle test set generator so you can index the batches right with predict_gen
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        test_dir,
        target_size =  (input_dim_length, input_dim_width),
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = False)
print("[INFO] train data directory set to", str(train_dir))



model = get_architecture(model_name=model_name, input_dim_length=input_dim_length, input_dim_width=input_dim_width,num_dense_layers=num_dense_layers,num_dense_nodes=num_dense_nodes,num_class=num_class,dropout_pct=dropout_pct, weights="imagenet")

#model.summary()
print("[INFO]",str(model_name), "loaded as initial model.")

#optimizer

if optimizer_choice == 'adam':
    #optim = adam
    optim = tf.keras.optimizers.Adam(lr=initial_lr, decay = decay_choice)
elif optimizer_choice == 'rmsprop':
    rmsprop = tf.keras.optimizers.RMSprop(lr=initial_lr, decay = decay_choice)
    optim = rmsprop
elif optimizer_choice == 'adadelta':
    adadelta = tf.keras.optimizers.Adadelta(lr=initial_lr, decay = decay_choice)
    optim = adadelta
elif optimizer_choice == 'sgd_nesterov':
    sgd_nesterov = tf.keras.optimizers.SGD(lr=initial_lr, nesterov = True, decay = decay_choice)
    optim = sgd_nesterov
elif optimizer_choice == 'amsgrad':
    amsgrad = tf.keras.optimizers.Adam(lr=initial_lr, amsgrad=True, decay = decay_choice)
    optim = amsgrad
elif optimizer_choice == 'sgd':
    sgd = tf.keras.optimizers.SGD(lr=initial_lr)
    optim = sgd
elif optimizer_choice == 'sgd_momentum':
    sgd_momentum = tf.keras.optimizers.SGD(lr=initial_lr, momentum=0.9, decay = decay_choice)    
    optim = sgd_momentum
elif optimizer_choice == 'nadam':
    nadam = tf.keras.optimizers.Nadam(lr=initial_lr)
    optim = nadam
elif optimizer_choice == 'adamax':
    adamax = tf.keras.optimizers.Adamax(lr=initial_lr)
    optim = adamax

print("[INFO]", optimizer_choice," selected")



#toggle multi-gpu
if num_gpu > 1:
    model = multi_gpu_model(model, num_gpu, cpu_merge=False) 

model.compile(optimizer=optim, loss = 'categorical_crossentropy', metrics = ['accuracy','categorical_crossentropy'])

#################

def lr_schedule_discrete(epoch):
    lr = initial_lr
    if epoch == 5:
        lr *= 1e-1
    elif epoch == 10:
        lr *= 1e-1
    elif epoch == 20:
        lr *= 1e-1
    elif epoch == 40:
        lr == initial_lr*1e-1
    elif epoch == 60:
        lr == initial_lr*1e-1
    print('Learning rate: ', lr)
    return lr


print("[INFO]", str(lr_schedule), "selected")

if lr_schedule == "one_cycle":
    lr_manager = OneCycleLR(
    max_lr=final_lr,
    end_percentage=0.2,
    scale_percentage=0.1,
    maximum_momentum=0.95,
    verbose=True)
    
    
    #clr_sched = CyclicLR(base_lr=initial_lr, max_lr=final_lr, step_size=(num_train_steps*num_epochs//2), mode='triangular')
    callbacks = lr_manager
    print("[INFO] one_cycle learing rate selected")

elif lr_schedule == "warmup_cosine_decay":
    warmup_epoch = int(0.2*num_epochs)
    #warmup_epoch = 2
    warmup_steps = int(warmup_epoch * num_train_steps)
    warmup_batches = warmup_epoch * num_train_steps
    warmup_cosine_decay = WarmUpCosineDecayScheduler(learning_rate_base=final_lr,
                                        total_steps=total_steps,
                                        warmup_learning_rate=initial_lr,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0,
                                       verbose = 1)
    callbacks = [warmup_cosine_decay, checkpoint]
    print("[INFO] warmup_cosine_decay learing rate selected")

elif lr_schedule == 'step_decay':
    step_schedule = step_decay_schedule(initial_lr=initial_lr, decay_factor=0.97, step_size=num_train_steps)
    callbacks = step_schedule


elif lr_schedule == "cyclical":
    clr_sched = CyclicLR(base_lr=initial_lr, max_lr=final_lr, step_size=cycle_period, mode=period_shape)
    callbacks = clr_sched

elif lr_schedule == "none":
    print("[INFO] No LR schedule used.")


elif lr_schedule == "penalized":
    print("Penalized learning rate method used.")
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=3, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    callbacks = reduce_lr

elif lr_schedule == "custom_schedule":
    print("[INFO] Penalized scheduled learning rate method used.")

    lr_scheduler = LearningRateScheduler(lr_schedule_discrete)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

    callbacks = lr_scheduler




elif lr_schedule == "warmup":  
    #warmup_epoch = int(0.05*num_epochs)
    warmup_epoch = 2
    warmup_batches = warmup_epoch * num_train_steps
    callbacks = WarmUpLearningRateScheduler(warmup_batches, init_lr=initial_lr, verbose=1)

#find_schedule = LRFinder(min_lr=initial_lr,max_lr=final_lr,steps_per_epoch=num_train_steps, epochs=num_epochs)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=1, min_lr=0)
filename= "./saved_models/"+str(model_name)+str()+str(timestamp)+str(lr_schedule)+str(initial_lr)+"_"+str(final_lr)+".h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filename, monitor='val_acc', verbose=1, save_best_only=True)
#es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, baseline= 0.5, patience=int(num_epochs//4))
es = tf.keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', verbose=1, baseline= 0.5, patience=20)



if use_SWA == True:
    num_SWA_epochs = 20
    SWA_schedule = SWA('SWA_weights_v02.h5', int(num_epochs-num_SWA_epochs))
    callbacks=[callbacks, checkpoint,es, SWA_schedule]
else:
     callbacks=[callbacks, checkpoint,reduce_lr, es]
print("[INFO] Training the model...")
# https://stackoverflow.com/questions/52433384/handle-invalid-corrupted-image-files-in-imagedatagenerator-flow-from-directory-i/52433477#52433477
# in case we have random corrupt images in the dataset, we want to gracefully skip that batch and keep training. 
# downside entire batch is skipped.
def my_gen(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except GeneratorExit:
            raise
        except:
            print("[INFO] Corrupt image. Skipping this batch...")

history = model.fit_generator(
        my_gen(training_set),
        steps_per_epoch = num_train_steps,
        epochs=num_epochs,
        validation_data=val_set,
        validation_steps=num_val_steps,
        # max_queue_size=50,
        # workers=1,
        use_multiprocessing=False,
        callbacks = [checkpoint])


print("[INFO] Training complete")

#evaluate model
print("[INFO] Evaluating the model...")
test_eval = model.evaluate_generator(my_gen(test_set), steps=num_test_steps, callbacks=None, use_multiprocessing=False, verbose=1)
print(test_eval)

#################
#training plots
#################
'''
plt.rcParams['figure.figsize'] = [18, 6]
find_schedule.plot_loss()
print("creating learning rate plot--loss")
plt.clf()

plt.rcParams['figure.figsize'] = [18, 6]
find_schedule.plot_lr()
print("creating learning rate plot--lr")
plt.clf()
'''


# list all data in history
print(history.history.keys())

plt.rcParams['figure.figsize'] = [18, 6]
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title(str(model_name)+" Train and Test Accuracies")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim(0.0, 1)
plt.show()
plt.savefig('acc_plot.png')
print("[INFO] Accuracy plot is created")
plt.clf() 
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(str(model_name)+"Train and Test Cross Entropy Loss Values")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim(0.0, 1)
plt.show()
plt.savefig('loss_plot.png')
print("[INFO] Loss plot is created")
plt.clf() 


#################
#confusion matrix
#################

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion Matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Ground Truth',
           xlabel='Predicted')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=46, ha="right",
             rotation_mode="anchor")
    plt.rcParams['figure.figsize'] = [15, 15]

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    print("[INFO] Confusion Matrix saved")
    plt.clf() 
    plt.show()
    plt.savefig('confusion_matrix.png')
    return ax




# from keras.utils import plot_model # INFO pydot` failed to call GraphViz. Please install GraphViz (https://www.graphviz.org/) and ensure that its executables are in the $PATH.
# from IPython.display import SVG
# #from keras.utils import model_to_dot
# from keras.utils.vis_utils import model_to_dot

# #plot model architecture plots
# plot_model(model, show_shapes=False, show_layer_names=False, to_file='model_architecture_simple.png')
# print("[INFO] Simple architecture plot saved with name model_architecture_simple.png")

# plot_model(model, show_shapes=True, show_layer_names=True, to_file='model_architecture_detailed.png')
# print("[INFO] Detailed architecture plot saved with name model_architecture_detailed.png")



#plot learning rate vs accuracy

if lr_schedule == "cyclical":
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.xlabel('Training Iterations')
    plt.ylabel('Learning Rate')
    plt.title("Learning Rate Policy")
    plt.plot(clr_sched.history['iterations'], clr_sched.history['lr'])
    plt.savefig("LR_schedule.png")
    plt.clf()
print("train vs val acc by epoch uploaded to comet")

#plot confusion matrices
plt.rcParams['figure.figsize'] = [20, 10]
test_steps_per_epoch = np.math.ceil(test_set.samples / test_set.batch_size)
print("total test steps is: ", str(test_steps_per_epoch))
predictions = model.predict_generator(test_set, steps=test_steps_per_epoch, verbose = 1)
y_pred = np.argmax(predictions, axis=1) # Get most likely class
true_classes = test_set.classes
y_true = test_set.classes[test_set.index_array]
test_acc = sum(y_pred==true_classes)/int(test_set.samples)
class_names = np.array(list((test_set.class_indices.keys())))
plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=False,title='Confusion matrix, test accuracy: '+str(round(sum(y_pred==true_classes)/int(test_set.samples),4)
))
plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True,title='Normalized confusion matrix, test accuracy: '+str(round(sum(y_pred==true_classes)/int(test_set.samples),4)
))
plt.clf()



