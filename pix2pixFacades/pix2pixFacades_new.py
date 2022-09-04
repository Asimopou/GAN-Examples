import tensorflow as tf

import os
import pathlib
import time
import datetime
import graphviz
import pydot

from matplotlib import pyplot as plt
from IPython import display

import torch



#resizing images 
def resize_image(in_img, r_img, height, width):
   r_img = tf.image.resize(r_img, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
   in_img = tf.image.resize(in_img, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
   return in_img, r_img

#croping randomly images 
def random_crop(in_img, r_img):
  stacked_img = tf.stack([in_img, r_imge], axis=0)
  cropped_img = tf.image.random_crop(stacked_img, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
  return cropped_img[0], cropped_img[1]

# Normalizing the images to [-1, 1]
def normalize_image(in_img, r_img):
  #divide pixe lvalue by 12.7 and subtract 1
  r_img = (r_img / 127.5) - 1  
  in_img = (in_img / 127.5) - 1
  return in_img, r_img

#tf.function is used to compile the function
@tf.function()
def jitter_image(in_img, r_img):
  # Resize image to  286x286
  input_image, r_img = resize_image(in_img, r_img, 286, 286)

  # Randomly  crop image to size  256x256
  in_img, r_img = random_crop(in_img, r_img)

  #randomly mirror image (50%)
  if tf.random.uniform(()) > 0.5:
    in_img = tf.image.flip_left_right(in_img)
    r_img = tf.image.flip_left_right(r_img)
  #return processed images
  return in_img, r_img


def load_image_for_train(a_file):
  in_img, r_img = load(a_file)
  in_img, r_img = jitter_image(in_img, r_img)
  in_img, r_img = normalize_image(in_img, r_img)
  return in_img, r_img

def load_image_for_test(image_file):
  in_img, r_img = load(image_file)
  in_img, r_img = resize_image(in_img, r_img,IMG_HEIGHT, IMG_WIDTH)
  in_img, r_img = normalize_image(in_img, r_img)
  return in_img, r_img

#define a model to downsample an input
def down_sample(filters, size, batchnorm=True):
  
  #define initial mean value and deviation of kernel weights
  init = tf.random_normal_initializer(0., 0.02)

  out = tf.keras.Sequential()
  out.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=init, use_bias=False))
  
  if batchnorm:
      out.add(tf.keras.layers.BatchNormalization())

  out.add(tf.keras.layers.LeakyReLU())
  return out

#Define a model to upsample an input
def up_sample(filters, size, dropout=False):
  #define initial mean value and deviation of kernel weights
  init = tf.random_normal_initializer(0., 0.02)

  out = tf.keras.Sequential()
  out.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
               padding='same', kernel_initializer=init, use_bias=False))

  out.add(tf.keras.layers.BatchNormalization())

  if dropout:
      out.add(tf.keras.layers.Dropout(0.5))

  out.add(tf.keras.layers.ReLU())
  return out


#Define the generator Model 
def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_st = [
    down_sample(64, 4, batchnorm=False),  # (batch_size, 128, 128, 64)
    down_sample(128, 4),  # specify (batch_size, 64, 64, 128)
    down_sample(256, 4),  # specify (batch_size, 32, 32, 256)
    down_sample(512, 4),  # specify (batch_size, 16, 16, 512)
    down_sample(512, 4),  # specify (batch_size, 8, 8, 512)
    down_sample(512, 4),  # specify (batch_size, 4, 4, 512)
    down_sample(512, 4),  # specify (batch_size, 2, 2, 512)
    down_sample(512, 4),  # specify (batch_size, 1, 1, 512)
  ]

  up_st = [
    up_sample(512, 4, use_dropout=True),         # specify (batch_size, 2, 2, 1024)
    up_sample(512, 4, use_dropout=True),         # specify(batch_size, 4, 4, 1024)
    up_sample(512, 4, apply_dropout=True),       # specify(batch_size, 8, 8, 1024)
    up_sample(512, 4),                           # specify(batch_size, 16, 16, 1024)
    up_sample(256, 4),                           # specify(batch_size, 32, 32, 512)
    up_sample(128, 4),                           # specify(batch_size, 64, 64, 256)
    up_sample(64, 4),                            # specify(batch_size, 128, 128, 128)
  ]

  #specify mean and deviation of initial kernel weights
  init = tf.random_normal_initializer(0., 0.03)

  last_layer = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=init,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x_in = inputs

  # Downsample through the model and specif
  skips = []
  for ds in down_st:
    x_in = ds(x_in)
    skips.append(x_in)

  #reverse the order of skips
  skips = reversed(skips[:-1])

  # Upsample and establish the forward  connections
  for up, skip in zip(up_st, skips):
    x_in = up(x_in)
    x_in = tf.keras.layers.Concatenate()([x_in, skip])

  x_in = last_layer(x_in)

  return tf.keras.Model(inputs=inputs,  outputs=x_in)

#Define the discriminator model
def Discriminator():
  init = tf.random_normal_initializer(0., 0.03)

  inpt = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  trgt = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x_d = tf.keras.layers.concatenate([inpt, trgt])  # (batch_size, 256, 256, channels*2)

  d1 = down_sample(64, 4, False)(x_d)  # (batchsize, 128, 128, 64)
  d2 = down_sample(128, 4)(down1)  # (batchsize, 64, 64, 128)
  d3 = down_sample(256, 4)(down2)  # (batchsize, 32, 32, 256)

  #pad data with 0 to create 34x34 object
  zero_p1 = tf.keras.layers.ZeroPadding2D()(d3)  # (batch_size, 34, 34, 256)
  conv1 = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=init,
                                use_bias=False)(zero_p1)  # (batch_size, 31, 31, 512)

  batch_norm1 = tf.keras.layers.BatchNormalization()(conv1)

  #specify leaky relu activation fucntion
  lk_relu = tf.keras.layers.LeakyReLU()(batch_norm1)

  zero_p2 = tf.keras.layers.ZeroPadding2D()(lk_relu)  # (batch_size, 33, 33, 512)

  last_layer_out = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=init)(zero_p2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inpt, trgt], outputs=last_layer_out)



#define the generator loss function
def gen_loss(discr_gen_output, gen_out, trgt):
  g_loss = loss_function(tf.ones_like(discr_gen_output), discr_gen_output)
  # Use mean absolute error 
  l_loss = tf.reduce_mean(tf.abs(trgt - gen_out))
  overall_gen_loss = g_loss + (LAMBDA * l_loss)
  return overall_gen_loss, g_loss, l_loss


#define discriminator loss 
def discr_loss(discr_real_output, discr_gen_output):
  r_loss = loss_function(tf.ones_like(discr_real_output), discr_real_output)
  g_loss = loss_function(tf.zeros_like(discr_gen_output), discr_gen_output)
  
  discr_loss = r_loss + g_loss

  return discr_loss

#function to generate and plot images
def generate_images(model, test_inpt, trgt):

  predict = model(test_inpt, training=True)
  plt.figure(figsize=(15, 15))

  displ_list = [test_inpt[0], trgt[0], predict[0]]
  title = ['Input Image', 'Real Image', 'Predicted Image']

  for cnt in range(3):
    plt.subplot(1, 3, cnt+1)
    plt.title(title[cnt])
    # Change the pixel values from [-1,1] in the [0, 1] range to plot.
    plt.imshow(displ_list[cnt] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


@tf.function
def train_step(in_img, trgt, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    #use in_img to generate an output from generator
    gen_out = generator(in_img, training=True)
    #feed the disciminator with input image and target image
    discr_real_output = discriminator([in_img, trgt], training=True)
    #feed the discriminator with input image and generated image
    discr_gen_output = discriminator([in_img, gen_out], training=True)
    
    #compute the discriminator loss 
    d_loss = discr_loss(discr_real_output, discr_gen_output)

    #compute the generator loss
    g_total_loss, g_gan_loss, g_l_loss = gen_loss(discr_gen_output, gen_out, trgt)
  
  #compute the gradients  for generator and discriminator
  gen_gradients = gen_tape.gradient(g_total_loss, generator.trainable_variables)
  disc_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', g_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', g_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', g_l_loss, step=step//1000)
    tf.summary.scalar('discr_loss', d_loss, step=step//1000)

#define the training process
def train(train_ds, test_ds, n_steps):
  exmpl_input, exmpl_target = next(iter(test_ds.take(1)))
  
  for stp, (in_img, trgt) in train_ds.repeat().take(n_steps).enumerate():
    if (stp) % 1000 == 0:
          display.clear_output(wait=True)
          #generate_images(generator, example_input, example_target)
          print(f"Step: {step//1000}K")

    # Training step
    train_step(in_img, trgt, stp)

    if (stp+1) % 100 == 0:
      print('+', end='', flush=True)

    # Save (checkpoint) the model every 1000 steps
    if (stp + 1) % 1000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

#define a function to load images
def load(image_file):
      # Read and decode an image file to a uint8 tensor
      image = tf.io.read_file(image_file)
      image = tf.io.decode_jpeg(image)

      # Split the image  into two tensors:
      # - one with a actual building facade image
      # - one with an sketch image 
      width = tf.shape(image)[1]
      width = width // 2
      in_img = image[:, width:, :]
      r_img =  image[:, :width, :]

      print(r_img.shape)
      print(in_img.shape)


      # Convert both images to float32 tensors
      in_img = tf.cast(in_img, tf.float32)
      r_img = tf.cast(r_img, tf.float32)

      print(r_img.shape)
      print(in_img.shape)

      return in_img, r_img






#HERE STARTS THE MAIN PROGRAMME

PATH = "C:\\python\\pix2pixFacades\\facades\\"

from PIL import Image

# The  training set consists of 400 images
BUFFER_SIZE = 400
# Use batch size of 1
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

TRAIN_OPTION = False
RELOAD_SAVED_MODEL = True

#show the first image in directory
inpt, real_in = load(PATH + 'train\\1.jpg')

#convert values to [0,1] to display
plt.figure()
plt.imshow(inp / 255.0)
plt.figure()
plt.imshow(re / 255.0)
plt.show()

plt.figure(figsize=(6, 6))
for i in range(4):
  j_inpt, j_real_in = jitter_image(inpt, real_in)
  plt.subplot(2, 2, i + 1)
  plt.imshow(rj_inpt / 255.0)
  plt.axis('off')
plt.show()


train_data = tf.data.Dataset.list_files(PATH + 'train\\*.jpg')
train_data = train_dataset.map(load_image_for_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)

train_data = train_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#train_dataset = train_dataset.batch(BATCH_SIZE)

try:
  test_data = tf.data.Dataset.list_files(PATH + 'test\\*.jpg')
except tf.errors.InvalidArgumentError:
  test_data = tf.data.Dataset.list_files(PATH + 'val\\*.jpg')
test_data = test_data.map(load_image_for_test)
test_data = test_data.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3

downmodel = down_sample(3, 4)
downresult = downmodel(tf.expand_dims(inp, 0))
print (downresult.shape)

upmodel = up_sample(3, 4)
upresult = upmodel(downresult)
print (upresult.shape)

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

generator_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])
plt.show()

LAMBDA = 100
loss_fucntion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

discriminator = Discriminator()
#plot the discriminator model
tf.keras.utils.plot_model(discriminator, to_file='disc.png', show_shapes=True, dpi=64)


discrriminator_out = discriminator([inp[tf.newaxis, ...], genenator_output], training=False)
plt.imshow(discriminator_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()
plt.show()

#define the optimizers for generator and discriminator
gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discr_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

chkpnt_dir = './training_checkpoints'
chkpnt_prefix = os.path.join(chkpnt_dir, "ckpt")
chkpnt = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                 discriminator_optimizer=discr_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


for exmpl_inpt, exmpl_trgt in test_data.take(1):
  generate_images(generator, exmpl_input, exmpl_trgt)


log_dir="C:/python/pix2pixFacades/logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

if (RELOAD_SAVED_MODEL):
    print("Reloading latest checkpoint")

    #chkpnt.restore(chkpnt_dir + "/ckpt-4")
    
    chkpnt.restore(tf.train.latest_checkpoint(chkpnt_dir))
    



if (TRAIN_OPTION):
    print("starting training...")
    train(train_data, test_data, steps=10000)



my_data = tf.data.Dataset.list_files(PATH + 'MyFacades\\proc*.jpg')
my_data = my_data.map(load_image_for_test)
my_data = my_data.batch(BATCH_SIZE)

#run the trained model on some examples from a custom set

print("Generating image from custom dataset..")
for inpt, trgt  in my_data.take(4): 
    generate_images(generator, inpt, trgt)
    
# Run the trained model on a few examples from the test set
for inpt, trgt in test_data.take(2):
    generate_images(generator, inpt, trgt)
