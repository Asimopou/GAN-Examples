import tensorflow as tf
print(tf.__version__)

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

def tensor_to_image(tensor):
      tensor = tensor*255                            #converts tensor to image 
      tensor = np.array(tensor, dtype=np.uint8)      #table with integer(uint8)
      if np.ndim(tensor)>3:                          #converts the table to an image with rgb 
        assert tensor.shape[0] == 1
        tensor = tensor[0]
      return PIL.Image.fromarray(tensor)



def make_generator_model():
    
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization(axis = -1))
    model.add(layers.ReLU())

    model.add(layers.Reshape((4, 4, 1024)))
    assert model.output_shape == (None, 4, 4, 1024)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(512, (5, 5), strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 4, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=2, padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    return model



def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=2, padding='same',
                                     input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    assert model.output_shape == (None, 32, 32, 64)
    
    model.add(layers.Conv2D(128, (5, 5), strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    assert model.output_shape == (None, 16, 16, 128)

    model.add(layers.Conv2D(256, (5, 5), strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    assert model.output_shape == (None, 8, 8, 256)

    model.add(layers.Conv2D(512, (5, 5), strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    assert model.output_shape == (None, 4, 4, 512)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    #model.add(layers.Dense(1, activation='sigmoid'))

    return model


epsilon = 0.00001

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

"""
def discriminator_loss(real_output, fake_output):
     return -tf.reduce_mean(tf.math.log(real_output + epsilon) + tf.math.log(1. - fake_output + epsilon))
   
def generator_loss(fake_output):
    return -tf.reduce_mean(tf.math.log(fake_output+epsilon))
"""


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = np.random.uniform(low= -1.0, high = 1.0, size = (BATCH_SIZE, noise_dim))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          generated_images = generator(noise, training=True)

          real_output = discriminator(images, training=True)
          fake_output = discriminator(generated_images, training=True)

          gen_loss = generator_loss(fake_output)
          disc_loss = discriminator_loss(real_output, fake_output)
          
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):

      print("Entering train...")
      for epoch in range(epochs):
            start = time.time()
            print(start)
            i=1
            for image_batch in dataset:
                print("Doing batch:"+str(i))
                train_step(image_batch)
                i=i+1

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)
            
            # Save the model every 20 epochs
            if (epoch + 1) % 20 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

      # Generate after the final epoch
      display.clear_output(wait=True)
      generate_and_save_images(generator,
                               epochs,
                               seed)


def generate_and_save_images(model, epoch, test_input):
      # Notice `training` is set to False.
      # This is so all layers run in inference mode (batchnorm).
      predictions = model(test_input, training=False)

      fig = plt.figure(figsize=(4, 4))

      for i in range(predictions.shape[0]):
          plt.subplot(4, 4, i+1)
          plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
          plt.axis('off')

      plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
      #plt.show()

# Read and decode an image file to a uint8 tensor
def load(image_file):
      image = tf.io.read_file(image_file)
      image = tf.io.decode_jpeg(image)
      return image




# Here starts the process.
#load the train images from the specified path 


EPOCHS = 10
noise_dim = 100
num_examples_to_generate = 1

LOAD_CHECKPOINT = True
TRAIN_MODEL = True



if (TRAIN_MODEL):
    BUFFER_SIZE =  30000 # small values for testing
    BATCH_SIZE =   128

    file_path = 'C:\\python\\facedataset\\'
    
    train_images = []

    #make a list of all files in train dataset
    train_set =os.listdir(file_path)

    #load BUFFER_SIZE images to create the train_images 
    for filename in  train_set[:BUFFER_SIZE]:
        im = load(file_path+filename)
        train_images.append(im)

    #convert train_images to array suitable for input to the model
    train_images = np.asarray(train_images)

    #print some images from train_images
    for n in range(1,10,1):
        plt.subplot(3, 3, n)
        plt.imshow(PIL.Image.fromarray(train_images[n]))
    plt.show()    

    #convert trainimages values to float and normalize to [-1, 1]
    train_images = train_images.reshape(train_images.shape[0], 64, 64, 3).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    print("SHUFFLING AND BATCHING DATA...")
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)



#Create the generator nodel 
print ("Creating Generator Model...")
generator = make_generator_model()

#save model structure in a file for printnig 
tf.keras.utils.plot_model(generator, to_file='DCGAN_generator.png', show_shapes=True, dpi=64)

print ("Creating a Random Image before training...")
noise = np.random.uniform(low=-1.0, high= 1.0, size=(1, 100))
generated_image = generator(noise, training=False)
print(generated_image.shape)
img = generated_image[0]
plt.imshow(img)
plt.show()

print ("Creating Discriminator Model...")
discriminator = make_discriminator_model()
#show  the model as a graph in a png file
tf.keras.utils.plot_model(discriminator, to_file='DCGAN_discriminator.png', show_shapes=True, dpi=64)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

print ("defining Oprimizers....")
generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

if (LOAD_CHECKPOINT):
    #checkpoint.restore(checkpoint_dir + "/ckpt-2")
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

if (TRAIN_MODEL):
    print("Starting training...")
    train(train_dataset, EPOCHS)

"""
print ("Generate and test image after training...")
for i in range(0,20)
noise = np.random.uniform(low=-1.0, high= 1.0, size=(1, 100))
generated_image = generator(noise, training=False)
print(generated_image.shape)
img = generated_image[0] * 255.0
plt.imshow(img)
plt.show()

decision = discriminator(generated_image)
print(decision)
"""
print ("Generate and test images after training...")
for i in range(0,10):
    #noise = tf.random.normal([1, 100])
    noise = np.random.uniform(low=-1.0, high= 1.0, size=(1, 100))
    generated_image = generator(noise, training=False)
    plt.subplot(5, 2, i+1)
    img = generated_image[0]
    plt.imshow(img)
    decision = discriminator(generated_image)
    print(decision)
plt.show()









