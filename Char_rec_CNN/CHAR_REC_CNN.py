import tensorflow as tf

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


def make_classifier_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (3, 3),input_shape=[28, 28, 1]))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))


    model.add(layers.Conv2D(64, (3, 3),input_shape=[28, 28, 1]))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))

   
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(10))
    model.add(layers.Softmax())
    
    return model


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
#@tf.function
def train_step(model, images):
    print("training step")
    for batch in range(images):
        
        with tf.GradientTape() as tape: # Forward pass
            y_true = batch[1]
            y_pred = classifier(batch, training=True)
            loss = classifier_loss(y_true=y_true, y_pred=y_pred)

      
    
    
def train(model, train_dataset, test_dataset, epochs):

      print("Entering train...")
      for epoch in range(epochs):
            print('Epoch {}'.format(epoch+1))
            start = time.time()
            print(start)
            epoch_loss_avg = tf.keras.metrics.Mean() # Keeping track of the training loss
            epoch_acc_avg  = tf.keras.metrics.Mean()  # Keeping track of the training accuracy
            
            i=1
            for image_batch in train_dataset:
                print("Doing batch:"+str(i))
                train_step(model, image_batch)
                i=i+1


            # Save the model every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

      



# Here starts the process.
#load the MNIST dataset 

TRAIN_MODEL = False 
LOAD_MODEL  = False

mnist_path = 'c:/python/CHAR_REC_CNN/dataset/mnist.npz'
(images, labels), (_, _) = tf.keras.datasets.mnist.load_data(mnist_path)
print(images.shape)
print(labels.shape)


train_images = images[0:10000]
train_labels = labels[0:10000]

print(train_images.shape)
print(train_labels.shape)


test_images = images[10000:10100]
test_labels = labels[10000:10100]

print(test_images.shape)
print(test_labels.shape)


print(train_images)
for n in range(100,110,1):
    plt.subplot(5, 2, n-99)
    plt.imshow(PIL.Image.fromarray(train_images[n]),cmap="gray")
plt.show()    


train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5  # Normalize the images to [-1, 1]


BUFFER_SIZE =  10000
BATCH_SIZE =   50


print("SHUFFLING AND BATCHING DATA...")
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset =  tf.data.Dataset.from_tensor_slices(test_images).shuffle(BUFFER_SIZE)



print ("Creating Clasifier Model...")
classifier= make_classifier_model()
print(classifier.summary())
#show  the model as a graph in a png file
tf.keras.utils.plot_model(classifier, to_file='CNN_CHAR_CLASSIFIER.png', show_shapes=True, dpi=64)



# This method returns a helper function to compute cross entropy loss
classifier_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
classifier_optimizer = tf.keras.optimizers.Adam(1e-4)


classifier.compile(optimizer=classifier_optimizer,
              loss=classifier_loss,
              metrics=['accuracy'])

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(classifier_optimizer=classifier_optimizer,
                                 classifier=classifier)
                    
if (LOAD_MODEL):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


print ("Testing an Image...")
img = test_images[15]
img = img.reshape(-1,28,28,1)

print(img.shape)
char_class = classifier.predict(img)
print(char_class)


test_loss, test_acc = classifier.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy before training:', test_acc)
print('\nTest accuracy before training:', test_loss)





checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(classifier_optimizer=classifier_optimizer,
                                 classifier=classifier)
                    

EPOCHS = 15


if (TRAIN_MODEL):
    print("Starting training...")

    history = classifier.fit(train_images, train_labels, batch_size = BATCH_SIZE,  validation_data = (test_images, test_labels),  epochs = EPOCHS)

    checkpoint.save(file_prefix = checkpoint_prefix)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    test_loss, test_acc = classifier.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy after training:', test_acc)
    print('\nTest loss after  training:', test_loss)

    print ("Testing an Image...")
    img = test_images[0]



    img = img.reshape(-1,28,28,1)

    print(img.shape)
    char_class = classifier.predict(img)
    print(char_class)



    print ("hello")