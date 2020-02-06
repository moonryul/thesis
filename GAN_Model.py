# GAN USING XENAKIS SCREEN'S METHOD AS THE GENERATIVE MODEL AND JAZZ MUSIC AS THE DISCRIMINATOR INPUT
# Model code from Tensorflow's DCGAN tutorial

#Importing libraries
import tensorflow as tf
import numpy as np
import glob
import soundfile as sf


# Reading audio files --> correct this so all files are read and put in different arrays 
def dataset():
#  soundfile puts the data in numpy arrays automatically (2 dim because of 2 channels as it's stereo not mono)
    for filepath in glob.iglob(r'C:\Users\jacin\Documents\dataset\*.flac'):
        data, samplerate = sf.read(filepath)
    return data

training_data = dataset()
training_data.shape

# Loading the numpy arrays into the dataset
BATCH_SIZE = 32
BUFFER_SIZE = 4096
train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Generator Model
def make_generator_model():
    model = tf.keras.Sequential()  
    model.add(tf.keras.layers.Dense(16384, input_shape=(500,))) 
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01)) 
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9)) 
    model.add(tf.keras.layers.Reshape((16, 8)))

    model.add(tf.keras.layers.Conv1D(16, 8, padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Dropout(rate=0.1))

    model.add(tf.keras.layers.Conv1D(32, 16, padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Dropout(rate=0.1))

    model.add(tf.keras.layers.Conv1D(64, 32, padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.summary()
    return model

generator = make_generator_model()

noise = tf.random.normal([1, 500])
generated_sound = generator(noise, training=False)

# Discriminator Model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16384, input_shape=(128, 128))) 
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01)) 
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9)) 
    model.add(tf.keras.layers.Reshape((128, 128)))

    model.add(tf.keras.layers.Conv1D(256, 256, padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Dropout(rate=0.1))

    model.add(tf.keras.layers.Conv1D(64, 512, padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Dropout(rate=0.1))

    model.add(tf.keras.layers.Conv1D(16, 1024, padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.summary()
    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_sound)
print (decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Making checkpoints
import os
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 500
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
#training
def train_step(training_data):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_sounds = generator(noise, training=True)

        real_output = discriminator(training_data, training=True)
        fake_output = discriminator(generated_sounds, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def train(dataset, epochs):
        for epoch in range(epochs):
         start = time.time()

         for batch in dataset:
                train_step(batch)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

      # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_sounds(generator, epochs, seed)

    def generate_and_save_sounds(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        for i in range(predictions.shape[0]):
                new_audio = sf.write('new_file.flac', predictions[i], samplerate)
                return new_audio

import time
train(train_dataset, EPOCHS)



