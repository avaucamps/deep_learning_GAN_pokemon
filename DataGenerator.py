import os
from pathlib import Path
import tensorflow as tf
import numpy as np


class DataGenerator:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    
    def get_image_data(self, data_path, image_shape):
        self.image_shape = image_shape
        all_images_path = self.load_filenames(data_path)
        image_count = len(all_images_path)

        image_dataset = tf.data.Dataset.from_tensor_slices(all_images_path)

        image_dataset = image_dataset.shuffle(buffer_size=image_count)
        image_dataset = image_dataset.repeat()

        image_dataset = image_dataset.map(self.parse_image, num_parallel_calls=4)
        image_dataset = image_dataset.map(self.perform_data_augmentation, num_parallel_calls=4)

        image_dataset = image_dataset.batch(self.batch_size)
        image_dataset = image_dataset.prefetch(1)

        return image_dataset, image_count


    def get_random_noise_data(self, n_real_images, random_noise_dimension):
        random_noise = np.random.uniform(-1.0, 1.0, size=[n_real_images, random_noise_dimension]).astype(np.float32)

        tmp = np.copy(random_noise)
        random_noise = np.concatenate((random_noise, tmp))

        random_noise_dataset = tf.data.Dataset.from_tensor_slices(random_noise)
        random_noise_dataset = random_noise_dataset.batch(self.batch_size)
        random_noise_dataset = random_noise_dataset.prefetch(1)

        return random_noise_dataset


    def load_filenames(self, data_path):
        all_images_path = []
        for i in Path(data_path).iterdir():
            all_images_path.append(str(i))

        return all_images_path


    def parse_image(self, filename):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=self.image_shape[2])

        return image


    def perform_data_augmentation(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.resize_images(image, (self.image_shape[0], self.image_shape[1]))
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image / 255

        return image