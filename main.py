from GAN import GAN
import os

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(base_path, 'data')

    gan = GAN(base_path=base_path, data_path=data_path)
    gan.train()