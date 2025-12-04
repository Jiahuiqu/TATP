import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat, savemat
from matplotlib import pyplot as plt
import random


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def elastic_transform(image, dx, dy):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    # if random_state is None:
    #     random_state = np.random.RandomState(None)
    #
    # shape = image.shape
    # dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    # dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))

    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')

    return distored_image.reshape(image.shape)


if __name__ == "__main__":
    moved_pix_all = []
    alpha = 200
    sigma = 5

    for i in range(1):
        # img = loadmat('D:/Desktop/dataset/chikusei/test/LRHS/' + str(i) + '.mat')['lrHS'].reshape(128, 64, 64)
        # img = loadmat(r'G:\dataset\pavia\test\gtHS\10.mat')['gtHS'].reshape(102, 160, 160)
        img = loadmat(r'/media/xd132/USER/LXY/AAAI/pavia/train/gtHS/1.mat')['gtHS'].reshape(102, 160, 160)

        img = np.expand_dims(img, axis=0)
        print(i, img.shape)
        img_ = np.zeros_like(img)
        shape = (160, 160, 1)
        random_state = np.random.RandomState(None)
        randomstate = random_state.rand(*shape)
        dx = gaussian_filter((randomstate * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((randomstate * 2 - 1), sigma, mode="constant", cval=0) * alpha
        print(dx.shape)
        print(dy.shape)

        moved_pix = np.sqrt(dx.max() ** 2 + dy.max() ** 2)
        moved_pix_all.append(moved_pix)

        for band in range(img.shape[1]):
            img_[:, band, :, :] = elastic_transform(np.expand_dims(img[:, band, :, :].squeeze(), axis=-1), dx,
                                                    dy).squeeze()
        # savemat('D:/Desktop/dataset/chikusei/test/LRHS_Elastic1000/'+ str(i) +'.mat',{'lrHS':img_})
        # savemat('D:/Desktop/destore.mat',{'destore':img_})
        #        {'LRHS': img_})
        # plt.imshow(img[:, 50, :, :].squeeze(), cmap='gray', vmin=0, vmax=1)
        # plt.show()
        # plt.imshow(img_[:, 50, :, :].squeeze(), cmap='gray', vmin=0, vmax=1)
        # plt.show()
        plt.imshow(img_[:, 60, :, :].squeeze(), cmap='gray', vmin=0, vmax=1)
        plt.show()
    print(np.mean(moved_pix_all))
    print(np.max(moved_pix_all))