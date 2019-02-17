import numpy as np
import matplotlib.pyplot as plt

def dataIterator(data, batch_size):
    """
    From great jupyter notebook by Tim Sainburg:
    http://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN
    This contains some changes
    Args:
        data:
        batch_size:

    Returns:

    """

    while True:
        length = data.shape[0] #
        idxs = np.arange(0, length)
        np.random.shuffle(idxs)
        for batch_idx in range(0, length, batch_size):
            if batch_idx + batch_size > length:
                break
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            batch = data[cur_idxs]
            yield batch


def sample_images(imgs, epoch, path):
    r, c = 5, 5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    # fig.savefig("images/%d.png" % epoch)
    fig.savefig('{}\\{}.png'.format(path, epoch))
    plt.close()
