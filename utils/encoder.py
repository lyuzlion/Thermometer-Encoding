import numpy as np
from sklearn.preprocessing import OneHotEncoder


class encoder(object):
    def __init__(self, level):
        self.k = level
        # self.onehotencoder = OneHotEncoder(sparse_output=False)


    def onehotencoder(self, arr):
        assert(len(arr.shape) == 3)
        # print(arr)
        one_hot = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2], self.k), dtype=float)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    one_hot[i][j][k][int(arr[i][j][k])] = 1
        # one_hot[(np.arange(arr.shape[0])[:, None, None], np.arange(arr.shape[1])[None, :, None], np.arange(arr.shape[2])[None, None, :], arr)] = 1
        return one_hot

    """
    input:natural image arr:n*w*h*c
    return: quantisized image n*w*h*c
    """

    def quantization(self, arr):
        quant = np.zeros(arr.shape)
        for i in range(1, self.k):
            quant[arr > 1.0 * i / self.k] += 1
        return quant

    """
    input:quantisized img shape:n*w*h*c
    retun:one-hot coded image shape:n*w*h*c*k
    """

    def onehot(self, arr):
        # n, w, h = arr.shape
        # arr = arr.reshape(-1, h)
        # for i in range(len(arr)):
        #     for j in range(len(arr[i])):
        #         print(arr[i][j])
        # print(n, w, h)
        # print(arr.shape)

        arr = self.onehotencoder(arr)
        # print(self.k)
        # print(arr.shape)
        # arr = arr.reshape(n, w, h, self.k)
        arr = arr.transpose(0, 3, 1, 2)
        return arr

    """
    input:one-hot coded img shape:n*w*h*c*k
    retun:trmp coded image shape:n*w*h*c*k
    """

    def tempcode(self, arr):
        tempcode = np.zeros(arr.shape)
        for i in range(self.k):
            tempcode[:, i, :, :] = np.sum(arr[:, :i + 1, :, :], axis=1)
        return tempcode

    def tempencoding(self,arr):
        return self.tempcode(self.onehot(self.quantization(arr)))

    def onehotencoding(self,arr):
        return self.onehot(self.quantization(arr))


    """
    from a thermometerencoding image to a normally coded image, for some visualization usage
    """

    def temp2img(self,tempimg):
        img = np.sum(tempimg, axis=1)
        img = np.ones(img.shape) * (self.k + 1) - img
        img = img * 1.0 / self.k
        return img
