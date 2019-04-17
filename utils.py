import numpy as np
from os.path import join
import cv2

# From keras
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    categorical = np.moveaxis(categorical,-1,1)
    return categorical

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_train_val(dataset, val_percent=0.05):
    """splits dataset into train and val dictionary"""
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    np.random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}

def load_resize(ids, dir, scale, mask=False):
    """Loads numpys and rescales if necessary"""
    for id in ids:

        # Load
        im = np.load(join(dir,id))
        if im.shape[0]!=im.shape[1]:
            print('length not equal to width')

        if np.max(im)>1 and mask ==True:
            print('mask label is not 1')
            im[np.nonzero(im)]=1
        # Rescale imgs
        if scale != 1 and mask == True:
            im = cv2.resize(im, None, scale=scale, interpolation=cv2.INTER_NEAREST)

        # Rescale segmentations
        elif scale != 1 and mask == False:
            im = cv2.resize(im, None, scale=scale, interpolation=cv2.INTER_CUBIC)

        # Generator
        yield im

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""
    for id in ids:
        imgs = load_resize(ids, dir_img, scale,mask=False)
        masks = load_resize(ids, dir_mask, scale,mask=True)

    return zip(imgs, masks)