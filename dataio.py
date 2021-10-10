import tensorflow as tf
import numpy as np





def make_coordiante(shape):
    x_coordinates = np.linspace(-1, +1, shape[0])
    y_coordinates = np.linspace(-1, +1, shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    Coordinates = np.stack([x_coordinates, y_coordinates]).T
    return Coordinates





def prepare_data_grid(Image, n_heads):
    s_n_heads = int(np.sqrt(n_heads))

    image_shape = Image.shape
    part_size = (int(image_shape[0] / s_n_heads), int(image_shape[1] / s_n_heads))

    Image_grided = []
    for counter1 in range(s_n_heads):
        for counter2 in range(s_n_heads):
            Image_grided.append(Image[counter1 * part_size[0]: (counter1 + 1) * part_size[0], counter2 * part_size[1]: (counter2 + 1) * part_size[1]])

    Coordinates = make_coordiante(Image_grided[0].shape)

    Image_grided_RGB = []
    for counter in range(len(Image_grided)):
        Image_grided_RGB.append(Image_grided[counter].flatten())

    Image_grided_RGB = np.stack(Image_grided_RGB).T

    return Coordinates, Image_grided_RGB





def prepare_dataset(Image, n_heads):

    Coordinates, Image_grided_RGB = prepare_data_grid(Image, n_heads)

    image_shape = Image.shape
    dataset = tf.data.Dataset.from_tensor_slices((Coordinates, Image_grided_RGB))

    if int(np.sqrt(n_heads)) != 1:
        dataset = dataset.batch(int(image_shape[0] * image_shape[1]/n_heads))
    else:
        #dataset = dataset.batch(int(image_shape[0] * image_shape[1]/4**2))
        dataset = dataset.batch(int(image_shape[0] * image_shape[1]))
        
    return dataset