import numpy as np
import h5py
import keras
from keras.utils import Sequence

class PCDataGenerator(Sequence):

  def __init__(self, hdf5_file, indices, batch_size=32, shuffle=True):
    """ Data generator for point cloud data stored in HDF5 file.

    Args:
        hdf5_file (str): Path to the HDF5 file containing the point cloud data.
        indices (list of ints): List of indices of the point clouds in the HDF5 file.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        shuffle (bool, optional): Shuffle the data. Defaults to True.
    """    
    self.hdf5_file = hdf5_file
    self.indices = indices
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    return int(np.floor(len(self.indices) / self.batch_size))

  def __getitem__(self, index):
    # Generate indexes of the batch
    batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

    # Generate data
    X, y = self.__data_generation(batch_indices)

    return X, y

  def on_epoch_end(self):
    if self.shuffle:
        np.random.shuffle(self.indices)

  def __data_generation(self, batch_indices):
    features_list = []
    points_list = []
    mask_list = []
    labels_list = []

    with h5py.File(self.hdf5_file, 'r') as f:
      for idx in batch_indices:
        grp = f[idx]
        features_list.append(grp['features'][:])
        points_list.append(grp['points'][:])
        mask_list.append(grp['mask'][:])
        labels_list.append(grp.attrs['label'])

    # Convert lists to numpy arrays
    features = np.array(features_list)
    points = np.array(points_list)
    mask = np.array(mask_list)
    labels = np.array(labels_list)

    return [points, features, mask], labels
