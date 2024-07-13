import numpy as np
import h5py
import keras
from keras.utils import Sequence
from sklearn.utils.class_weight import compute_class_weight

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
    self.get_data_shape()
    self.get_data_distribution()
    self.on_epoch_end()
  
  def get_data_shape(self):
    self.pc_pos = True
    with h5py.File(self.hdf5_file, 'r') as f:
      grp = f[self.indices[0]]
      self.input_shapes = dict()
      self.input_shapes['features'] = (grp['features'].shape[0], grp['features'].shape[1])
      self.input_shapes['points'] = (grp['points'].shape[0], grp['points'].shape[1])
      self.input_shapes['mask'] = (grp['mask'].shape[0], grp['mask'].shape[1])
      self.input_shapes["npoints"] = self.input_shapes['points'][0]
    
    return

  def get_data_distribution(self):
    """ Get the distribution of the data labels to obtain the weights for the loss function.
    """
    self.class_weight = dict()
    with h5py.File(self.hdf5_file, 'r') as f:
      labels = []
      for idx in self.indices:
        grp = f[idx]
        labels.append(grp.attrs['label'])
      labels = np.array(labels)
      labels = np.argmax(labels, axis=1)
      self.class_weight = compute_class_weight(class_weight='balanced', classes = np.unique(labels), y = labels)
      self.class_weight = dict(enumerate(self.class_weight))
    return

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
