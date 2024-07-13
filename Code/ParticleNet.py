import tensorflow as tf
from tensorflow import keras
from keras.layers import Lambda
import sys

# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)

def print_shape(x, layer_name):
    tf.print(f"{layer_name} shape:", tf.shape(x))
    return x

class BatchDistanceMatrix(tf.keras.layers.Layer):
    """ Compute the distance matrix between points of a batch of point clouds"""
    
    def __init__(self, name='Initial_Batch_Distance_Matrix'):
        super(BatchDistanceMatrix, self).__init__(name=name)
        
    def call(self, inputs):
        A, B = inputs
        with tf.name_scope('dmat'):
            r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
            r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
            m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
            D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
            return D
        
class knn(tf.keras.layers.Layer):
    """ Get the k nearest neighbors of each point in a batch of point clouds """
    
    def __init__(self, name='Initial_KNN', num_points = 100, k = 20):
        super(knn, self).__init__(name=name + "_k_"+str(k))
        self.num_points = num_points
        self.k = k
        
    def call(self, inputs):
        topk_indices, features = inputs
        with tf.name_scope('knn'):
            queries_shape = tf.shape(features)
            batch_size = queries_shape[0]
            batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, self.num_points, self.k, 1))
            indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
            return tf.gather_nd(features, indices)
        
class ReduceLayer(tf.keras.layers.Layer):
    """ Reduce the tensor """
    
    def __init__(self, name='Initial_ReduceLayer', pooling = 'average', axis = 2):
        super(ReduceLayer, self).__init__(name=name + "_pooling_"+pooling)
        self.pooling = pooling
        self.axis = axis
        
    def call(self, inputs):
        x = inputs
        if self.pooling == 'max':
            return tf.reduce_max(x, axis = self.axis)  # (N, P, C')
        else:
            return tf.reduce_mean(x, axis = self.axis)  # (N, P, C')

class SqueezeLayer(tf.keras.layers.Layer):
    """ Squeeze the tensor """
    
    def __init__(self, name='Initial_Squeeze_Layer', axis = 2):
        super(SqueezeLayer, self).__init__(name=name +"_axis_"+str(axis))
        self.axis = axis
        
    def call(self, inputs):
        x = inputs
        return tf.squeeze(x, axis=self.axis)

class Get_Knn_fts(tf.keras.layers.Layer):
    """ Get the relative coordinates of the knn points """
    
    def __init__(self, name='Initial_Get_Knn_Fts', k = 20):
        super(Get_Knn_fts, self).__init__(name=name+"_k_"+str(k))
        self.k = k
        
    def call(self, inputs):
        fts, knn_fts = inputs
        knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, self.k, 1))  # (N, P, K, C)
        knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)
        return knn_fts
        
class Get_Knn_Index(tf.keras.layers.Layer):
    """ Get the top k index of the distance matrix"""
    def __init__(self, name='Initial_Top_K_Index', k = 20):
        super(Get_Knn_Index, self).__init__(name=name + "_k_"+str(k))
        self.k = k
    
    def call(self, inputs):
        D = inputs
        _, indices = tf.math.top_k(-D, k=self.k + 1)  # (N, P, K+1)
        return indices[:, :, 1:]  # (N, P, K)

class MaskCoordShiftLayer(tf.keras.layers.Layer):
    """ Mask the coordinate shift for non-valid points 
    Change the non-valid points to 999 in order to difficult the selection of them in the knn"""
    def __init__(self, shift_value=999.):
        super(MaskCoordShiftLayer, self).__init__(name = 'MaskCoordShiftLayer')
        self.shift_value = shift_value
        
    def call(self, mask):
        mask = tf.cast(tf.not_equal(mask, 0), dtype='float32')  # 1 if valid
        coord_shift = tf.multiply(self.shift_value, tf.cast(tf.equal(mask, 0), dtype='float32'))  # make non-valid positions to 999
        return coord_shift

class AddShifttoCoordLayer(tf.keras.layers.Layer):
    """ Add the shift value to the coordinates """
    def __init__(self, name):
        super(AddShifttoCoordLayer, self).__init__(name = name)
        
    def call(self, inputs):
        coord_shift, points_or_fts = inputs
        return tf.keras.layers.Add()([coord_shift, points_or_fts])

class BatchNormalLayerfts(tf.keras.layers.Layer):
    def __init__(self, name):
        super(BatchNormalLayerfts, self).__init__(name='%s_fts_bn' % name)
        self.bn = tf.keras.layers.BatchNormalization()
    def call(self, features):
        return tf.squeeze(self.bn(tf.expand_dims(features, axis=2)), axis=2)
      
# EdgeConv Funcion
def edge_conv(points, features, num_points, K, channels, with_bn=True, activation='relu', pooling='average', name='edgeconv'):
    """EdgeConv
    Args:
        K: int, number of neighbors
        in_channels: # of input channels
        channels: tuple of output channels
        pooling: pooling method ('max' or 'average')
    Inputs:
        points: (N, P, C_p)
        features: (N, P, C_0)
    Returns:
        transformed points: (N, P, C_out), C_out = channels[-1]
    """

    # Permite organizar las operaciones de manera jerárquica
    with tf.name_scope('edgeconv'):

        # distance
        D = BatchDistanceMatrix(name = name +"_Batch_Distance_Matrix")([points, points])  # (N, P, P)
        
        indices = Get_Knn_Index(name = name + "_top_k_index", k = K)(D)  # (N, P, K)

        fts = features
        knn_fts = knn(name = name + "_KNN", num_points = num_points, k = K)([indices, fts])
        
        x = Get_Knn_fts(name = name + "_Knn_fts", k = K)([fts, knn_fts]) # (N, P, K, 2*C)
        
        # knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, K, 1))  # (N, P, K, C)
        # knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)

        # x = knn_fts
        for idx, channel in enumerate(channels):
            x = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=False if with_bn else True, kernel_initializer='glorot_normal', name='%s_conv%d' % (name, idx))(x)
            if with_bn:
                x = keras.layers.BatchNormalization(name='%s_bn%d' % (name, idx))(x)
            if activation:
                x = keras.layers.Activation(activation, name='%s_act%d' % (name, idx))(x)

        fts = ReduceLayer(name = name + "Reduce_Layer", pooling = pooling, axis = 2)(x, )  # (N, P, C')
        # if pooling == 'max':
        #     fts = tf.reduce_max(x, axis=2)  # (N, P, C')
        # else:
        #     fts = tf.reduce_mean(x, axis=2)  # (N, P, C')
        
        # shortcut (características residuales)

        sc = keras.layers.Conv2D(channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',
                                 use_bias=False if with_bn else True, kernel_initializer='glorot_normal', name='%s_sc_conv' % name)(tf.keras.backend.expand_dims(features, axis=2))
        if with_bn:
            # x = Lambda(print_shape, arguments={'layer_name': f'{name}_conv{idx}_bn'})(x)
            sc = keras.layers.BatchNormalization(name='%s_sc_bn' % name)(sc)
        sc = SqueezeLayer(name = name + "_SqueezeLayer", axis=2)(sc)

        output = tf.keras.layers.Add(name = name + "_ResidualAddition")([sc, fts])
        if activation:
            return keras.layers.Activation(activation, name='%s_sc_act' % name)(output)  # (N, P, C')
        else:
            return output

def _particle_net_base(points, features=None, mask=None, model_params=None, name='particle_net'):
    # points : (N, P, C_coord)
    # features:  (N, P, C_features), optional
    # mask: (N, P, 1), optinal

    with tf.name_scope(name):
        if features is None:
            features = points

        if mask is not None:
            shift_value = model_params.get("Mask",{}).get("shift") if model_params.get("Mask",{}).get("shift") is not None else 999
            coord_shift = MaskCoordShiftLayer(shift_value=shift_value)(mask)

        fts = BatchNormalLayerfts(name = name)(features)
        
        for layer_idx, layer_param in enumerate(model_params["EdgeConv"]["convparams"]):
            K = layer_param["K"]
            channels = layer_param["channels"]
            # A los supuestos puntos (posteriores características) se le añade 999 si no son válidos (es decir, si es un relleno)
            if layer_idx == 0:
                pts = AddShifttoCoordLayer(name = "AddShifttoCoordLayer"+"_layer_"+str(layer_idx))([coord_shift, points])
            else:
                pts = AddShifttoCoordLayer(name = "AddShifttoCoordLayer"+"_layer_"+str(layer_idx))([coord_shift, fts])
            # pts = tf.keras.layers.Add()([coord_shift, points]) if layer_idx == 0 else tf.keras.layers.Add()([coord_shift, fts])
            
            pooling = model_params.get("EdgeConv",{}).get("pooling") if model_params.get("EdgeConv",{}).get("pooling") is not None else "average"
            with_bn = model_params.get("EdgeConv",{}).get("BN") if model_params.get("EdgeConv",{}).get("BN") is not None else True
            activation = model_params.get("EdgeConv",{}).get("activation") if model_params.get("EdgeConv",{}).get("activation") is not None else "relu"
            fts = edge_conv(pts, fts, model_params["num_points"], K, channels, with_bn = with_bn, activation = activation,
                            pooling=pooling, name='%s_%s%d' % (name, 'EdgeConv', layer_idx))

        if mask is not None:
            fts = tf.keras.layers.Multiply(name = "Apply_Mask_Layer")([fts, mask])

        final_pooling = model_params.get("FinalPooling") if model_params.get("FinalPooling") is not None else "average"
        pool = ReduceLayer(name = "FinalPool", pooling = final_pooling, axis = 1)(fts)  # (N, C)

        if model_params.get("ConvHead") is not None:
            x = pool
            for layer_idx, layer_param in enumerate(model_params["ConvHead"]["params"]):
                units = layer_param["units"]
                drop_rate = layer_param["drop_rate"]
                activation = layer_param["activation"]
                x = keras.layers.Dense(units, activation=activation)(x)
                if drop_rate is not None and drop_rate > 0:
                    x = keras.layers.Dropout(drop_rate)(x)
            out = keras.layers.Dense(model_params["num_classes"], activation='softmax')(x)
            return out  # (N, num_classes)
        else:
            return pool

def get_particle_net(data, model_params=None):
    r"""ParticleNet model from `"ParticleNet: Jet Tagging via Particle Clouds"
    <https://arxiv.org/abs/1902.08570>`_ paper.
    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """
    
    if model_params is None:
        model_params = dict()
        # conv_params: list of tuple in the format (K, (C1, C2, C3))
        model_params["EdgeConv"] = dict()
        model_params["EdgeConv"]["convparams"] = [{"K":16, "channels":(64, 64, 64)},
                                                    {"K":16, "channels":(128, 128, 128)},
                                                    {"K":16, "channels":(256, 256, 256)}]
    else:
        sys.path.append(model_params["model_directory"])

    if data.pc_pos is not None:
        points = keras.Input(name='points', shape=data.input_shapes['points'])
        features = keras.Input(name='features', shape=data.input_shapes['features']) if 'features' in data.input_shapes else None
        mask = keras.Input(name='mask', shape=data.input_shapes['mask']) if 'mask' in data.input_shapes else None
        model_params["num_points"] = data.input_shapes['npoints']
        inputs = {"points" : points, "features": features, "mask": mask}
    else:
        raise ValueError("Data must have the point cloud position information to use ParticleNet")
        
    # points = keras.Input(name='points', shape=(500, 2))
    # features = keras.Input(name='features', shape=(500, 2))
    # mask = keras.Input(name='mask', shape=(500, 1))
    # model_params["num_points"] = 500
    # inputs = [points, features, mask]
    outputs = _particle_net_base(points, features, mask, model_params, name='ParticleNet')
    return keras.Model(inputs=inputs, outputs=outputs, name='ParticleNet'), inputs