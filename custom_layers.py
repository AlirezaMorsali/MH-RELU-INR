import tensorflow as tf
import numpy as np




class FourierLayer(tf.keras.layers.Layer):
    def __init__(self, units, scale):
        super(FourierLayer, self).__init__()
        self.units = units
        self.scale = scale


    def build(self, input_shape):
        self.in_features = int(input_shape[-1])
        self.B = tf.random.normal((self.in_features, self.units))

        super(FourierLayer, self).build(input_shape)

    def call(self, inputs):
        out = tf.concat([tf.cos(self.scale * tf.matmul(inputs, self.B)), tf.sin(self.scale * tf.matmul(inputs, self.B))], axis=-1)
        return out





class SineLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, units, bias=True, is_first=False, omega_0=30.):
        super(SineLayer, self).__init__()
        self.in_features = in_features
        self.units = units
        self.is_first = is_first
        self.omega_0 = omega_0

        self.dense = tf.keras.layers.Dense(self.units,
                                           use_bias=bias,
                                           kernel_initializer=self.init_weights(),
                                           input_shape=(self.in_features,))
        
    
    def init_weights(self):
        if self.is_first:
            return tf.keras.initializers.RandomUniform(minval=-1 / self.in_features,
                                                       maxval= 1 / self.in_features)
        else:
            return tf.keras.initializers.RandomUniform(minval=-np.sqrt(6. / self.in_features) / self.omega_0,
                                                       maxval= np.sqrt(6. / self.in_features) / self.omega_0)
    

    def call(self, input_tensor):
        befor_activation = self.dense(input_tensor)
        after_activation = tf.sin(self.omega_0 * befor_activation)
        return after_activation







class PureSparseLayer(tf.keras.layers.Layer):
    def __init__(self, units, alpha=1, use_bias=True, activation=None, kernel_initializer=None, full="output"):
        super(PureSparseLayer, self).__init__()
        self.units = units
        self.alpha = alpha
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.full = full
        


    def build(self, input_shape):
        self.in_features = int(input_shape[-1])

        

        if self.full not in ["input", "output"]:
            raise NameError('full argument must be "input" or "output"')

        if self.alpha > self.in_features:
            self.alpha = self.in_features
            print(f"alpha set to : {self.alpha}")
        
        n_sparse_parameters = self.alpha * self.units
      
        
        
        if self.full == "input":
            Total_Indexs = []
            for_each_row = n_sparse_parameters // self.in_features
            remain = n_sparse_parameters % self.in_features

            remain_index = np.random.choice(self.in_features, remain, replace=False)
            row_indexs = np.random.choice(self.in_features, self.in_features, replace=False)
            for counter, row_index in enumerate(row_indexs):
                if row_index in remain_index:
                    column_indexs = np.random.choice(self.units, for_each_row + 1, replace=False)
                else:
                    column_indexs = np.random.choice(self.units, for_each_row, replace=False)
                Total_Indexs.append(np.stack([row_index * np.ones_like(column_indexs), column_indexs], axis=1))

            self.Total_Indexs = np.concatenate(Total_Indexs, axis=0)
        elif self.full == "output":
            Total_Indexs = []
            for_each_column = n_sparse_parameters // self.units
            remain = n_sparse_parameters % self.units

            remain_index = np.random.choice(self.units, remain, replace=False)
            column_indexs = np.random.choice(self.units, self.units, replace=False)
            for counter, column_index in enumerate(column_indexs):
                if column_index in remain_index:
                    row_indexs = np.random.choice(self.in_features, for_each_column + 1, replace=False)
                else:
                    row_indexs = np.random.choice(self.in_features, for_each_column, replace=False)
                Total_Indexs.append(np.stack([row_indexs, column_index * np.ones_like(row_indexs)], axis=1))

            self.Total_Indexs = np.concatenate(Total_Indexs, axis=0)
        else:
            raise NameError('full argument must be "input" or "output"')
            
            
        
        if self.kernel_initializer is None:
            self.kernel = tf.Variable(tf.initializers.glorot_uniform()((n_sparse_parameters,)), trainable=True)
        else:
            self.kernel = tf.Variable(self.kernel_initializer((n_sparse_parameters,)), trainable=True)

            
            
        if self.use_bias:
            self.bias = tf.Variable(tf.zeros((self.units,)), trainable=True)

        super(PureSparseLayer, self).build(input_shape)


    def call(self, inputs):
        new_kernel = tf.SparseTensor(indices=self.Total_Indexs,
                                     values=self.kernel,
                                     dense_shape=(self.in_features, self.units))
      
        out = tf.sparse.sparse_dense_matmul(inputs, new_kernel)
        if self.use_bias:
            out = out + self.bias
        if self.activation is not None:
            return self.activation(out) 
        return out







class QuasiSparseLayer(tf.keras.layers.Layer):
    def __init__(self, units, alpha=1, use_bias=True, activation=None, kernel_initializer=None, full="output"):
        super(QuasiSparseLayer, self).__init__()
        self.units = units
        self.alpha = alpha
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.full = full
        


    def build(self, input_shape):
        self.in_features = int(input_shape[-1])

        

        if self.full not in ["input", "output"]:
            raise NameError('full argument must be "input" or "output"')

        if self.alpha > self.in_features:
            self.alpha = self.in_features
            print(f"alpha set to : {self.alpha}")
        
        n_sparse_parameters = self.alpha * self.units
      
        
        
        if self.full == "input":
            Total_Indexs = []
            for_each_row = n_sparse_parameters // self.in_features
            remain = n_sparse_parameters % self.in_features

            remain_index = np.random.choice(self.in_features, remain, replace=False)
            row_indexs = np.random.choice(self.in_features, self.in_features, replace=False)
            for counter, row_index in enumerate(row_indexs):
                if row_index in remain_index:
                    column_indexs = np.random.choice(self.units, for_each_row + 1, replace=False)
                else:
                    column_indexs = np.random.choice(self.units, for_each_row, replace=False)
                Total_Indexs.append(np.stack([row_index * np.ones_like(column_indexs), column_indexs], axis=1))

            self.Total_Indexs = np.concatenate(Total_Indexs, axis=0)
        elif self.full == "output":
            Total_Indexs = []
            for_each_column = n_sparse_parameters // self.units
            remain = n_sparse_parameters % self.units

            remain_index = np.random.choice(self.units, remain, replace=False)
            column_indexs = np.random.choice(self.units, self.units, replace=False)
            for counter, column_index in enumerate(column_indexs):
                if column_index in remain_index:
                    row_indexs = np.random.choice(self.in_features, for_each_column + 1, replace=False)
                else:
                    row_indexs = np.random.choice(self.in_features, for_each_column, replace=False)
                Total_Indexs.append(np.stack([row_indexs, column_index * np.ones_like(row_indexs)], axis=1))

            self.Total_Indexs = np.concatenate(Total_Indexs, axis=0)
        else:
            raise NameError('full argument must be "input" or "output"')
            
        

        self.Mask = np.zeros((self.in_features, self.units))
        self.Mask[self.Total_Indexs[:,0], self.Total_Indexs[:,1]] = 1
        self.Mask = tf.constant(self.Mask, dtype=tf.float32)

        if self.kernel_initializer is None:
            self.kernel = tf.Variable(tf.initializers.glorot_uniform()((self.in_features, self.units)), trainable=True)
        else:
            self.kernel = tf.Variable(self.kernel_initializer((self.in_features, self.units)), trainable=True)

            
            
        if self.use_bias:
            self.bias = tf.Variable(tf.zeros((self.units,)), trainable=True)

        super(QuasiSparseLayer, self).build(input_shape)


    def call(self, inputs):
        new_kernel = self.Mask * self.kernel
      
        out = tf.matmul(inputs, new_kernel)
        if self.use_bias:
            out = out + self.bias
        if self.activation is not None:
            return self.activation(out) 
        return out