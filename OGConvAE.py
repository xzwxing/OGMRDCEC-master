import numpy as np
import tensorflow as tf
from keras.layers import Layer, Dense, Flatten, Reshape,Lambda, Input
from keras.models import Sequential, Model
from keras.callbacks import CSVLogger
from sklearn.cluster import KMeans


# --- OG functions ---
def OG_function(x, w, og_type=1):
    if og_type == 1:
        return x**3 * w**3 + x**2 * w**2 + x*w
    elif og_type == 2:
        return x**3 * w**3 + x**2 * w**2
    elif og_type == 3:
        return x**3 * w**3 + x*w
    elif og_type == 4:
        return x**3 * w**3
    elif og_type == 5:
        return x**2 * w**2 + x*w
    elif og_type == 6:
        return x**2 * w**2
    else:
        raise ValueError("og_type must be 1–6")

def OG_wrapper(patches, kernel, og_type, filters):
    patches = tf.expand_dims(patches, -1)
    og_value = OG_function(patches, kernel, og_type=og_type)
    output = tf.reduce_sum(og_value, axis=3)
    return output

class OGDense(Layer):
    def __init__(self, units, activation=None, og_type=1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_fn = tf.keras.activations.get(activation)
        self.og_type = og_type

    def build(self, input_shape):
        in_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(in_dim, self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.expand_dims(inputs, -1)
        w = tf.expand_dims(self.kernel, 0)

        og = OG_function(x, w, og_type=self.og_type)
        out = tf.reduce_sum(og, axis=1)

        out = out + self.bias
        return self.activation_fn(out) if self.activation_fn else out

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": tf.keras.activations.serialize(self.activation_fn),
            "og_type": self.og_type
        })
        return config

class OGConv2D(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding="valid",
                 activation=None, og_type=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding_type = padding.lower()
        self.activation_fn = tf.keras.activations.get(activation)
        self.og_type = og_type

    def build(self, input_shape):
        kh, kw = self.kernel_size
        in_channels = input_shape[-1]
        self.in_channels = in_channels
        self.kernel = self.add_weight(name="kernel",
                                      shape=(kh, kw, in_channels, self.filters),
                                      initializer="glorot_uniform",
                                      trainable=True)
        self.bias = self.add_weight(name="bias",
                                    shape=(self.filters,),
                                    initializer="zeros",
                                    trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1,1,1,1],
            padding=self.padding_type.upper()
        )
        B = tf.shape(patches)[0]
        H = patches.shape[1]
        W = patches.shape[2]
        KC = patches.shape[3]

        patches = tf.reshape(patches, [B, H, W, KC])
        kernel = tf.reshape(self.kernel, [1,1,1,KC,self.filters])
        output = OG_wrapper(patches, kernel, self.og_type, self.filters)
        output = output + self.bias
        return self.activation_fn(output) if self.activation_fn else output

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 4:
            batch_size = input_shape[0]
            h = input_shape[1]
            w = input_shape[2]
        else:
            batch_size = None
            h = input_shape[0]
            w = input_shape[1]
            
        stride_h, stride_w = self.strides
        kh, kw = self.kernel_size

        out_h = None
        if h is not None:
            if self.padding_type == "same":
                out_h = int(np.ceil(h / stride_h))
            else:
                out_h = int(np.floor((h - kh + 1) / stride_h))
        
        out_w = None
        if w is not None:
            if self.padding_type == "same":
                out_w = int(np.ceil(w / stride_w))
            else:
                out_w = int(np.floor((w - kw + 1) / stride_w))

        if len(input_shape) == 4:
            return (batch_size, out_h, out_w, self.filters)
        else:
            return (out_h, out_w, self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding_type,
            "activation": tf.keras.activations.serialize(self.activation_fn),
            "og_type": self.og_type
        })
        return config

class OGTransposeConv2D(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding="valid",
                 activation=None, og_type=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding_type = padding.lower()
        self.activation_fn = tf.keras.activations.get(activation)
        self.og_type = og_type

    def build(self, input_shape):
        kh, kw = self.kernel_size
        in_channels = input_shape[-1]
        self.in_channels = in_channels
        self.kernel = self.add_weight(name="kernel",
                                      shape=(kh, kw, self.filters, in_channels),
                                      initializer="glorot_uniform",
                                      trainable=True)
        self.bias = self.add_weight(name="bias",
                                    shape=(self.filters,),
                                    initializer="zeros",
                                    trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        h = inputs.shape[1] * self.strides[0] if inputs.shape[1] is not None else None
        w = inputs.shape[2] * self.strides[1] if inputs.shape[2] is not None else None
        out_shape = [batch, h, w, self.filters] if h is not None and w is not None else None
        trans = tf.nn.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape=tf.convert_to_tensor(out_shape) if out_shape is not None else None,
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding_type.upper()
        )
        og_out = OG_function(trans, 1.0, self.og_type)
        og_out = og_out + self.bias
        return self.activation_fn(og_out) if self.activation_fn else og_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding_type,
            "activation": tf.keras.activations.serialize(self.activation_fn),
            "og_type": self.og_type
        })
        return config


def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 32], og_type=1):
    model = Sequential()

    # ---------------- encoder ----------------
    model.add(Input(shape=input_shape))
    model.add(OGConv2D(filters=filters[0], kernel_size=5, strides=2,
                       padding='same', activation='relu', og_type=og_type,
                       name='og_conv1'))
    model.add(OGConv2D(filters=filters[1], kernel_size=5, strides=2,
                       padding='same', activation='relu', og_type=og_type,
                       name='og_conv2'))
    model.add(OGConv2D(filters=filters[2], kernel_size=3, strides=2,
                       padding='same', activation='relu', og_type=og_type,
                       name='og_conv3'))

    # flatten & embedding
    model.add(Flatten())
    model.add(OGDense(units=filters[3],
                      activation=None,
                      og_type=og_type,
                      name='embedding'))

    # ---------------- decoder----------------
    H, W = input_shape[0], input_shape[1]
    h_enc = int(tf.math.ceil(H / 2 / 2 / 2))
    w_enc = int(tf.math.ceil(W / 2 / 2 / 2))

    model.add(OGDense(units=filters[2] * h_enc * w_enc,
                      activation='relu',
                      og_type=og_type,
                      name='og_dense_decoder'))
    model.add(Reshape((h_enc, w_enc, filters[2])))

    model.add(OGTransposeConv2D(filters=filters[1], kernel_size=3, strides=2,
                                padding='same', activation='relu', og_type=og_type, name='og_deconv3'))
    model.add(OGTransposeConv2D(filters=filters[0], kernel_size=5, strides=2,
                                padding='same', activation='relu', og_type=og_type, name='og_deconv2'))
    model.add(OGTransposeConv2D(filters=input_shape[2], kernel_size=5, strides=2,
                                padding='same', og_type=og_type, name='og_deconv1'))

    model.add(Lambda(lambda x: x[:, :H, :W, :], name='output_crop'))

    return model

if __name__ == "__main__":
    from datasets import load_mnist
    x, y = load_mnist()
    x = x.astype('float32') / 255.0

    model = CAE(input_shape=x.shape[1:], filters=[32,64,128,10], og_type=1)
    model.compile(optimizer='adam', loss='mse')
    csv_logger = CSVLogger('pretrain-log.csv')

    model.fit(x, x, batch_size=256, epochs=10, callbacks=[csv_logger])

    feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
    features = feature_model.predict(x)
    features = np.reshape(features, (features.shape[0], -1))

    km = KMeans(n_clusters=10)
    pred = km.fit_predict(features)
    print("Clustering finished.")
