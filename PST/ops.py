
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

class WaveLetPooling(tf.keras.layers.Layer):
    """
    Implemetation of Wavelet Pooing
    """
    def __init__(self, name):
        super(WaveLetPooling, self).__init__()
        self._name = name
        square_of_2 = tf.math.sqrt(tf.constant(2, dtype=tf.float32))
        L = tf.math.divide(
            tf.constant(1, dtype=tf.float32),
            tf.math.multiply(square_of_2, tf.constant([[1, 1]], dtype=tf.float32))
        )
        H = tf.math.divide(
            tf.constant(1, dtype=tf.float32),
            tf.math.multiply(square_of_2, tf.constant([[-1, 1]], dtype=tf.float32))
        )

        self.LL = tf.reshape(tf.math.multiply(tf.transpose(L), L), (1, 2, 2, 1))
        self.LH = tf.reshape(tf.math.multiply(tf.transpose(L), H), (1, 2, 2, 1))
        self.HL = tf.reshape(tf.math.multiply(tf.transpose(H), L), (1, 2, 2, 1))
        self.HH = tf.reshape(tf.math.multiply(tf.transpose(H), H), (1, 2, 2, 1))

    def call(self, inputs):
        LL, LH, HL, HH = self.repeat_filters(inputs.shape[-1])
        return [_conv2d(inputs, LL),
                _conv2d(inputs, LH),
                _conv2d(inputs, HL),
                _conv2d(inputs, HH)]

    def compute_output_shape(self, input_shape):
        shape = (
            input_shape[0], input_shape[1] // 2,
            input_shape[2] // 2, input_shape[3]
        )

        return [shape, shape, shape, shape]

    def repeat_filters(self, repeats):
        # Can we optimize this?
        return [
            tf.transpose(tf.repeat(self.LL, repeats, axis=0), (1, 2, 3, 0)),
            tf.transpose(tf.repeat(self.LH, repeats, axis=0), (1, 2, 3, 0)),
            tf.transpose(tf.repeat(self.HL, repeats, axis=0), (1, 2, 3, 0)),
            tf.transpose(tf.repeat(self.HH, repeats, axis=0), (1, 2, 3, 0))
        ]


class WaveLetUnPooling(tf.keras.layers.Layer):
    """
    Implementation of WaveLet Unpooling
    """
    def __init__(self, name):
        super(WaveLetUnPooling, self).__init__()
        self._name = name
        square_of_2 = tf.math.sqrt(tf.constant(2, dtype=tf.float32))
        L = tf.math.divide(
            tf.constant(1, dtype=tf.float32),
            tf.math.multiply(square_of_2, tf.constant([[1, 1]], dtype=tf.float32))
        )
        H = tf.math.divide(
            tf.constant(1, dtype=tf.float32),
            tf.math.multiply(square_of_2, tf.constant([[-1, 1]], dtype=tf.float32))
        )

        self.LL = tf.reshape(tf.math.multiply(tf.transpose(L), L), (1, 2, 2, 1))
        self.LH = tf.reshape(tf.math.multiply(tf.transpose(L), H), (1, 2, 2, 1))
        self.HL = tf.reshape(tf.math.multiply(tf.transpose(H), L), (1, 2, 2, 1))
        self.HH = tf.reshape(tf.math.multiply(tf.transpose(H), H), (1, 2, 2, 1))

    def call(self, inputs):
        LL_in, LH_in, HL_in, HH_in, tensor_in = inputs
        LL, LH, HL, HH = self.repeat_filters(LL_in.shape[-1])
        out_shape = tf.shape(tensor_in)

        return tf.concat([
            _conv2d_transpose(LL_in, LL, output_shape=out_shape),
            _conv2d_transpose(LH_in, LH, output_shape=out_shape),
            _conv2d_transpose(HL_in, HL, output_shape=out_shape),
            _conv2d_transpose(HH_in, HH, output_shape=out_shape),
            tensor_in,
        ], axis=-1)

    def compute_output_shape(self, input_shape):
        _ip_shape = input_shape[0]
        shape = (
            _ip_shape[0],
            _ip_shape[1] * 2,
            _ip_shape[2] * 2,
            sum(ips[3] for ips in input_shape)
        )

        return shape

    def repeat_filters(self, repeats):
        # Can we optimize this?
        return [
            tf.transpose(tf.repeat(self.LL, repeats, axis=0), (1, 2, 3, 0)),
            tf.transpose(tf.repeat(self.LH, repeats, axis=0), (1, 2, 3, 0)),
            tf.transpose(tf.repeat(self.HL, repeats, axis=0), (1, 2, 3, 0)),
            tf.transpose(tf.repeat(self.HH, repeats, axis=0), (1, 2, 3, 0)),
        ]

class WhiteningAndColoring(tf.keras.layers.Layer):
    """
    Source: https://github.com/eridgd/WCT-TF/blob/master/ops.py#L24
    """
    def __init__(self, alpha=1.0):
        super(WhiteningAndColoring, self).__init__()
        self.alpha = alpha

    def call(self, inputs):
        content, style = inputs
        eps = 1e-8
        alpha = self.alpha

        content_t = tf.transpose(tf.squeeze(content), (2, 0, 1))
        style_t = tf.transpose(tf.squeeze(style), (2, 0, 1))

        Cc, Hc, Wc = tf.unstack(tf.shape(content_t))
        Cs, Hs, Ws = tf.unstack(tf.shape(style_t))

        # CxHxW -> CxH*W
        content_flat = tf.reshape(content_t, (Cc, Hc * Wc))
        style_flat = tf.reshape(style_t, (Cs, Hs * Ws))

        # Content covariance
        mc = tf.reduce_mean(content_flat, axis=1, keepdims=True)
        fc = content_flat - mc
        fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc * Wc, tf.float32) - 1.) + tf.eye(Cc) * eps

        # Style covariance
        ms = tf.reduce_mean(style_flat, axis=1, keepdims=True)
        fs = style_flat - ms
        fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs * Ws, tf.float32) - 1.) + tf.eye(Cs) * eps

        # tf.svd is slower on GPU, see https://github.com/tensorflow/tensorflow/issues/13603
        with tf.device('/cpu:0'):
            Sc, Uc, _ = tf.linalg.svd(fcfc)
            Ss, Us, _ = tf.linalg.svd(fsfs)

        # Filter small singular values
        k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.float32)).numpy()
        k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.float32)).numpy()

        k_c, k_s = int(k_c), int(k_s)
        # Whiten content feature
        Dc = tf.linalg.diag(tf.pow(Sc[:k_c], -0.5))
        fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:, : k_c], Dc), Uc[:, : k_c], transpose_b=True), fc)

        # Color content with style
        Ds = tf.linalg.diag(tf.pow(Ss[:k_s], 0.5))
        fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:, : k_s], Ds), Us[:, : k_s], transpose_b=True), fc_hat)

        # Re-center with mean of style
        fcs_hat = fcs_hat + ms

        # Blend whiten-colored feature with original content feature
        blended = alpha * fcs_hat + (1 - alpha) * (fc + mc)

        # CxH*W -> CxHxW
        blended = tf.reshape(blended, (Cc, Hc, Wc))
        # CxHxW -> 1xHxWxC
        blended = tf.expand_dims(tf.transpose(blended, (1, 2, 0)), 0)

        return blended


class TfReduceSum(tf.keras.layers.Layer):
    def __init__(self):
        super(TfReduceSum, self).__init__()

    def call(self, inputs):
        return tf.reduce_sum(inputs)


def _conv2d_transpose(x, kernel, output_shape):
    conv = tf.nn.conv2d_transpose(
            x, kernel,
            output_shape=output_shape,
            strides=[1, 2, 2, 1],
            padding='SAME')
    return conv


def _conv2d(x, kernel):
    conv = tf.nn.conv2d(x, kernel, strides=[1, 2, 2, 1], padding='SAME')
    return conv


def _get_output(x, layer):
    if "_pooling" in layer.name:
        # return 4 outputs
        ll, lh, hl, hh = layer(x)
        return ll, [lh, hl, hh, x]
    return layer(x), None


def _copy_input(layer):
    # :1 to remove batch_size
    if hasattr(layer, 'input_shape'):
        ip_shape = layer.input_shape[1:]
    else:
        ip_shape = layer.shape[1:]
    return tf.keras.layers.Input(shape=ip_shape)


def get_predict_function(model, layers, name):
    skips_out = None

    if layers[0] == 'in_img':
        ip = model.get_layer(layers[0]).input
        start = 1
    elif 'unpooling' in layers[0]:
        # multi inputs
        ip = [
            _copy_input(l) for l in model.get_layer(layers[0]).input
        ]
        start = 0
    else:
        ip = _copy_input(model.get_layer(layers[0]))
        start = 0

    x, skips = _get_output(ip, model.get_layer(layers[start]))

    if skips is not None:
        skips_out = skips

    for layer in layers[start + 1:]:
        x, skips = _get_output(x, model.get_layer(layer))
        if skips is not None:
            skips_out = skips

    outputs = [x] if skips_out is None else [x, skips_out]
    return tf.keras.models.Model(inputs=ip, outputs=outputs, name=name)


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)
