import tensorflow as tf


class GLULayer(tf.keras.layers.Layer):
    def __init__(self, input_dim=2, units=2, w=None, v=None, dtype='float32'):
        super(GLULayer, self).__init__()

        # introduces this for testing to be deterministic
        if w is not None:
            w_init = w
            self.W = tf.Variable(
                initial_value=w_init,
                trainable=True,
            )

        else:
            w_init = tf.random_normal_initializer()
            self.W = tf.Variable(
                initial_value=w_init(shape=(input_dim, units), dtype=dtype),
                trainable=True,
            )

        # introduces this for testing to be deterministic
        if v is not None:
            v_init = v
            self.V = tf.Variable(
                initial_value=v_init,
                trainable=True,
            )

        else:
            v_init = tf.random_normal_initializer()
            self.V = tf.Variable(
                initial_value=v_init(shape=(input_dim, units), dtype=dtype),
                trainable=True,
            )

        b_init = tf.zeros_initializer()
        c_init = tf.zeros_initializer()

        self.b = tf.Variable(
            initial_value=b_init(shape=(units, 1), dtype=dtype), trainable=True
        )

        self.c = tf.Variable(
            initial_value=c_init(shape=(units, 1), dtype=dtype), trainable=True
        )

    def call(self, X):
        if X.dtype != self.W.dtype:
            raise Exception("Matrices have incompatible data type.")

        if X.shape[1] != self.W.shape[0]:
            raise Exception("Matrices have incompatible shape.")

        linear_transform = tf.linalg.matmul(X, self.W) + self.b
        intermediate_linear_transform = tf.linalg.matmul(X, self.V) + self.c
        sigmoid_non_linear = tf.math.sigmoid(intermediate_linear_transform)

        # element wise multiplication
        return tf.math.multiply(linear_transform, sigmoid_non_linear)


