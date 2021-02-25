from unittest import TestCase
from GLU import GLULayer
import tensorflow as tf
import numpy as np


class TestGLULayer(TestCase):

    def test_call_compatible_datatype(self):
        glu = GLULayer(input_dim=4, units=2,
                       w=tf.constant([2, 2, 2, 2], shape=(2, 2), dtype='float32'),
                       v=tf.constant([3, 3, 3, 3], shape=(2, 2), dtype='float32'),
                       dtype='float32')
        X = tf.constant([1, 1, 1, 1], shape=(2, 2), dtype='float32')
        out = glu(X)
        result = tf.constant([3.99, 3.99, 3.99, 3.99], shape=(2, 2), dtype='float32')

        np.testing.assert_array_almost_equal(out, result, decimal=2)

    def test_call_incompatible_datatype(self):
        with self.assertRaises(Exception) as context:
            glu = GLULayer(input_dim=4, units=2,
                           w=tf.constant([2, 2, 2, 2], shape=(2, 2), dtype='float32'),
                           v=tf.constant([3, 3, 3, 3], shape=(2, 2), dtype='float32'),
                           dtype='float32')
            X = tf.constant([1, 1, 1, 1], shape=(2, 2), dtype='int32')
            glu(X)

        expected_exception = "Matrices have incompatible data type."
        self.assertEqual(expected_exception, str(context.exception))

    def test_call_incompatible_shape(self):
        with self.assertRaises(Exception) as context:
            glu = GLULayer(input_dim=4, units=2,
                           w=tf.constant([2, 2, 2, 2], shape=(2, 2), dtype='float32'),
                           v=tf.constant([3, 3, 3, 3], shape=(2, 2), dtype='float32'),
                           dtype='float32')
            X = tf.constant([1, 1, 1, 1], shape=(1, 4), dtype='float32')
            glu(X)

        expected_exception = "Matrices have incompatible shape."
        self.assertEqual(expected_exception, str(context.exception))
