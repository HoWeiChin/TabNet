import tensorflow as tf


class FullyConnected:
    def __init__(self, num_hidden_layers, num_hidden_nodes,
                 num_final_nodes, activation="relu"):
        """
        :param num_layers (int): number of hidden layers of data in a batch
        :param num_nodes (int): number of nodes in a hidden layer
        :param activation (str): name of activation function
        :return:
        """

        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.num_final_nodes = num_final_nodes
        self.activation = activation

    def get_model(self, input_dim, batch_size=None):
        """

        :param input_dim: size of vector
        :param batch_size:
        :return:
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(input_dim,),
                                 batch_size=batch_size)
                  )

        for layer_index in range(self.num_hidden_layers):
            model.add(tf.keras.layers.Dense(
                units=self.num_hidden_nodes,
                activation=self.activation
            ))

        model.add(tf.keras.layers.Dense(
                units=self.num_final_nodes,
                activation=self.activation
            ))

        return model


