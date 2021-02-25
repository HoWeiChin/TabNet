from dask import dataframe as dd
import tensorflow as tf


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def __numerise(self):
        # convert to pandas df
        pd_df = dd.read_csv(self.data_path).compute()
        x_data = pd_df.drop(pd_df.columns[-1], axis=1)
        y_data = pd_df.iloc[:, -1]
        return x_data.to_numpy(), y_data.to_numpy()

    def load(self):
        """
        :param tf_buffer (int): number of buffer for tensorflow prefetch function
        :return: tensors
        """
        try:
            x_np_data, y_np_data = self.__numerise()
            x_tensor_data = tf.data.Dataset.from_tensor_slices(x_np_data)
            y_tensor_data = tf.data.Dataset.from_tensor_slices(y_np_data)
            return x_tensor_data, y_tensor_data

        except FileNotFoundError:
            print("File can't be loaded, it's not in the directory.")


