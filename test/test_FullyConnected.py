from unittest import TestCase
from FullyConnected import FullyConnected


class TestFullyConnected(TestCase):
    def test_get_model(self):
        FC = FullyConnected(num_hidden_layers=2, num_hidden_nodes=2, num_final_nodes=1)
        FC_model = FC.get_model(input_dim=3)
        self.assertEqual(FC_model.output_shape, (None, 1))
