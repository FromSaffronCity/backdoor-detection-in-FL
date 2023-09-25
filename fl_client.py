from collections import OrderedDict
import flwr as fl
import argparse

from utils.util_class import CNNModel
from utils.util_function import *

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, is_bad_client=False):
        super().__init__()
        self.model = model
        self.is_bad_client = is_bad_client

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({key: torch.tensor(value) for key, value in params_dict})
        self.model.load_state_dict(state_dict=state_dict, strict=True)

    def get_parameters(self, config):
        return [value.cpu().numpy() for _, value in self.model.state_dict().items()]

    def fit(self, parameters, config):
        train_dataloader, num_examples = get_train_dataloader(poison_data=self.is_bad_client)
        self.set_parameters(parameters=parameters)
        train(model=self.model, train_dataloader=train_dataloader, num_epochs=1)
        metrics = {}
        return self.get_parameters(config={}), num_examples, metrics

    def evaluate(self, parameters, config):
        test_dataloader, num_examples = get_test_dataloader(poison_data=self.is_bad_client)
        self.set_parameters(parameters=parameters)
        loss, accuracy = test(model=self.model, test_dataloader=test_dataloader)
        return float(loss), num_examples, {"Accuracy": float(accuracy)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client script in federated machine learning system.")

    parser.add_argument("--address", default="localhost:8080", type=str, metavar='X', help="Address of server (Default:- localhost:8080)", dest="server_address")
    parser.add_argument("--bad", action="store_true", help="Whether to use bad client in federated learning (Default:- False)", dest="is_bad_client")

    parser.set_defaults(is_bad_client=False)
    args = parser.parse_args()

    cnn_model = CNNModel()

    if torch.cuda.is_available():
        cnn_model = cnn_model.cuda()

    fl.client.start_numpy_client(server_address=args.server_address, client=FLClient(model=cnn_model, is_bad_client=args.is_bad_client))