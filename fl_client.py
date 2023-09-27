from collections import OrderedDict
import flwr as fl
import argparse

from utils.util_class import CNNModel
from utils.util_function import *

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, is_bad_client=False, perturb_weights=False):
        super().__init__()
        self.model, self.is_bad_client, self.perturb_weights = model, is_bad_client, perturb_weights
        self.train_dataloader, self.train_examples_using = get_train_dataloader(poison_data=self.is_bad_client)
        self.test_dataloader, self.test_examples_using = get_test_dataloader(poison_data=self.is_bad_client)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({key: torch.tensor(value) for key, value in params_dict})
        self.model.load_state_dict(state_dict=state_dict, strict=True)

    def get_parameters(self, config):
        return [value.cpu().numpy() for _, value in self.model.state_dict().items()]

    def fit(self, parameters, config):
        if self.perturb_weights:
            parameters = [param + np.random.normal(size=param.shape) * 0.1 for param in parameters]

        self.set_parameters(parameters=parameters)
        train(model=self.model, train_dataloader=self.train_dataloader, num_epochs=1)
        metrics = {}
        return self.get_parameters(config={}), self.train_examples_using, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters=parameters)
        loss, accuracy, class_accuracy = test(model=self.model, test_dataloader=self.test_dataloader)
        accuracy, class_accuracy = {"Cumulative Accuracy": float(accuracy)}, {key: float(value) for key, value in class_accuracy.items()}
        return float(loss), self.test_examples_using, {**accuracy, **class_accuracy}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client script in federated machine learning system.")

    parser.add_argument("--address", default="localhost:8080", type=str, metavar='X', help="Address of server (Default:- localhost:8080)", dest="server_address")
    parser.add_argument("--bad", action="store_true", help="Whether to use bad client in federated learning (Default:- False)", dest="is_bad_client")
    parser.add_argument("--perturb", action="store_true", help="Whether to perturb weights in each round (Default:- False)", dest="perturb_weights")

    parser.set_defaults(is_bad_client=False)
    parser.set_defaults(perturb_weights=False)

    args = parser.parse_args()
    cnn_model = CNNModel()

    if torch.cuda.is_available():
        cnn_model = cnn_model.cuda()

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FLClient(model=cnn_model, is_bad_client=args.is_bad_client, perturb_weights=args.perturb_weights)
    )