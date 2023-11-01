from collections import OrderedDict
import flwr as fl
import argparse

from utils.util_class import CNNModel
from utils.util_function import *

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, dataset, poison_rate=0, perturb_weights=0):
        super().__init__()
        self.model, self.dataset, self.poison_rate, self.perturb_weights = model, dataset, poison_rate, perturb_weights
        self.train_dataloader, self.train_examples_using = get_train_dataloader(dataset=dataset, poison_rate=poison_rate)
        self.test_dataloader, self.test_examples_using = get_test_dataloader(dataset=dataset, poison_rate=poison_rate)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({key: torch.tensor(value) for key, value in params_dict})
        self.model.load_state_dict(state_dict=state_dict, strict=True)

    def get_parameters(self, config):
        return [value.cpu().numpy() for _, value in self.model.state_dict().items()]

    def fit(self, parameters, config):
        if self.perturb_weights > 0:
            parameters = [params + np.random.normal(size=params.shape) * self.perturb_weights for params in parameters]

        self.set_parameters(parameters=parameters)
        train(model=self.model, train_dataloader=self.train_dataloader, num_epochs=1)
        metrics, config = {}, {}
        return self.get_parameters(config=config), self.train_examples_using, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters=parameters)
        loss, accuracy, class_accuracy = test(model=self.model, test_dataloader=self.test_dataloader, dataset=self.dataset)
        accuracy, class_accuracy = {"Cumulative Accuracy": float(accuracy)}, {key: float(value) for key, value in class_accuracy.items()}
        return float(loss), self.test_examples_using, {**accuracy, **class_accuracy}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client script in federated machine learning system.")

    parser.add_argument("--address", default="localhost:8080", type=str, metavar='X', help="Address of server (Default: localhost:8080)", dest="server_address")
    parser.add_argument("--poison", default=0, type=float, metavar='X', help="Percentage of target class samples to poison (Default: 0)", dest="poison_rate")
    parser.add_argument("--perturb", default=0, type=float, metavar='X', help="Weight assigned to Gaussian noise for perturbation (Default: 0)", dest="perturb_weights")
    parser.add_argument("--dataset", default="CIFAR-10", type=str, metavar='X', help="Dataset to use from either CIFAR-10 or MNIST (Default: CIFAR-10)", dest="dataset")

    args = parser.parse_args()

    dataset_parameters_mappings = {"CIFAR-10": {"in_channels": 3, "padding": 0}, "MNIST": {"in_channels": 1, "padding": 2}}

    assert args.dataset in dataset_parameters_mappings.keys(), "Invalid dataset name provided!"

    cnn_model = CNNModel(in_channels=dataset_parameters_mappings[args.dataset]["in_channels"], padding=dataset_parameters_mappings[args.dataset]["padding"])

    if torch.cuda.is_available():
        cnn_model = cnn_model.cuda()

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FLClient(model=cnn_model, dataset=args.dataset, poison_rate=args.poison_rate, perturb_weights=args.perturb_weights)
    )