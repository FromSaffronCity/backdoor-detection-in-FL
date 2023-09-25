import flwr as fl
import argparse

def get_metrics(client_metrics):
    correct_predictions = [num_examples * metrics["Accuracy"] for num_examples, metrics in client_metrics]
    total_examples = [num_examples for num_examples, _ in client_metrics]
    return {"Accuracy": sum(correct_predictions) / sum(total_examples)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server script in federated machine learning system.")

    parser.add_argument("--address", default="localhost:8080", type=str, metavar='X', help="Address of server (Default:- localhost:8080)", dest="server_address")
    parser.add_argument("--round", default=3, type=int, metavar='X', help="Number of federated learning rounds (Default:- 3)", dest="num_rounds")

    args = parser.parse_args()

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=get_metrics)
    )