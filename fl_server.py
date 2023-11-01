import flwr as fl
import argparse

def get_metrics(client_metrics):
    mean_accuracy = {}
    
    for index, (num_examples, metrics) in enumerate(client_metrics):
        if index > 0:
            mean_accuracy = {key: value + metrics[key] for key, value in mean_accuracy.items()}
        else:
            mean_accuracy = {key: value for key, value in metrics.items()}
    
    return {key: value / len(client_metrics) for key, value in mean_accuracy.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server script in federated machine learning system.")

    parser.add_argument("--address", default="localhost:8080", type=str, metavar='X', help="Address of server (Default: localhost:8080)", dest="server_address")
    parser.add_argument("--round", default=3, type=int, metavar='X', help="Number of federated learning rounds (Default: 3)", dest="num_rounds")

    args = parser.parse_args()

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=get_metrics)
    )