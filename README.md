# Defense Against the Backdoor Attacks in Federated Learning

This repository contains codes and scripts belonging to the CSE6801 (Distributed Computing Systems) project. In this project, we attempted to detect and mitigate the effect of backdoor attacks in federated machine learning. We used [**Flower**](https://github.com/adap/flower) framework to implement the federated learning system.

## Guidelines

- Create a *virtual environment* or *Conda environment* with Python version `3.10`.
- `pip install` the Python packages and libraries, including Flower (`flwr`), PyTorch (`torch`), and TorchVision (`torchvision`).
- For running the project, follow the instructions provided in [**this tutorial**](https://flower.dev/docs/framework/tutorial-quickstart-pytorch.html) with the scripts uploaded to this repository.

## Examples

- To run the FL server script.

    ```sh
    python fl_server.py -h OR python fl_server.py --help  # To see the help messages
    python fl_server.py --round 7                         # To run the server script
    ```

- To run the FL client script.

    ```sh
    python fl_client.py -h OR python fl_client.py --help  # To see the help messages
    python fl_client.py                                   # To run the client script as a benign client
    python fl_client.py --bad                             # To run the client script as an attacker
    ```

## Contribution

Md Hasebul Hasan and Ajmain Yasar Ahmed Sahil contributed to this project.

## Reference

- [**Flower Python API Reference**](https://flower.dev/docs/framework/ref-api-flwr.html)
- [**The `Strategy` Abstraction**](https://flower.dev/docs/framework/how-to-implement-strategies.html)
- [**Build an FL Strategy from Scratch**](https://flower.dev/docs/framework/tutorial-series-build-a-strategy-from-scratch-pytorch.html)
