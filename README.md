# Defense Against the Backdoor Attacks in Federated Learning

This repository contains codes and scripts belonging to the CSE6801 (Distributed Computing Systems) project. In this project, we attempted to detect and mitigate the effect of backdoor attacks in federated machine learning. We used [**Flower**](https://github.com/adap/flower) framework to implement the federated learning system. Therefore, we can simulate a federated learning with a server and multiple clients locally. Besides, we can actually train and evaluate a model with federated learning setting, involving multiple distributed machines, by providing the network address of the server machine (*server address can be provided as a command line argument*). In this project, we used [**CIFAR-10 dataset**](https://www.cs.toronto.edu/~kriz/cifar.html) to train and evaluate a model.

***Keep in mind that the codes and scripts in this repository do not have the implementation of majority consensus, a crucial part of our approach. This part was implemented by Md Hasebul Hasan and you will find the full implementation in [this repository](https://github.com/Hasebul/distributed_computing).***

## Guidelines

- Create a *virtual environment* or *Conda environment* with Python version `3.10`.
- `pip install` the Python packages and libraries, including `flwr`, `torch`, and `torchvision`.
- To run the project, follow the instructions in [**this tutorial**](https://flower.dev/docs/framework/tutorial-quickstart-pytorch.html) with codes and scripts uploaded to this repository.
- You may also download the codes and scripts uploaded to [**this repository**](https://github.com/Hasebul/distributed_computing) to incorporate *majority consensus* in federated learning.

## Examples

- To run the FL server script.
    ```sh
    python fl_server.py -h                 # To see the help messages
    python fl_server.py --address IP:PORT  # To run the server script with server's network address
    python fl_server.py --round 10         # To run the server script with 10 FL rounds
    ```
    
- To run the FL client script.
    ```sh
    python fl_client.py --help             # To see the help messages
    python fl_client.py --address IP:PORT  # To run the server script with server's network address
    python fl_client.py                    # To run the client script as a benign client
    python fl_client.py --bad              # To run the client script as an attacker
    python fl_client.py --perturb          # To perturb weights with Gaussian noise in each training round
    ```

## Contribution

Md Hasebul Hasan and Ajmain Yasar Ahmed Sahil contributed to this project.

## Reference

- [**Flower Python API Reference**](https://flower.dev/docs/framework/ref-api-flwr.html)
- [**The `Strategy` Abstraction**](https://flower.dev/docs/framework/how-to-implement-strategies.html)
- [**Build an FL Strategy from Scratch**](https://flower.dev/docs/framework/tutorial-series-build-a-strategy-from-scratch-pytorch.html)
- [**Add Gaussian Noise to Weights**](https://discuss.pytorch.org/t/is-there-any-way-to-add-noise-to-trained-weights/29829)