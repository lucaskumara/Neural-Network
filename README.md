# Neural-Network
A neural network built from scratch to determine odd parity bit.

## Purpose
This assignment was the final assignment of my intro to artificial intelligence class. The purpose of the assignment was to create a basic neural network and train it be able to identify the correct odd parity bit of a four bit binary number with a decreasing margin of error.

The network was very basic as it only required one hidden layer, however some level of customization was required such as being able to modify the learning rate, the number of input nodes, the number of hidden layer nodes and the number of output nodes. The number of input nodes and output nodes were four and one respectively as the network would take in each bit of the binary number and output what it thought the parity bit would be (1 or 0).

The network is trained over 10,000 epochs and displays information every 200 epochs regarding the mean squared error. Upon completion, the network shows what it has learned and attempts to correctly solve the training examples in the .csv file.

## Usage
You can run the network by simply cloning the repo and running the `network.py` using the command shown below.
```
$ python3 network.py {learning rate} {input nodes} {hidden nodes} {output nodes} {training example .csv file}
```
An example of this would be:
```
$ python3 network.py 0.1 4 8 1 examples.csv
```
This will train the network on the provided .csv file. Remember that you must adjust the number of input/output nodes to fit the data that you wish to train it on.
