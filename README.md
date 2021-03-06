# Can we learn gradients by Hamiltonian Neural Networks?

This project was carried out as part of the Optimization for Machine Learning course (CS-439) at EPFL in the spring 2020 semester. 

*Team*:

   - Aleksandr Timofeev (<aleksandr.timofeev@epfl.ch>)
   - Andrei Afonin (<andrei.afonin@epfl.ch>)
   - Yehao Liu (<yehao.liu@epfl.ch>)
    
 
The No Free Lunch Theorem suggests that there is no universally best learner and restricting the hypothesis class by introducing our prior knowledge about the task we are solving is the only way we can improve the state of affairs. This motivates the use of the learned optimizer for the given task and the use of different regularization methods. For instance, the Heavy Ball method considers the gradient descent procedure as a sliding of a heavy ball on the surface of the loss function, which results in faster convergence. More generally, one can consider the gradient descent procedure as a movement of some object on the surface of the loss function under different forces: potential, dissipative (friction) and other external forces. Such a physical process can be described by port-Hamiltonian system of equations. In this work, we propose to learn the optimizer and impose the physical laws governed by the port-Hamiltonian system of equations into the optimization algorithm to provide implicit bias which acts as regularization and helps to find the better generalization optimums. We impose physical structure by learning the gradients of the parameters: gradients are the solutions of the port-Hamiltonian system, thus their dynamics is governed by the physical laws, that are going to be learned.

To summarize, we propose a new framework based on Hamiltonian Neural Networks which is used to learn and improve gradients for   the gradient descent step. Our experiments on an artificial task and MNIST dataset demonstrate that our method is able to outperform many basic optimizers and achieve comparable performance to the previous LSTM-based one. Furthermore, we explore how methods can be transferred to other architectures with different hyper-parameters, e.g. activation functions. To this end, we train HNN-based optimizer for a small neural network with the sigmoid activation on MNIST dataset and then train the same network but with the ReLU activation using the already trained optimizer. The results show that our method is transferable in this case unlike the LSTM-based optimizer.

To test optimizers we use the following tasks:

- [x] Quadratic functions (details are given in `main.ipynb`)
- [x] MNIST

## Prerequisites
- Ubuntu
- Python 3
- NVIDIA GPU

## Installation
* Clone this repo:
```bash
git clone https://github.com/AfoninAndrei/OPT-ML.git
cd OPT-ML
```
* Install dependencies:
```bash
pip install requirements.txt
```
## Usage
* To reproduce the results: simply go through `main.ipynb`. Or run it on [Colab](https://colab.research.google.com/drive/1TREA4XwnU2WrxIGyx5XlIGJprtsAoUN_?usp=sharing)
* All implementations are in `src`.

## Method

In fact, gradient descent is fundamentally a **sequence** of updates (from the output layer of the neural net back to the input), in between which a **state** must be stored. Thus we can think of an optimizer as a simple feedforward network (or RNN, etc.) that gives us nest update each iteration. 
The loss of the optimizer is the sum (weights are set to 1 in our experiments) of the losses of the optimizee as it learns. 

<img src="figs/loss.png" width="600" />

The plan is thus to use gradient descent on parameters of model-based optimizers in order to minimize this loss, which should give us an optimizer that is capable of optimizing efficiently.

As the [paper](https://arxiv.org/pdf/1606.04474.pdf) mentions, it is important that the gradients in dashed lines in the figure below are **not** propagated during gradient descent.

<img src="figs/backprop.png" width="600" />

Basically this is nothing we wouldn't expect: the loss of the optimizer neural net is simply the average training loss of the optimizee as it is trained by the optimizer. The optimizer takes in the gradient of the current coordinate of the optimizee as well as its previous state, and outputs a suggested update that we hope will reduce the optimizee's loss as fast as possible.

Optimization is done coordinatewise such that to optimize each parameter by its own state. Any momentum or energy term used in the optimization is based on each parameter's own history, independent on others. Each parameter's optimization state is not shared across other coordinates.

In our approach, the role of the optimizer is given to a Hamiltonian Neural Network which is presented in figure below:

<img src="figs/graph.png" width="600" />

## Acknowledgement
* Some parts of the code were taken from here: [chenwydj/learning-to-learn-by-gradient-descent-by-gradient](https://github.com/chenwydj/learning-to-learn-by-gradient-descent-by-gradient-descent).
* The idea of the project has been inspired by [Learning to learn by gradient descent by gradient descent](https://arxiv.org/pdf/1606.04474.pdf) and [Dissipative SymODEN](https://arxiv.org/pdf/2002.08860.pdf)
