# Neural Network Backpropagation Visualizer

A small, framework-free web application that visually explains backpropagation through a tiny fully connected neural network (2 inputs → 2 hidden neurons → 1 output).

Everything is implemented with plain HTML, CSS, and JavaScript.

## Features

- Interactive sliders for inputs, target value, and learning rate.
- Step-by-step walkthrough of:
	1. Forward pass (pre-activations and activations).
	2. Loss computation.
	3. Gradients at the output layer.
	4. Gradients for the hidden layer and input weights.
	5. Weight updates via gradient descent and loss comparison.
- SVG-based visualization of the network with animated highlighting of the active part of the computation graph.
- Compact math panel showing equations and the exact numeric values for each step.

## How to Run

From the project root, you can serve the static files with any simple HTTP server. For example, using Python:

```bash
cd /workspaces/machine_learning_final
python -m http.server 8000
```

Then open in your browser:

```bash
http://localhost:8000/
```

Move the sliders, click through the steps, and watch how every quantity in the network changes.