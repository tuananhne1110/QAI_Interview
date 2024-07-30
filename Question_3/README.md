# Machine Learning and Deep Learning Approaches for MNIST Classification

This repository contains implementations of two different approaches for classifying the MNIST dataset: a logistic regression model and a neural network model. Below is a detailed explanation of each methodology, along with their advantages and disadvantages.

## Machine Learning Approach (Logistic Regression)

### Methodology

1. **Data Loading and Preprocessing**:
   - **Load Images**:
     - The `load_images` function reads images from a gzip-compressed file.
     - It reads the header to get the number of images, rows, and columns.
     - It then reads the image data and reshapes it into a 2D array where each row represents an image.
   - **Prepend Bias**:
     - The `prepend_bias` function adds a bias term (a column of ones) to the data matrix.
   - **Load Labels**:
     - The `load_labels` function reads the labels from the gzip-compressed file and reshapes them into a column vector.
   - **One-Hot Encoding**:
     - The `one_hot_encode` function converts each label into a vector with a 1 in the position corresponding to the label and 0s elsewhere.

2. **Model Definition**:
   - **Sigmoid Function**:
     - The `sigmoid` function maps the weighted sum of inputs to a value between 0 and 1, which can be interpreted as a probability.
   - **Forward Propagation**:
     - The `forward` function computes the weighted sum of inputs and applies the sigmoid function to produce the predicted probabilities.
   - **Classification**:
     - The `classify` function uses the `forward` function to compute the predictions and then selects the class with the highest probability using `np.argmax`.

3. **Training**:
   - **Loss Function**:
     - The `loss` function computes the cross-entropy loss between the predicted probabilities and the true labels.
   - **Gradient Calculation**:
     - The `gradient` function computes the gradient of the loss with respect to the weights.
   - **Training Loop**:
     - The `train` function iteratively updates the weights using gradient descent, reports the training progress periodically, and calculates the loss and accuracy.

4. **Evaluation and Visualization**:
   - **Test Accuracy**:
     - The `test` function evaluates the model on the test set and computes the accuracy.
   - **Visualization**:
     - The `visualize_predictions` function displays a subset of the test images along with the predicted and true labels.

### Advantages

- **Simplicity**: Easy to understand and implement.
- **Interpretability**: Each weight corresponds to a feature, making it easier to understand the influence of each feature on the predictions.
- **Efficiency**: Computationally efficient for small to medium-sized datasets.

### Disadvantages

- **Limited Expressiveness**: Can only model linear relationships between features and the target variable.
- **Performance**: Generally performs worse on complex datasets compared to more advanced models.
- **Feature Engineering**: Requires careful feature engineering and preprocessing to perform well.

## Deep Learning Approach (Neural Networks)

### Methodology

1. **Data Loading and Preprocessing**:
   - **Load Data**:
     - The `PreProc` class handles data loading and preprocessing.
     - It reads the data from CSV files, normalizes the feature values, and one-hot encodes the labels.
   - **Normalization**:
     - Normalizes the input features to a range of 0 to 1, which helps in speeding up the convergence of the training process.
   - **Reshaping**:
     - Expands the dimensions of the input and output data to fit the expected input format of the neural network.

2. **Model Definition**:
   - **Initialization**:
     - The `NeuralNetwork` class initializes the weights and biases for three layers (input, hidden, and output).
   - **Activation Functions**:
     - Uses ReLU (Rectified Linear Unit) activation for the hidden layers and softmax activation for the output layer.
   - **Forward Pass**:
     - The `forward` function computes the activations for each layer.
   - **Backward Pass**:
     - The `backward` function calculates the gradients for each layer using backpropagation and updates the weights and biases.

3. **Training**:
   - **Loss Calculation**:
     - The loss is computed as the mean squared error between the predicted and true labels.
   - **Gradient Descent**:
     - The weights and biases are updated using gradient descent.
   - **Triplet Loss**:
     - Adds a triplet loss to improve the model's ability to distinguish between similar and dissimilar samples.

4. **Evaluation and Visualization**:
   - **Testing**:
     - The model is evaluated on the test set to compute the loss and accuracy.
   - **Visualization**:
     - Training metrics such as loss, accuracy, and triplet loss are visualized using plots.

### Advantages

- **High Capacity**: Can model complex, non-linear relationships in data.
- **Automatic Feature Learning**: Can learn features directly from raw data, reducing the need for manual feature engineering.
- **Scalability**: Can handle large datasets and complex models with many parameters.

### Disadvantages

- **Complexity**: Requires more expertise to implement and tune hyperparameters.
- **Computational Resources**: Needs more computational power and memory.
- **Interpretability**: Often considered black-box models, making them harder to interpret.

## Detailed Comparison

### Training Process

- **Logistic Regression**: 
  - Involves calculating the loss and gradient, and updating the weights in a straightforward manner.
  - Training is generally fast and requires less computational power.
  
- **Neural Networks**: 
  - Involves multiple layers and non-linear activations, making the forward and backward passes more complex.
  - Training can be slower and requires more computational resources due to the larger number of parameters.

### Model Complexity

- **Logistic Regression**:
  - Linear model with a single layer of weights.
  - Limited in capturing non-linear relationships.

- **Neural Networks**:
  - Multi-layered with non-linear activation functions.
  - Capable of capturing complex patterns and relationships in data.

### Performance

- **Logistic Regression**:
  - Performs well on simple, linearly separable datasets.
  - May struggle with high-dimensional or complex datasets.

- **Neural Networks**:
  - Generally performs better on complex datasets with non-linear relationships.
  - Can achieve higher accuracy but requires careful tuning and more data.

### Interpretability

- **Logistic Regression**:
  - Highly interpretable as each feature's weight indicates its influence on the prediction.
  - Suitable for scenarios where model transparency is important.

- **Neural Networks**:
  - Less interpretable due to the complexity of multiple layers and weights.
  - Research is ongoing to improve interpretability, such as through techniques like SHAP (SHapley Additive exPlanations).

### Use Cases

- **Logistic Regression**:
  - Ideal for binary classification problems and scenarios requiring model interpretability.
  - Commonly used in fields like healthcare, finance, and social sciences.

- **Neural Networks**:
  - Suitable for tasks like image recognition, natural language processing, and complex pattern recognition.
  - Widely used in areas like computer vision, speech recognition, and autonomous systems.

In conclusion, the choice between logistic regression and neural networks depends on the complexity of the problem, the size of the dataset, the need for interpretability, and the available computational resources. Logistic regression is a good starting point for simple problems and quick solutions, while neural networks are preferred for more complex tasks where higher performance is required.
