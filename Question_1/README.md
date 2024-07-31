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
