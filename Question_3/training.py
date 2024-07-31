import numpy as np
import matplotlib.pyplot as plt
from preproc import PreProc
from neural_network import NeuralNetwork
from utils import triplet_loss, shuffle

epochs = 20
batch_size = 1000
lr = 1e-3

x_train, y_train = PreProc().load_data('./Question_3/mnist_train.csv')
x_test, y_test = PreProc().load_data('./Question_3/mnist_test.csv')

nn = NeuralNetwork(x_train.shape[1], 256, 128, y_train.shape[1], lr=lr)

l = []
acc = []
triplet_losses = []

for i in range(epochs):
    loss = 0
    accuracy = 0
    triplet_loss_sum = 0
    if shuffle:
        x_train, y_train = shuffle(x_train, y_train)
    for batch in range(x_train.shape[0] // batch_size):
        x = x_train[batch * batch_size: (batch + 1) * batch_size,]
        y = y_train[batch * batch_size: (batch + 1) * batch_size,]
        nn.forward(x, y)
        loss += np.mean((nn.out - nn.y) ** 2)
        accuracy += np.mean(np.argmax(nn.out, axis=1) == np.argmax(nn.y, axis=1))
        nn.backward()
        
        # Compute triplet loss for a random set of triplets
        anchor_idx = np.random.choice(batch_size)
        positive_idx = np.random.choice(np.where(np.argmax(y, axis=1) == np.argmax(y[anchor_idx]))[0])
        negative_idx = np.random.choice(np.where(np.argmax(y, axis=1) != np.argmax(y[anchor_idx]))[0])
        
        anchor = x[anchor_idx]
        positive = x[positive_idx]
        negative = x[negative_idx]
        
        triplet_loss_value = triplet_loss(anchor, positive, negative)
        triplet_loss_sum += np.mean(triplet_loss_value)
        
    loss = loss / (x_train.shape[0] // batch_size)
    l.append(loss)
    accuracy = accuracy / (x_train.shape[0] // batch_size)
    acc.append(accuracy)
    
    triplet_loss_mean = triplet_loss_sum / (x_train.shape[0] // batch_size)
    triplet_losses.append(triplet_loss_mean)
    
    print('Epoch {epoch}: loss = {loss}, accuracy = {accuracy}, triplet_loss = {triplet_loss}'.format(
        epoch=i, loss=loss, accuracy=accuracy, triplet_loss=triplet_loss_mean))

# Save model
nn.save_model('neural_network.pkl')

# Test model
def test_model(nn, x_test, y_test):
    nn.forward(x_test, y_test)
    loss = np.mean((nn.out - nn.y) ** 2)
    accuracy = np.mean(np.argmax(nn.out, axis=1) == np.argmax(nn.y, axis=1))
    print('Test: loss = {loss}, accuracy = {accuracy}'.format(loss=loss, accuracy=accuracy))
    return loss, accuracy

test_loss, test_accuracy = test_model(nn, x_test, y_test)

# Visualization
def visualize_predictions(nn, x_test, y_test, num_images=10):
    nn.forward(x_test)
    predictions = np.argmax(nn.out, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        plt.subplot(2, num_images // 2, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')  # Assuming MNIST images are 28x28
        plt.title(f"Pred: {predictions[i]}\nTrue: {true_labels[i]}")
        plt.axis('off')
    plt.show()

# Visualize predictions
visualize_predictions(nn, x_test, y_test)
