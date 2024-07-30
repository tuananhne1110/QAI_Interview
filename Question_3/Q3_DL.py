import numpy as np
import matplotlib.pyplot as plt
import pickle

# Preprocessing
class PreProc:
    def __init__(self):
        pass
    
    def one_hot_encode(self, y, levels):
        res = np.zeros((len(y), levels))
        for i in range(len(y)):
            res[i, y[i]] = 1
        return res

    def normalize(self, x):
        return x / np.max(x)

    def read_csv(self, fname):
        data = np.loadtxt(fname, skiprows=1, delimiter=',')
        y = data[:, :1]
        x = data[:, 1:]
        return x, y

    def load_data(self, fname):
        x, y = self.read_csv(fname)
        x = self.normalize(x)
        y = np.int16(y)
        y = self.one_hot_encode(y, levels=10)

        x = np.expand_dims(x, axis = -1)
        y = np.expand_dims(y, axis = -1)
        return x, y

x_train, y_train = PreProc().load_data('mnist_train.csv')
x_test, y_test = PreProc().load_data('mnist_test.csv')

# Neural Network
class NeuralNetwork:
    def __init__(self, d_in, d1, d2, d_out, lr = 1e-3):
        self.d_in = d_in
        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out
        self.lr = lr
        self.init_weights()
        
    def init_weights(self):
        self.w1 = np.random.randn(self.d1, self.d_in)
        self.b1 = np.random.randn(self.d1, 1)
        
        self.w2 = np.random.randn(self.d2, self.d1)
        self.b2 = np.random.randn(self.d2, 1)
        
        self.w3 = np.random.randn(self.d_out, self.d2)
        self.b3 = np.random.randn(self.d_out, 1)

    def relu(self, x):
        return np.maximum(x, 0)

    def drelu(self, x):
        return np.diag(1.0 * (x > 0))

    def soft_max(self, x):
        x = x - np.max(x, axis=0)
        return np.exp(x) / np.sum(np.exp(x), axis=0)
  
    def forward(self, x, y=None):
        self.x = x
        if y is not None:
            self.y = y
        
        self.z1 = np.matmul(self.w1, self.x) + self.b1
        self.a1 = np.apply_along_axis(self.relu, 1, self.z1)
        
        self.z2 = np.matmul(self.w2, self.a1) + self.b2
        self.a2 = np.apply_along_axis(self.relu, 1, self.z2)
        
        self.z3 = np.matmul(self.w3, self.a2) + self.b3
        self.out = np.apply_along_axis(self.soft_max, 1, self.z3)
        
    def transpose(self, x):
        return np.transpose(x, [0, 2, 1])

    def backward(self):
        delta = 2*self.transpose(self.out - self.y)
        self.dw3 = np.mean(
            np.matmul(self.transpose(delta), self.transpose(self.a2)),
            axis = 0
        )
        self.db3 = np.mean(self.transpose(delta), axis = 0)
        
        delta = np.matmul(
            np.matmul(delta, self.w3),
            np.squeeze(np.apply_along_axis(self.drelu, 1, self.z2))
        )
        self.dw2 = np.mean(
            np.matmul(self.transpose(delta), self.transpose(self.a1)),
            axis = 0
        )
        self.db2 = np.mean(self.transpose(delta), axis = 0)
        
        delta = np.matmul(
            np.matmul(delta, self.w2),
            np.squeeze(np.apply_along_axis(self.drelu, 1, self.z1))
        )
        self.dw1 = np.mean(
            np.matmul(self.transpose(delta), self.transpose(self.x)),
            axis = 0
        )
        self.db1 = np.mean(self.transpose(delta), axis = 0)
        
        self.w3 = self.w3 - self.lr * self.dw3
        self.b3 = self.b3 - self.lr * self.db3
        
        self.w2 = self.w2 - self.lr * self.dw2
        self.b2 = self.b2 - self.lr * self.db2
        
        self.w1 = self.w1 - self.lr * self.dw1
        self.b1 = self.b1 - self.lr * self.db1

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = np.sum((anchor - positive) ** 2, axis=-1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=-1)
    return np.maximum(pos_dist - neg_dist + margin, 0.0)

epochs = 20
batch_size = 1000
shuffle = True
lr = 1e-3

def shuffle(x, y):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    return x[idx,], y[idx,]

x_train, y_train = PreProc().load_data('mnist_train.csv')
x_test, y_test = PreProc().load_data('mnist_test.csv')

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
        x = x_train[batch*batch_size: (batch+1)*batch_size,]
        y = y_train[batch*batch_size: (batch+1)*batch_size,]
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
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(l, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(acc, label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(triplet_losses, label='Triplet Loss')
plt.xlabel('Epochs')
plt.ylabel('Triplet Loss')
plt.title('Triplet Loss')
plt.legend()

plt.tight_layout()
plt.show()
