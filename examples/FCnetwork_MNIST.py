import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from mdsdl.utilities import train_test_split, MSE, dMSE_dy
from mdsdl.fully_connected import FCNetwork, FullyConnectedLayer, ActivationLayer

digits = load_digits()
print(digits.data.shape)

plt.matshow(digits.images[10], cmap='gray')


# Scale the feature value range to [0, 1]
X = digits.data / 16.

# Target variable: One-hot-encoding: map each number of the interval [0,9] to a vector with 
# 9 zeros and a one, such that, e.g.,  2 is mapped to [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
Y = digits.target
Y = np.eye(10, dtype=int)[Y]

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, fraction=0.7, seed=None)

print("shape of X_train:", X_train.shape)
print("shape of Y_train:", Y_train.shape)
X_train.shape


nn = FCNetwork(MSE, dMSE_dy)

nn.add_layer(FullyConnectedLayer(X_train.shape[-1], 80))
nn.add_layer(ActivationLayer())
nn.add_layer(FullyConnectedLayer(80, 40))
nn.add_layer(ActivationLayer())
nn.add_layer(FullyConnectedLayer(40, 10))
nn.add_layer(ActivationLayer())

train_mse = nn.train(X_train, Y_train, epochs=200, learning_rate=0.1)

fig, ax = plt.subplots(dpi=80)
ax.plot(train_mse)
ax.set(xlabel='epochs', ylabel='trianing MSE')
plt.show()


Y_pred = nn.predict(X_test)






print("true values : ")
for y in Y_test[:5]:
    print(np.argmax(y), np.array(y, dtype=float))
    
print("predicted values : ")
for yp in Y_pred[:5]:
    print(np.argmax(yp), np.array(yp, dtype=float))




confusion_matrix = np.zeros((10, 10), dtype=int)
for y, yp in zip(Y_test, Y_pred):
    pred_digit = np.argmax(yp)
    true_digit = np.argmax(y)
    confusion_matrix[true_digit, pred_digit] += 1
fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix, cmap='magma_r', origin='lower')
plt.colorbar(im)
ax.set(xlabel='predicted digit', ylabel='true digit', xticks=range(10), yticks=range(10));
plt.show()