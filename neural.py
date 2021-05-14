import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from sklearn import datasets


number_of_points =100
centers = [[-0.5, 0.5], [0.5, -0.5]]
x,y = datasets.make_blobs(n_samples = number_of_points, random_state = 123 , centers=centers, cluster_std=0.4)
#print(x)
#print(y)

number_of_points =100
centers = [[-0.5, 0.5], [0.5, -0.5]]
x,y = datasets.make_blobs(n_samples = number_of_points, random_state = 123 , centers=centers, cluster_std=0.4)
x_data=torch.Tensor(x)
y_data=torch.Tensor(y.reshape(100, 1))


plt.scatter(x[y==0, 0], x[y==0, 1])


def scatter_plot():
    plt.scatter(x[y==0, 0], x[y==0, 1])
    plt.scatter(x[y==1, 0], x[y==1, 1])

scatter_plot()

class Model(nn.Module):                                          #constructing a model using Linear class
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        pred = torch.sigmoid(self.linear(x))         #>>>>> https://www.javatpoint.com/pytorch-testing-of-perceptron-model
        return pred


torch.manual_seed(2)
model = Model(2, 1)
print(list(model.parameters()))

[w, b] = model.parameters()
w1, w2 = w.view(2)
def get_params():
    return(w1.item(), w2.item(), b[0].item())

def plot_fit(title):
    plt.title = title
    w1, w2, b1 = get_params()
    x1 = np.array([-2.0, 2.0])
    x2 = (w1 * x1 + b1)/-w2
    plt.plot(x1, x2, 'r')
    scatter_plot()


plot_fit('Initial Model')


criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


epochs =1000
losses = []
for i in range(epochs):
    y_pred = model.forward(x_data)
    loss = criterion(y_pred, y_data)
    print("epoch:", i, "loss", loss.item())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("epoch")


plt.plot("Trained Model")






