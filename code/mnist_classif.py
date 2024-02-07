import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from mlp import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download the training set
trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)

# Download the test set
testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)

# Create data loaders
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# subsample the training set
trainset.data = trainset.data[:2000]

my_mlp = MLP(n_in=784, n_out=10)
my_mlp = my_mlp.to(device)
losses = my_mlp.train(trainloader, num_epochs=50, device=device, verbose=True)

# Plot the loss
plt.figure()
plt.plot(losses)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('images/loss_mlp.png')
plt.close()

# choose 9 random images
idxs = np.random.randint(0, len(testset), 9)
imgs = testset.data[idxs]
# predict the class with the trained mlp
inputs = imgs.reshape(9, -1)
inputs = inputs.float()
out = my_mlp.forward(inputs.to(device))

# make 3x3 subplot with the images and their predicted class
fig, axs = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        axs[i, j].imshow(imgs[i*3+j], cmap='gray')
        axs[i, j].set_title("Predicted: {}".format(out[i*3+j].argmax().item()))
        axs[i, j].axis('off')
plt.savefig('images/predictions_mlp.png')
plt.close()

