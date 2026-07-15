import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 42 
num_classes = 10 
classes = ['frog', 'truck', 'deer', 'automobile', 'bird', 'horse', 'ship', 'cat', 'dog', 'airplane']
num_workers = -1
