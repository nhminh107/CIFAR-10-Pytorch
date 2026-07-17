import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 42 
num_classes = 10 
classes = ['frog', 'truck', 'deer', 'automobile', 'bird', 'horse', 'ship', 'cat', 'dog', 'airplane']
num_workers = -1
batch_size = 64

class_dict = {
    classes[0]: 0, 
    classes[1]: 1, 
    classes[2]: 2,
    classes[3]: 3, 
    classes[4]: 4, 
    classes[5]: 5,
    classes[6]: 6, 
    classes[7]: 7, 
    classes[8]: 8,
    classes[9]: 9
}
