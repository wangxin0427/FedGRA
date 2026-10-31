from .lenet5 import LeNet5
from .resnet import CifarCNN
# 根据数据集名自动选择模型
def create_model(dataset_name, device):
    if dataset_name == "mnist":
        model = LeNet5(num_classes=10)
    elif dataset_name == "fashion_mnist":
        model = LeNet5(num_classes=10)
    elif dataset_name == "cifar10":
        model = CifarCNN(num_classes=10)
    elif dataset_name == "cifar100":
        model = CifarCNN(num_classes=100)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return model.to(device)