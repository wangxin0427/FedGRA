from torchvision import datasets, transforms

def load_dataset(dataset_name, root='./data'):
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = datasets.MNIST(root, train=True, download=True, transform=transform)
        test = datasets.MNIST(root, train=False, transform=transform)

    elif dataset_name == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(root, train=False, transform=transform)

    elif dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
        test = datasets.CIFAR10(root, train=False, transform=transform_test)

    elif dataset_name == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train = datasets.CIFAR100( root, train=True, download=True, transform=transform_train)
        test = datasets.CIFAR100( root, train=False, transform=transform_test)

    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    return train, test

