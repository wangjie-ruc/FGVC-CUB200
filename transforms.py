from torchvision import transforms

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
normalize = transforms.Normalize(mean=mean, std=std)


train_transform = transforms.Compose([
    transforms.Resize([448, 448]),
    # transforms.RandomCrop(448),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    normalize,
])


eval_transform = transforms.Compose([
    transforms.Resize([448, 448]),
    # transforms.CenterCrop(448),
    transforms.ToTensor(),
    normalize,
])
