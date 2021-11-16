from torchvision.transforms import (ToTensor,
                                    Compose,
                                    RandomHorizontalFlip,
                                    RandomVerticalFlip,
                                    RandomRotation,
                                    Normalize)


def get_transforms(p_hor=.5, p_ver=.5, r_degree=10, mean=.5, std=.5, n_channel=300):
    return Compose([
        ToTensor(),
        RandomHorizontalFlip(p=p_hor),
        RandomVerticalFlip(p=p_ver),
        RandomRotation(degrees=r_degree),
        Normalize(mean=[mean] * n_channel, std=[std] * n_channel)
    ])
