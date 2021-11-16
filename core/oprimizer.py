from torch.optim import (SGD,
                         Adam, )

OPTIMIZATION_FACTORY = {
    'sgd': SGD,
    'adam': Adam
}
