from argparse import ArgumentParser, Namespace
from pathlib import Path

from torch.utils.data import DataLoader, random_split

# utils
from core.utils import fix_all_seeds

# loss
from core.loss import LOSS_FACTORY

# optimization
from core.oprimizer import OPTIMIZATION_FACTORY

# dataloader
from core.stone_loader import Stone

# settings
from settings import (BASE_DIR,
                      device)

# models
from core.model import Model3DV1

# trainer
from core.trainer import TrainerV1

# scheduler
from core.scheduler import SCHEDULER_FACTORY

# transformer
from core.transforms import get_transforms


def main(arguments: Namespace) -> None:
    fix_all_seeds(seed=arguments.seed)

    if arguments.train:

        criterion = LOSS_FACTORY[arguments.loss_fn]
        opt = OPTIMIZATION_FACTORY[arguments.opt_fn]
        scheduler = SCHEDULER_FACTORY[arguments.scheduler_fn]
        model = Model3DV1(n_channels=1, n_feature=arguments.n_feature_map)
        in_f = Path(arguments.in_dir)
        if not in_f.is_absolute():
            in_f = BASE_DIR.joinpath(in_f)

        label_f = Path(arguments.label_file)
        if not label_f.is_absolute():
            label_f = BASE_DIR.joinpath(label_f)

        dataset = Stone(images_dir=in_f,
                        label_xlx=label_f,
                        transformers=get_transforms(p_hor=arguments.random_h_flip,
                                                    p_ver=arguments.random_v_flip,
                                                    r_degree=arguments.random_degree_rotate,
                                                    n_channel=arguments.n_channel
                                                    ))

        train_size = int(len(dataset) * arguments.train_size)
        dev_size = len(dataset) - train_size
        train_set, dev_set = random_split(dataset, [train_size, dev_size])

        dl_train = DataLoader(train_set,
                              batch_size=arguments.batch_size,
                              shuffle=True,
                              num_workers=arguments.n_worker)

        dl_dev = DataLoader(dev_set,
                            batch_size=arguments.batch_size,
                            shuffle=False,
                            num_workers=arguments.n_worker)

        trainer = TrainerV1(model=model,
                            criterion=criterion,
                            opt=opt,
                            scheduler=scheduler,
                            device=device,
                            weight_decay=args.weight_decay,
                            lr=arguments.lr,
                            momentum=arguments.momentum)
        trainer.train(train_ld=dl_train, dev_ld=dl_dev, epochs=arguments.epochs)
    else:
        print("Wrong Option!")


if __name__ == '__main__':
    parser = ArgumentParser()

    # mode
    parser.add_argument("--input", dest="in_dir", help="input path", type=str)
    parser.add_argument("--label_file", help="label excel file", type=str)
    parser.add_argument("--train", help="training runtime", action="store_true")

    # runtime stabilization
    parser.add_argument("--seed", help="random seed", type=int, default=2021)
    parser.add_argument("--n_worker", help='number of workers', type=int, default=4)

    # hyper-parameter
    parser.add_argument("--x_size", help="image x size", type=int, default=150)
    parser.add_argument("--y_size", help="image y size", type=int, default=150)
    parser.add_argument("--z_size", help="image z size", type=int, default=150)
    parser.add_argument("--epochs", help='number of epochs', type=int, default=10)
    parser.add_argument("--batch_size", help='number of batches', type=int, default=4)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--momentum", help="momentum parameter", type=float, default=0.9)
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=5e-4)
    parser.add_argument('--loss_fn', help="loss function", type=str, default="mae",
                        choices=list(LOSS_FACTORY.keys()))
    parser.add_argument("--opt_fn", help="optimization function", type=str, default="sgd",
                        choices=list(OPTIMIZATION_FACTORY.keys()))
    parser.add_argument("--scheduler_fn", help="scheduler function", type=str, default="step_lr",
                        choices=list(SCHEDULER_FACTORY.keys()))

    # data augmentation parameters
    parser.add_argument("--random_h_flip", help="random horizontal flip probability", type=float, default=.5)
    parser.add_argument("--random_v_flip", help="random vertical flip probability", type=float, default=.5)
    parser.add_argument("--random_degree_rotate", help="random rotate degrees", type=float, default=10.)
    parser.add_argument("--n_channel", help="number of channels", type=int, default=150)
    parser.add_argument("--n_feature_map", help="number of feature map", type=int, default=32)
    parser.add_argument("--train_size", help="train set size", type=float, default=.9)
    parser.add_argument("--dev_size", help="dev set size", type=float, default=.1)

    args = parser.parse_args()
    main(arguments=args)
