from argparse import ArgumentParser, Namespace
from pathlib import Path

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


def main(arguments: Namespace) -> None:
    fix_all_seeds(seed=arguments.seed)

    if arguments.train:

        criterion = LOSS_FACTORY[arguments.loss_fn]
        opt = OPTIMIZATION_FACTORY[arguments.opt_fn]
        model = Model3DV1(n_channels=arguments.n_channel, n_feature=arguments.n_feature_map)
        in_f = Path(arguments.in_dir)
        if not in_f.is_absolute():
            in_f = BASE_DIR.joinpath(in_f)

        trainer = TrainerV1(model=model,
                            criterion=criterion,
                            opt=opt,
                            device=device,
                            weight_decay=args.weight_decay,
                            lr=arguments.lr,
                            momentum=arguments.momentum)
        print(trainer)
    else:
        print("Wrong Option!")


if __name__ == '__main__':
    parser = ArgumentParser()

    # mode
    parser.add_argument("--input", dest="in_dir", help="input path", type=str)
    parser.add_argument("--train", help="training runtime", action="store_true")

    # runtime stabilization
    parser.add_argument("--seed", help="random seed", type=int, default=2021)
    parser.add_argument("--n_worker", help='number of workers', type=int, default=4)

    # hyper-parameter
    parser.add_argument("--x_size", help="image x size", type=int, default=150)
    parser.add_argument("--y_size", help="image y size", type=int, default=150)
    parser.add_argument("--z_size", help="image z size", type=int, default=150)
    parser.add_argument("--epochs", help='number of epochs', type=int, default=10)
    parser.add_argument("--n_batch", help='number of batches', type=int, default=4)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--momentum", help="momentum parameter", type=float, default=0.9)
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=5e-4)
    parser.add_argument('--loss_fn', help="loss function", type=str, default="mae", choices=list(LOSS_FACTORY.keys()))
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

    args = parser.parse_args()
    main(arguments=args)
