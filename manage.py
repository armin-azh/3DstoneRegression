from argparse import ArgumentParser, Namespace

# utils
from core.utils import fix_all_seeds


def main(arguments: Namespace) -> None:
    fix_all_seeds(seed=arguments.seed)

    if arguments.train:
        pass

    else:
        print("Wrong Option!")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train", help="training runtime", action="store_true")

    parser.add_argument("--seed", help="random seed", type=int, default=2021)
    parser.add_argument("--random_h_flip", help="random horizontal flip probability", type=float, default=.5)
    parser.add_argument("--random_v_flip", help="random vertical flip probability", type=float, default=.5)
    parser.add_argument("--random_degree_rotate", help="random rotate degrees", type=float, default=10.)
    parser.add_argument("--n_channel", help="number of channels", type=int, default=300)

    args = parser.parse_args()
    main(arguments=args)
