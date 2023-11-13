import argparse
import sys
import time
import warnings

from tqdm import TqdmExperimentalWarning

from tqn.plot import plot
from tqn.train import train
from tqn.utils.evaluate_policy import play

if __name__ == "__main__":
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning)

    parser = argparse.ArgumentParser(prog="Transformer Q-Network")

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true")
    action.add_argument("--play", action="store_true")
    action.add_argument("--plot", action="store_true")

    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=int(time.time()))

    if "--train" in sys.argv:
        parser.add_argument("--network", type=str, default="TQN",
                            choices=["DQN", "TQN"])

    if "--play" in sys.argv:
        parser.add_argument("--model", type=str, required=True)
    if "--plot" in sys.argv:
        parser.add_argument("--plot-type", type=str, choices=[
            "environments",
            "q-vs-frame",
            "q-vs-loss",
            "q-vs-reward",
            "sample-efficiency-probability-improvement",
            "score-vs-episode",
            "statistics"
        ], required=True)

    args = parser.parse_args()

    if args.train:
        train(args.env_id, args.seed, args.network)
    elif args.play:
        play(args.env_id, args.model, args.seed)
    elif args.plot:
        plot(args.env_id, args.plot_type)
