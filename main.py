import argparse
from runner_mem2seq import Mem2SeqRunner
from runner_splitmem import SplitMemRunner
from runner_hidden import SplitHiddenRunner
def parser():
    parser = argparse.ArgumentParser(description="End-to-End Personalized Task Oriented Dialog System with bAbI and "
                                                 "Personalized bAbI datasets.")
    parser.add_argument("--task", type=str, choices=["1", "2", "3", "4", "5"], help="bAbI task number")
    parser.add_argument("--model", required=True, choices=["mem2seq", "split_mem", "personal_context"],
                        help="The model to use")
    parser.add_argument("--data", required=True, choices=["babi", "personal", "personal_context"])
    parser.add_argument("--name", type=str, required=True, help="Identify a run")
    parser.add_argument("--log", action='store_true', default=True)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val", type=int, default=5, help="Evaluate the model in these many epochs")
    parser.add_argument("-b", type=int, default=8, dest='batch_size',)
    parser.add_argument("--cuda", action='store_true', default=False)
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--out_file", type=str, default='')

    return parser.parse_args()


if __name__ == "__main__":
    args = parser()

    if args.model == "mem2seq":
        runner = Mem2SeqRunner
    elif args.model == "split_mem":
        runner = SplitMemRunner
    elif args.model == "personal_context":
        runner = Mem2SeqRunner
    elif args.model == "hidden":
        runner = SplitHiddenRunner
    else:
        raise ModuleNotFoundError()


    runner_class = runner(args)

    runner_class.trainer()

