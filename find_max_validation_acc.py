import pickle
import argparse
def getValAcc(filename):
    with open(filename, "rb") as pkl_file:
        d = pickle.load(pkl_file)
        return max(d[1]['val']['acc'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the best validation accuracy from plot pickle file")
    parser.add_argument("pkl_file", help="Path of the pickle file")
    args = parser.parse_args()
    print(getValAcc(args.pkl_file))