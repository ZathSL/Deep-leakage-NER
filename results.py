import argparse
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", help="percorso file da leggere")

    args = parser.parse_args()

    # open a file, where you stored the pickled data
    file = open(args.file, 'rb')

    # dump information to that file
    data = pickle.load(file)

    # close the file
    file.close()

    for key,value in data.items():
        print(key, value)