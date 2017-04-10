
import argparse
import shutil


def clean_directory(dir_name):
    shutil.rmtree(dir_name)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="Directory of cache")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    clean_directory(args.directory)

