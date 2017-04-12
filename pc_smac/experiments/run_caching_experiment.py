import argparse
import time

from sklearn.externals import joblib

from pc_smac.pc_smac.data_loader.data_loader import DataLoader
from pc_smac.pc_smac.pipeline_space.feature_preprocessing_nodes.kernel_pca import KernelPcaNode


def run_caching_experiment(data_path):
    data_loader = DataLoader(data_path)
    data = data_loader.get_data()

    X_train = data["X_train"]
    y_train = data["y_train"]
    print(X_train.shape, y_train.shape)

    # K = 2
    # X_folds = np.array_split(X_train, K)
    # y_folds = np.array_split(y_train, K)
    #
    # X_train_1 = list(X_folds)
    # X_train_2 = X_train_1.pop(1)
    # X_train_1 = np.concatenate(X_train_1)
    # y_train_1 = list(y_folds)
    # y_train_2 = y_train_1.pop(1)
    # y_train_1 = np.concatenate(y_train_1)

    #### CACHE transformer ####
    fp_node = KernelPcaNode()
    transformer_name, transformer = fp_node.initialize_algorithm(hyperparameters={})
    print("Start")
    print(transformer_name, transformer)
    start_time = time.time()
    transformer_after = transformer.fit(X_train, y_train)
    print("Timing transform: {}".format(time.time() - start_time))
    fp_node = KernelPcaNode()
    transformer_name_2, transformer_2 = fp_node.initialize_algorithm(hyperparameters={})
    print("Start")
    print(transformer_name_2, transformer_2)
    start_time = time.time()
    transformer_after = transformer_2.fit(X_train, y_train)
    joblib.dump(transformer_after, '/Users/jorntuyls/Desktop/transform.pkl')
    print("Timing transform and persist: {}".format(time.time() - start_time))

    start_time = time.time()
    print("Start")
    transformer_after = joblib.load('/Users/jorntuyls/Desktop/transform.pkl')
    print("    Load time: {}".format(time.time() - start_time))
    X1 = transformer_after.transform(X_train)
    print("Timing load transformer and transform : {}".format(time.time() - start_time))

    #### CACHE result ####

    fp_node = KernelPcaNode()
    transformer_name_2, transformer_2 = fp_node.initialize_algorithm(hyperparameters={})
    print("Start")
    print(transformer_name_2, transformer_2)
    start_time = time.time()
    X = transformer_2.fit(X_train, y_train).transform(X_train)
    joblib.dump(X, '/Users/jorntuyls/Desktop/X.pkl')
    print("Timing transform and persist result: {}".format(time.time() - start_time))

    start_time = time.time()
    print("Start")
    X2 = joblib.load('/Users/jorntuyls/Desktop/X.pkl')
    print("Timing load result and transform : {}".format(time.time() - start_time))

    print(X1 == X2)



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--location", type=str, help="Dataset directory")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Seed for sampling configurations")
    parser.add_argument("-p", "--pname", type=str, default=None, help="Preprocessor name")
    parser.add_argument("-o", "--outputdir", type=str, default=None, help="Output directory")
    parser.add_argument("-ds", "--downsampling", type=int, default=None,
                        help="Number of data points to downsample to")
    parser.add_argument("-cd", "--cachedir", type=str, default=None, help="Cache directory")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    run_caching_experiment(data_path=args.location)

