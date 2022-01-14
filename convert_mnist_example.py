from lexnet.utils import convert

# MNIST data folder
source_folder = '/home/alex/Desktop/Data/MNIST'

# convert from idx format to csv
convert(f"{source_folder}/train-images.idx3-ubyte", f"{source_folder}/train-labels.idx1-ubyte",
        f"{source_folder}/mnist_train.csv", 60000)
convert(f"{source_folder}/t10k-images.idx3-ubyte", f"{source_folder}/t10k-labels.idx1-ubyte",
        f"{source_folder}/mnist_test.csv", 10000)