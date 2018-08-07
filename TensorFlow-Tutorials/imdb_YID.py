import os
import glob
import download

data_dir = 'data/IMDB/'
data_url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'


# download IMDB datafile and extract
def download_IMDB_and_extract():
    download.maybe_download_and_extract(data_url, data_dir)



# x+ = [f(x) x in seq]
# x- = [f(x) x in seq]
# x_ = x+ + x-

# def f
def text_to_one_line(path):
    with open(path, "rt") as file:
        lines = file.readlines()
        text = " ".join(lines)

# def fn to get seq and convert to get dataset:x and labelset:y
def get_data_set(train=True):
    """
    1. get path of train or test
    2. build glob pattern, using it to get seq of files
    3. [f(x), x in seq], f  already has, build data and label set
    """

    # 1
    train_or_test = "train" if train == True else "test"
    data_path = os.path.join(data_dir, "aclImdb",train_or_test)

    # 2
    pos_glob_pattern = os.path.join(data_path, "pos", "*.txt")
    neg_glob_pattern = os.path.join(data_path, "neg", "*.txt")
    pos_file_path_seq = glob.glob(pos_glob_pattern)
    neg_file_path_seq = glob.glob(neg_glob_pattern)

    # 3
    pos_dataset = [text_to_one_line(path) for path in pos_file_path_seq]
    neg_dataset = [text_to_one_line(path) for path in neg_file_path_seq]
    x = pos_dataset + neg_dataset
    y = [1.0] * len(pos_dataset) + [0.0] * len(neg_dataset)

    return x, y

# download_IMDB_and_extract()
x, y = get_data_set()
