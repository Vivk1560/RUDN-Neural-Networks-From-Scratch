from pathlib import Path
import urllib.request
from zipfile import ZipFile


DATASET_DIR = Path("datasets")

TRAIN_CSV_PATH = DATASET_DIR / "mnist_train.csv"
TEST_CSV_PATH = DATASET_DIR / "mnist_test.csv"

TRAIN_ZIP_PATH = DATASET_DIR / "mnist_train.csv.zip"
TEST_ZIP_PATH = DATASET_DIR / "mnist_test.csv.zip"

TRAIN_URL = (
    "https://github.com/nederjair/mnist_dataset/"
    "raw/refs/heads/main/mnist_train.zip"
)

TEST_URL = (
    "https://github.com/nederjair/mnist_dataset/"
    "raw/refs/heads/main/mnist_test.zip"
)


def load_mnist():
    """
    Download, extract, and load the MNIST dataset.

    The dataset is downloaded only when the corresponding CSV
    files are not already available locally.

    Returns:
        tuple: Training and testing data as lists of CSV records.
    """

    # Create the dataset directory if it does not exist
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Download training dataset if it is not available
    if not TRAIN_CSV_PATH.is_file():
        urllib.request.urlretrieve(TRAIN_URL, TRAIN_ZIP_PATH)

        with ZipFile(TRAIN_ZIP_PATH, "r") as zip_file:
            zip_file.extractall(path=DATASET_DIR)

    # Download testing dataset if it is not available
    if not TEST_CSV_PATH.is_file():
        urllib.request.urlretrieve(TEST_URL, TEST_ZIP_PATH)

        with ZipFile(TEST_ZIP_PATH, "r") as zip_file:
            zip_file.extractall(path=DATASET_DIR)

    # Load training data
    with open(TRAIN_CSV_PATH, "r") as file:
        train_data_list = file.readlines()

    # Load testing data
    with open(TEST_CSV_PATH, "r") as file:
        test_data_list = file.readlines()

    return train_data_list, test_data_list
