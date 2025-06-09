from datasets import load_from_disk

portland_attrs = {
    "train_dataset": load_from_disk("../data/imdb_train_portland"),
    "test_dataset": load_from_disk("../data/imdb_test_portland"),
}
