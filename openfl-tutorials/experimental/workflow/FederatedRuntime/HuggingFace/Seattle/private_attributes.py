from datasets import load_from_disk

seattle_attrs = {
    "train_dataset": load_from_disk("../data/imdb_train_seattle"),
    "test_dataset": load_from_disk("../data/imdb_test_seattle"),
}
