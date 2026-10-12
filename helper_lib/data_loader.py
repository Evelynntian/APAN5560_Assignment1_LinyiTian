from app.cnn_data import get_data_loaders


def load_data(batch_size=64):
    return get_data_loaders(batch_size=batch_size)