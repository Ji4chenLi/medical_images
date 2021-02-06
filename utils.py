import pickle


def is_nan(x):
    return True if x != x else False


def load_txt(text_data_path):

    with open(text_data_path, 'rb') as f:
        text_dict = pickle.load(f)

    return text_dict
