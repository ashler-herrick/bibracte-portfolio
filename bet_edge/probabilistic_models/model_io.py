import os
import pickle


def pickle_model(rel_path: str, model, model_name):
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)

    full_path = os.path.join(os.path.abspath(rel_path), f"{model_name}.pkl")
    with open(full_path, "wb") as file:
        pickle.dump(model, file)

    print(f"Model pickled successfully in {full_path}")


def unpickle_model(rel_path, model_name):
    full_path = os.path.abspath(os.path.join(rel_path, f"{model_name}.pkl"))
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file {full_path} does not exist.")

    with open(full_path, "rb") as file:
        loaded_instance = pickle.load(file)
    print(f"Model loaded successfully from {full_path}")
    return loaded_instance
