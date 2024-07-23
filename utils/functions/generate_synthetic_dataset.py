import pandas as pd
import numpy as np

def generate_synthetic_dataset(target_col='CAUSABAS_encoded', n_samples=50000, rand_state=76669641):
    np.random.seed(rand_state)
    randint = np.random.randint
    proportions = [70.607127, 17.897052, 3.827541, 2.630884, 2.432908, 1.592609, 1.011879]
    proportions = np.array(proportions) / 100
    class_counts = (proportions * n_samples).astype(int) + 1
    X = np.concatenate([
        randint(0, 100, size=(n_samples, 1)), randint(0, 8, size=(n_samples, 1)),
        randint(0, 3, size=(n_samples, 1)), randint(0, 2, size=(n_samples, 1)),
        randint(0, 6, size=(n_samples, 1)), randint(0, 6, size=(n_samples, 1))], axis=1)
    y = np.concatenate([
        np.full(class_counts[0], 1), np.full(class_counts[1], 3), np.full(class_counts[2], 0),
        np.full(class_counts[3], 5), np.full(class_counts[4], 2), np.full(class_counts[5], 4),
        np.full(class_counts[6], 6)], axis=0)

    # Ensure y values are integers and replace invalid values with 1
    y = np.array(y, dtype=int)
    valid_classes = {0, 1, 2, 3, 4, 5, 6}
    y = np.where(np.isin(y, list(valid_classes)), y, 1)

    # Shuffle the dataset
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = pd.DataFrame(X[indices], columns=[f'feat_{i+1}' for i in range(X.shape[1])])
    y = pd.Series(y[indices], name=target_col)

    return pd.concat([X, y], axis=1)