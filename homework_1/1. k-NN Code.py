# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def calculate_precision(true_labels, predicted_labels):
    true_labels = true_labels.flatten()
    predicted_labels = np.array(predicted_labels).flatten()
    correct_predictions = np.sum(true_labels == predicted_labels)
    total_instances = len(true_labels)
    accuracy = correct_predictions / total_instances
    return accuracy

def max_count_labels(labels, indices, k):
    nearest_labels = labels[indices].flatten()
    return int(np.sum(nearest_labels) > k / 2)

def compute_euclidean_distance(data_points, single_point):
    return np.sqrt(np.sum((data_points - single_point)**2, axis=1))

def scale_features_normalization(features, apply_scaling):
    if not apply_scaling:
        return features
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    return (features - min_vals) / (max_vals - min_vals)

# k-Nearest Neighbors classifier method
def k_nearest_neighbor_classifier(data, neighbors=3, scale_data=True):
    shuffled_data = shuffle(data)
    feature_count = data.shape[1] - 1
    X, y = shuffled_data[:, :feature_count], shuffled_data[:, -1:]
    scaled_X = scale_features_normalization(X, scale_data)

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

    def predict_labels(X_set, is_training=True):
        predictions = []
        for i, x in enumerate(X_set):
            if is_training and neighbors == 1:
                predictions.append(y_train[i])
            else:
                distances = compute_euclidean_distance(X_train, x)
                if is_training:
                    nearest_indices = np.argsort(distances)[1:neighbors+1]
                else:
                    nearest_indices = np.argsort(distances)[:neighbors]
                predictions.append(max_count_labels(y_train, nearest_indices, neighbors))
        return predictions

    train_predictions = predict_labels(X_train)
    test_predictions = predict_labels(X_test, is_training = False)

    train_precision = calculate_precision(y_train, train_predictions)
    test_precision = calculate_precision(y_test, test_predictions)

    return train_precision, test_precision

def plot_results(k_values, accuracies, std_devs, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, accuracies, yerr = std_devs, fmt='-D', capsize = 5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, 52)
    plt.xticks(k_values)
    plt.show()

if __name__ == "__main__":
    dataset = pd.read_csv('wdbc.csv', header=None).values

    k_range = range(1, 52, 2)
    iterations = 20

    results = {
        'normalized': {'training': [], 'testing': []},
        'non_normalized': {'training': [], 'testing': []}
    }

    for scaling in [True, False]:
        key = 'normalized' if scaling else 'non_normalized'
        for k in k_range:
            train_acc, test_acc = [], []
            for _ in range(iterations):
                train_prec, test_prec = k_nearest_neighbor_classifier(dataset, k, scaling)
                train_acc.append(train_prec)
                test_acc.append(test_prec)
            results[key]['training'].append((np.mean(train_acc), np.std(train_acc)))
            results[key]['testing'].append((np.mean(test_acc), np.std(test_acc)))


    for data_type in ['normalized', 'non_normalized']:
        for set_type in ['training', 'testing']:
            means, stds = zip(*results[data_type][set_type])
            plot_results(k_range, means, stds,
                        f'k-NN {set_type.capitalize()} Set ({data_type.capitalize()} Data)',
                        'Value of K', f'Accuracy Over {set_type.capitalize()} Data')

    for data_type in ['normalized', 'non_normalized']:
        print(f"\nAccuracies for {data_type} data:")
        for k, (train_acc, test_acc) in zip(k_range, zip(results[data_type]['training'], results[data_type]['testing'])):
            print(f"k = {k}:")
            print(f"Training accuracy: {train_acc[0]:.4f} ± {train_acc[1]:.4f}")
            print(f"Testing accuracy: {test_acc[0]:.4f} ± {test_acc[1]:.4f}")

        print(f"\nMaximum accuracies for {data_type} data:")

    print("\nMaximum testing accuracies:")
    for data_type in ['normalized', 'non_normalized']:
        test_accuracies = [acc[0] for acc in results[data_type]['testing']]
        max_test_acc = max(test_accuracies)
        max_test_k = k_range[test_accuracies.index(max_test_acc)]
        print(f"{data_type.capitalize()} data: Maximum accuracy = {max_test_acc:.4f} at k = {max_test_k}")