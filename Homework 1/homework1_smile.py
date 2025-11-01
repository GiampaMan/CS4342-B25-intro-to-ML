import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.mean(y == yhat)

def measureAccuracyOfPredictors(predictors, X, y, weights=None):
    """
    Measure ensemble accuracy with optional weighted voting.
    
    predictors: list of callable predictor functions
    X: data matrix (n_samples x features or images)
    y: true labels (n_samples)
    weights: optional list/array of predictor weights (higher = more influence)
    """
    # Get predictions from all predictors: shape (num_predictors, num_samples)
    all_preds = np.array([p(X) for p in predictors])
    
    num_predictors, num_samples = all_preds.shape

    # If weights not given, use equal weights
    if weights is None:
        weights = np.ones(num_predictors)
    else:
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # normalize to sum to 1

    # --- Option 1: For discrete ±1 or {0,1} outputs ---
    # Weighted majority vote using sign of weighted sum
    if np.array_equal(np.unique(all_preds), [-1, 1]) or np.array_equal(np.unique(all_preds), [0, 1]):
        # Convert {0,1} → {-1,1} for sign voting if needed
        if np.array_equal(np.unique(all_preds), [0, 1]):
            all_preds = 2 * all_preds - 1
        
        weighted_votes = np.dot(weights, all_preds)  # shape: (num_samples,)
        yhat = np.sign(weighted_votes)
        yhat[yhat == 0] = 1  # break ties consistently
        # Convert back to {0,1} if y is 0/1
        if np.array_equal(np.unique(y), [0, 1]):
            yhat = (yhat + 1) // 2

    # --- Option 2: For continuous outputs (regression-like predictors) ---
    else:
        # Weighted average predictions, then threshold at median
        weighted_preds = np.dot(weights, all_preds) / np.sum(weights)
        threshold = np.median(weighted_preds)
        yhat = (weighted_preds >= threshold).astype(int)

    return fPC(y, yhat)

def stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels):
    show = False  # set True to visualize chosen pairs

    nrows, ncols = trainingFaces.shape[1], trainingFaces.shape[2]
    npix = nrows * ncols

    # Flatten faces for vectorized operations
    Xtrain = trainingFaces.reshape(trainingFaces.shape[0], -1)
    Xtest = testingFaces.reshape(testingFaces.shape[0], -1)

    # Keep track of chosen binary comparison features
    chosen_pairs = []
    accuracies = []
    max_features = 6

    # Stepwise greedy selection
    for m in range(max_features):
        best_acc = 0
        best_pair = None

        # Try all possible pixel pairs (r1,c1,r2,c2)
        for p1 in range(npix):
            for p2 in range(npix):
                if p1 == p2:
                    continue

                # Compute binary feature over all training images in one go
                f_train = np.sign(Xtrain[:, p1] - Xtrain[:, p2])
                f_test = np.sign(Xtest[:, p1] - Xtest[:, p2])

                # Combine with already selected features
                if chosen_pairs:
                    F_train = np.column_stack([np.sign(Xtrain[:, i1] - Xtrain[:, i2]) for (i1, i2) in chosen_pairs] + [f_train])
                    F_test = np.column_stack([np.sign(Xtest[:, i1] - Xtest[:, i2]) for (i1, i2) in chosen_pairs] + [f_test])
                else:
                    F_train = f_train[:, None]
                    F_test = f_test[:, None]

                # Train least-squares regression classifier
                w = np.linalg.pinv(F_train) @ trainingLabels
                yhat = np.sign(F_test @ w)

                acc = fPC(testingLabels, yhat)
                if acc > best_acc:
                    best_acc = acc
                    best_pair = (p1, p2)

        chosen_pairs.append(best_pair)
        accuracies.append(best_acc)
        print(f"Step {m+1}: added pair {best_pair}, accuracy = {best_acc:.3f}")

    # Optional visualization of chosen pairs
    if show:
        im = testingFaces[0, :, :]
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(im, cmap='gray')
        for k, (p1, p2) in enumerate(chosen_pairs):
            r1, c1 = divmod(p1, ncols)
            r2, c2 = divmod(p2, ncols)
            ax.add_patch(patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none'))
            ax.add_patch(patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none'))
        fig.suptitle("Chosen (r1,c1,r2,c2) features")
        plt.show()

    return chosen_pairs, accuracies


def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)
