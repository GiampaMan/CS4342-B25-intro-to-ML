import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.mean(y == yhat)

def measureAccuracyOfPredictors(predictors, X, y):
    # Get predictions from all predictors
    all_preds = np.array([p(X) for p in predictors])

    if np.array_equal(np.unique(all_preds), [-1, 1]) or np.array_equal(np.unique(all_preds), [0, 1]):
        # Converts {0,1} to {-1,1}
        if np.array_equal(np.unique(all_preds), [0, 1]):
            all_preds = 2 * all_preds - 1
        
        # Simple majority vote 
        votes = np.sum(all_preds, axis=0)
        yhat = np.sign(votes)
        yhat[yhat == 0] = 1  
        
        # Convert back to {0,1} if y is 0/1
        if np.array_equal(np.unique(y), [0, 1]):
            yhat = (yhat + 1) // 2

    else:
        # average predictions
        mean_preds = np.mean(all_preds, axis=0)
        threshold = np.median(mean_preds)
        yhat = (mean_preds >= threshold).astype(int)

    return fPC(y, yhat)

def stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels):
    nrows, ncols = trainingFaces.shape[1], trainingFaces.shape[2]
    npix = nrows * ncols

    # Flatten faces for vectorized operations
    Xtrain = trainingFaces.reshape(trainingFaces.shape[0], -1)
    Xtest = testingFaces.reshape(testingFaces.shape[0], -1)

    # Keep track of chosen binary comparison features
    chosen_pairs = []
    training_accuracies = []
    testing_accuracies = []
    max_features = 6

    #start Header
    print(f"{'n':<6}{'trainingAccuracy':<20}{'testingAccuracy'}")

    # Stepwise greedy selection
    for m in range(max_features):
        best_acc = 0
        best_pair = None
        best_train_acc = 0

        for p1 in range(npix):
            for p2 in range(npix):
                if p1 == p2:
                    continue
                if (p1, p2) in chosen_pairs:
                    continue  # skip already used pair

                # Compute binary feature
                f_train = np.sign(Xtrain[:, p1] - Xtrain[:, p2])
                f_test = np.sign(Xtest[:, p1] - Xtest[:, p2])


                if chosen_pairs:
                    F_train = np.column_stack(
                        [np.sign(Xtrain[:, i1] - Xtrain[:, i2]) for (i1, i2) in chosen_pairs] + [f_train]
                    )
                    F_test = np.column_stack(
                        [np.sign(Xtest[:, i1] - Xtest[:, i2]) for (i1, i2) in chosen_pairs] + [f_test]
                    )
                else:
                    F_train = f_train[:, None]
                    F_test = f_test[:, None]

                # Train least-squares regression classifier
                w = np.linalg.pinv(F_train) @ trainingLabels
                yhat_train = np.sign(F_train @ w)
                yhat_test = np.sign(F_test @ w)

                acc_train = fPC(trainingLabels, yhat_train)
                acc_test = fPC(testingLabels, yhat_test)

                if acc_test > best_acc:
                    best_acc = acc_test
                    best_train_acc = acc_train
                    best_pair = (p1, p2)

        chosen_pairs.append(best_pair)
        training_accuracies.append(best_train_acc)
        testing_accuracies.append(best_acc)

        # Mask used pixels
        Xtrain[:, [best_pair[0], best_pair[1]]] = -1
        Xtest[:, [best_pair[0], best_pair[1]]] = -1

        # Print formatted progress
        print(f"{(m+2)*200:<6}{best_train_acc:<20.3f}{best_acc:.3f}")

    return chosen_pairs, training_accuracies, testing_accuracies




def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)
