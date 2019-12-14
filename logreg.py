import numpy as np
from load import load_defect_data
from sklearn.linear_model import LogisticRegression


def f1_score(clf, x, y):
    pred = clf.predict(x)
    true_positives = np.sum(np.multiply(pred, y))
    false_positives = np.sum(np.multiply(np.logical_xor(pred, y), pred))
    false_negatives = np.sum(np.multiply(np.logical_xor(pred, y), (1 - pred)))

    precision = true_positives * 1.0 / (true_positives + false_positives)
    recall = true_positives * 1.0 / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    return (
        np.round(precision, decimals=4),
        np.round(recall, decimals=4),
        np.round(f1, decimals=4)
    )


if __name__ == "__main__":
    data = load_defect_data()
    x_orig, y = data[:, :-1], data[:, -1]
    nsamples = len(x_orig)

    # Add an intercept term to x
    x_orig = np.hstack((np.ones((nsamples, 1)), x_orig))

    # x_orig = np.delete(x_orig, [2, 3, 7, 10, 12, 13], axis=1)
    x_orig = np.delete(x_orig, [5], axis=1)

    print("Removing no columns from the data")

    x = x_orig
    train_size = (nsamples * 7) // 8
    train_x, train_y = x[:train_size, :], y[:train_size]
    test_x, test_y = x[train_size:, :], y[train_size:]

    # clf = LogisticRegression()
    # clf.fit(train_x, train_y)
    # pred_y = clf.predict(test_x)

    clf = LogisticRegression(solver='newton-cg')
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)

    scores = []
    scores.append((f1_score(clf, test_x, test_y),
                   f1_score(clf, train_x, train_y)))

    for i in range(1, x_orig.shape[1]):

        print(f"Removing column {i} from data")
        # Split into training and testing sets
        x = np.delete(x_orig, i, axis=1)

        train_size = (nsamples * 7) // 8
        train_x, train_y = x[:train_size, :], y[:train_size]
        test_x, test_y = x[train_size:, :], y[train_size:]

        # clf = LogisticRegression()
        # clf.fit(train_x, train_y)
        # pred_y = clf.predict(test_x)

        clf = LogisticRegression(solver='newton-cg')
        clf.fit(train_x, train_y)
        pred_test_y = clf.predict(test_x)
        pred_y = clf.predict(train_x)

        scores.append((f1_score(clf, test_x, test_y),
                       f1_score(clf, train_x, train_y)))

    for i, score in enumerate(scores):
        test = score[0]
        train = score[1]
        print(f"{i}: Test: {test} Train: {train}")
    breakpoint()
