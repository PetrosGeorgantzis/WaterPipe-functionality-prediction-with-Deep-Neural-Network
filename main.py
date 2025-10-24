# ---------------------------------------------------------------------------
# FILE: main.py
"""
Main script: prepares data, selects model, trains and writes submission CSV.
Usage examples:
    python main.py --model mlp
    python main.py --model rf
    python main.py --model xgb

"""
import argparse
import numpy as np
import pandas as pd

from data_builder import load_dataset, improve_values, make_dataset, normalize_data, make_predictions
import models


def labels_onehot_to_classidx(onehot):
    # expects shape (N,3)
    return np.argmax(onehot, axis=1)


def classidx_to_submission_label(idx_array):
    # 0 -> functional, 1 -> functional needs repair, 2 -> non functional
    labels = {0: 'functional', 1: 'functional needs repair', 2: 'non functional'}
    return [labels[int(i)] for i in idx_array]


def write_submission(ids, probs, out_csv='submission.csv'):
    # probs: Nx3 probability array
    idxs = np.argmax(probs, axis=1)
    labels = classidx_to_submission_label(idxs)
    df = pd.DataFrame({'id': ids, 'status_group': labels})
    df.to_csv(out_csv, index=False)
    print(f'Wrote submission to {out_csv}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'rf', 'xgb'], default='mlp')
    parser.add_argument('--train-values', default='training_values.csv')
    parser.add_argument('--train-labels', default='training_labels.csv')
    parser.add_argument('--test-values', default='test.csv')
    parser.add_argument('--out', default='submission.csv')
    args = parser.parse_args()

    print('Loading training values...')
    train_raw = load_dataset(args.train_values)
    print('Improving/cleaning training values...')
    train_improved, funder_arr, installer_arr, scheme_arr = improve_values(train_raw.copy(), None, True)

    print('Building numeric training dataset...')
    train_numbers = make_dataset(train_improved, funder_arr, installer_arr, scheme_arr)
    X_train = normalize_data(train_numbers)

    print('Loading labels...')
    y_df = load_dataset(args.train_labels)
    y_onehot = make_predictions(y_df)
    y_idx = labels_onehot_to_classidx(y_onehot)

    # Train chosen model
    print(f"Training model: {args.model}")
    if args.model == 'mlp':
        model = models.train_mlp(X_train, y_onehot, input_dim=X_train.shape[1], epochs=50)
        predict_fn = models.predict_mlp
    elif args.model == 'rf':
        model = models.train_random_forest(X_train, y_idx)
        predict_fn = models.predict_rf
    else:  # xgb
        model = models.train_xgboost(X_train, y_idx, num_round=100)
        predict_fn = models.predict_xgb

    # Prepare test data
    print('Preparing test set...')
    test_raw = load_dataset(args.test_values)
    test_improved, _, _, _ = improve_values(test_raw.copy(), train_improved, False)
    test_numbers = make_dataset(test_improved, funder_arr, installer_arr, scheme_arr)
    X_test = normalize_data(test_numbers)

    # Predict
    print('Predicting...')
    probs = predict_fn(model, X_test)

    # Save submission
    write_submission(test_raw['id'], probs, out_csv=args.out)

    # Save model
    try:
        models.save_model(model, f'model_{args.model}.joblib')
        print('Model saved.')
    except Exception as e:
        print('Could not save model with generic joblib . Reason:', e)


if __name__ == '__main__':
    main()