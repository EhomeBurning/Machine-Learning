import numpy as np
import pandas as pd


def train(df_train):
    X = df_train.iloc[:, :-1].as_matrix()
    y = df_train.iloc[:, 0].as_matrix()
    tokenlist = df_train.columns
    classifier = NaiveBayes(X, y, tokenlist)

    return classifier


class NaiveBayes:
    def __init__(self, X, y, tokenlist):
        num_train_docs = y.shape[0]
        word_count = X.shape[1]
        absV    = word_count
        denom_0 = np.sum(X[y == 0], axis=(0,1)) + absV
        denom_1 = np.sum(X[y == 1], axis=(0,1)) + absV

        dfp = np.zeros((2, word_count))
        # Compute the prior probability that a training email is spam.
        prior_0 = np.sum(y == 0) / num_train_docs
        prior_1 = np.sum(y == 1) / num_train_docs

        # Compute the conditional probabilities.
        for k in range(word_count):
            numer_k_0 = np.sum(X[y == 0][:, k]) + 1
            numer_k_1 = np.sum(X[y == 1][:, k]) + 1
            phi_k_0 = numer_k_0 / denom_0
            phi_k_1 = numer_k_1 / denom_1
            dfp[0, k] = phi_k_0
            dfp[1, k] = phi_k_1

        self.prior_0 = prior_0
        self.prior_1 = prior_1
        self.dfp = dfp
        self.tokenlist = tokenlist


    def classify(self, x):
        """
        Given a preprocessed email, classify it as SPAM or NOT SPAM.
        """
        log_prob_y_equals_0 = np.log(self.prior_0)
        log_prob_y_equals_1 = np.log(self.prior_1)

        probs_x_given_0 = self.dfp[0, np.where(x != 0)]
        log_probs_x_given_0 = np.log(probs_x_given_0.reshape(probs_x_given_0.shape[1]))
        probs_x_given_1 = self.dfp[1, np.where(x != 0)]
        log_probs_x_given_1 = np.log(probs_x_given_1.reshape(probs_x_given_1.shape[1]))
        
        log_probs_0 = log_prob_y_equals_0 + x[x != 0]*np.log(probs_x_given_0)
        log_probs_1 = log_prob_y_equals_1 + x[x != 0]*np.log(probs_x_given_1)

        posterior_0 = np.sum(log_probs_0)
        posterior_1 = np.sum(log_probs_1)
        
        return np.argmax((posterior_0, posterior_1))


    def to_dataframe(self):
        df = pd.DataFrame(np.zeros((self.dfp.shape[0], self.dfp.shape[1]+1)), columns=self.tokenlist)
        df.iloc[0,0] = self.prior_0
        df.iloc[1,0] = self.prior_1
        df.iloc[:, 1:] = self.dfp

        return df

    def __str__(self):
        return str(self.to_dataframe())




def test(model, df_test):
    num_test_docs = df_test.shape[0]
    output = np.zeros(num_test_docs)
    X = df_test.iloc[:, 1:].as_matrix()

    for i in range(num_test_docs):
        output[i] = model.classify(X[i])

    return output


def compute_error(y, guessed_y):
    num_test_docs = y.shape[0]
    return float(np.sum(1*(guessed_y != y))) / float(num_test_docs)


def k_most_indicative_words(k, dfp):
    log_dfp = np.log(dfp.as_matrix())
    diff_log_dfp = log_dfp[1] - log_dfp[0]
    sorted_frame = np.argsort(diff_log_dfp)[::-1]
    words = dfp.columns[sorted_frame[:k]]

    return list(words)