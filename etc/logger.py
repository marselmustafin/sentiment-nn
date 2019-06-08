from datetime import datetime as dt
import os
import pickle


class Logger:
    RESULTS_DIR = os.path.abspath("") + "/results/"
    SUMMARY_FILENAME = "summary.txt"
    HISTORY_FILENAME = "history{training}.txt"

    def __init__(self, results_dir=RESULTS_DIR):
        self.dir = results_dir + "-".join(str(dt.now()).split()) + "/"
        self.summary_file = self.dir + self.SUMMARY_FILENAME

        self.history_counter = 0
        self.history_file = self.dir + self.HISTORY_FILENAME

        self.init_directory(self.dir)
        self.write("=== MODEL SETUP ===\n")

    def setup(self, ternary=False, embeddings=False, train_set=None,
              test_set=None, vocab_size=0, earlystop=None, extra_train=None,
              epochs=0, batch_size=0, dropout=0):
        self.write("classification: %sry" % ("terna" if ternary else "bina"))
        self.write("Twitter embeddings: %s" % (True if embeddings else False))
        self.write("Train set size: %s" % len(train_set))
        self.write("Test set size: %s" % len(test_set))
        self.write("Vocabulary size: %s" % vocab_size)
        self.write("earlystop | monitor: {0}, min_delta: {1}, patience: {2}"
                   .format(earlystop.monitor, earlystop.min_delta,
                           earlystop.patience))
        self.write("epochs: %s" % epochs)
        self.write("batch_size: %s" % batch_size)
        self.write("dropout: %s" % dropout)
        self.write("extra train (ynacc): %s" % (extra_train or False))
        self.write("====================\n")

    def init_directory(self, path):
        os.makedirs(path)

    def write(self, text):
        with open(self.summary_file, "a") as myfile:
            myfile.write(text + "\n")

    def write_history(self, history):
        with open(self.history_file.format(training=self.history_counter),
                  'wb') as history_file:
            pickle.dump(history.history, history_file)
        self.history_counter += 1
