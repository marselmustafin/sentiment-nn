from datetime import datetime as dt
import os


class Logger:
    RESULTS_DIR = os.path.abspath("") + "/results/"
    MODEL_SUMMARY_FILENAME = "summary.txt"

    def __init__(self):
        self.dir = self.RESULTS_DIR + "-".join(str(dt.now()).split()) + "/"
        self.summary_file = self.dir + self.MODEL_SUMMARY_FILENAME

        self.init_directory(self.dir)
        self.write("=== MODEL SETUP ===\n")

    def setup(self, ternary=False, embeddings=False, train_set=None,
              test_set=None, vocab_size=0, earlystop=None,
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
        self.write("====================\n")

    def init_directory(self, path):
        command = "mkdir -p %s" % path
        return os.system(command)

    def write(self, text):
        with open(self.summary_file, "a") as myfile:
            myfile.write(text + "\n")
