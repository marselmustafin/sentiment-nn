from datetime import datetime as dt
import os


class Logger:
    RESULTS_DIR = os.path.abspath("") + "/results/"
    MODEL_SUMMARY_FILENAME = "summary.txt"

    def __init__(self):
        self.dir = self.RESULTS_DIR + str(dt.now()) + "/"
        self.summary_file = self.dir + self.MODEL_SUMMARY_FILENAME

        self.init_directory(self.dir)

    def pre_setup(self, preprocessor=None, manual_features=None,
                  auto_features=None):
        self.write("=== MODEL SETUP ===\n")
        self.write("preprocessing: %s" % (True if preprocessor else False))
        self.write(
            "manual features: %s" % (True if manual_features else False))
        self.write("auto_features: %s" % (True if auto_features else False))

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
        try:
            return os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

    def write(self, text):
        with open(self.summary_file, "a") as myfile:
            myfile.write(text + "\n")


# logger = Logger()
# logger.pre_setup(preprocessor="ekphrasis",
#                  manual_features=True, auto_features=True)
# print(logger.RESULTS_DIR)
