import os

from resnap.core.make_feature import FeatureMaker

XSF_DIR = os.path.join(os.path.dirname(__file__), "xsf_files")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_files")


def example_maker():
    fm = FeatureMaker(XSF_DIR, DATA_DIR,
                      {"rcutfac": 1.0, "rfac0": 0.94,
                       "twojmax": 10, "R_1": 1.8},
                      mode="structure", screen=False,
                      num_of_samples=10)
    fm.save_samples(DATA_DIR, "re.mat")


if __name__ == '__main__':
    example_maker()
