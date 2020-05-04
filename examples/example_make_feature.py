from re_ml.core.make_feature import FeatureMaker
import os

XSF_DIR = os.path.join(os.path.dirname(__file__), "xsf_files")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_files")

fm = FeatureMaker(XSF_DIR, DATA_DIR,
                  {"rcutfac": 2.0, "rfac0": 0.94, "twojmax": 10, "R_1": 0.9},
                  mode="structure", screen=False, num_of_samples=10)
fm.save_samples(DATA_DIR, "re.mat")