from ..re_ml.core.make_feature import FeatureMaker

XSF_DIR = "/Users/th/Downloads/xsffiles/"

DATA_DIR = '/Users/th/Downloads/datafiles/'

fm = FeatureMaker(XSF_DIR, DATA_DIR,
                  {"rcutfac": 2.0, "rfac0": 0.94, "twojmax": 10, "R_1": 0.9},
                  mode="structure", screen=False)
fm.save_samples(DATA_DIR, "re.mat")