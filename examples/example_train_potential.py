from re_ml.core.train_potential import PotentialTrainer
import numpy as np
import os

DATA_DIR = os.path.dirname(__file__)

pt = PotentialTrainer(os.path.join(DATA_DIR, "re.mat"), "RIDGE")
error = pt.cross_validation(np.array([0, 0.01, 0.1, 1]), plot_image=False)
pt.make_potential(DATA_DIR, alpha=0.0)
