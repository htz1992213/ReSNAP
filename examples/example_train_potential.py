from re_ml.core.train_potential import PotentialTrainer
import numpy as np
import os

OUTPUT_DIR = "/Users/th/Downloads/datafiles"

pt = PotentialTrainer(os.path.join(OUTPUT_DIR, "re.mat"), "RIDGE")
error = pt.cross_validation(np.array([0, 0.01, 0.1, 1]), plot_image=False)
