import math
import os
import numpy as np
from monty.serialization import loadfn

# This file is to making four-atom tetragonal cell
# for the elastic constants calculation.


class UnitCell:

    def __init__(self, metal="Re_snap3"):
        self.metal = metal

    def write_lammps_data(self, filename):  # for orthogonal box

        ac_ratio = loadfn(os.path.join(os.path.dirname(__file__),
                                       "ac_ratio.yaml"))
        a = ac_ratio[self.metal]["a"]
        c = ac_ratio[self.metal]["c"]
        avec = [a, 0., 0.]
        bvec = [0., a * math.sqrt(3), 0.]
        cvec = [0., 0., c]
        lattice = np.array([avec, bvec, cvec])

        coord = list()
        coord.append(np.matmul(np.array([0.0, 0.0, 0.25]), lattice))
        coord.append(np.matmul(np.array([0.5, 0.5, 0.25]), lattice))
        coord.append(np.matmul(np.array([0.0, 0.333333, 0.75]), lattice))
        coord.append(np.matmul(np.array([0.5, 0.833333, 0.75]), lattice))

        f = open(filename, "w")
        f.write("LAMMPS data file\n")
        f.write("\n")
        f.write("4 atoms\n")
        f.write("1 atom types\n")

        f.write("0.00000 %.5f xlo xhi\n" % a)
        f.write("0.00000 %.5f ylo yhi\n" % (a*math.sqrt(3)))
        f.write("0.00000 %.5f zlo zhi\n\n" % c)

        f.write("Masses\n\n")
        f.write("1 186.207\n\n")

        f.write("Atoms # atomic\n\n")
        for i in range(3):
            f.write(str(i+1)+" 1 "+"%.5f   %.5f   %.5f\n" %
                    (coord[i][0], coord[i][1], coord[i][2]))
        f.write(str(4)+" 1 "+"%.5f   %.5f   %.5f\n" %
                (coord[-1][0], coord[-1][1], coord[-1][2]))
        f.close()
