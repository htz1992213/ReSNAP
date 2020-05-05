import numpy as np
from math import *
import os
import time
import sys
import argparse
from monty.serialization import loadfn


class LatticeConstant:

    def __init__(self, directory, metal="Re", timer=5, job="run", amin=-0.004,
                 amax=0.005, cmin=-0.02, cmax=-0.01, step=0.002):
        valid = {"run", "energy"}
        if job not in valid:
            raise ValueError("Job type must be of of %r." % valid)
        self.directory = directory
        self.metal = metal
        self.timer = timer
        self.job = job
        metal_info = loadfn(os.path.join(os.path.dirname(__file__),
                                         "ac_ratio.yaml"))
        self.a0 = metal_info[self.metal]["a"]
        self.c0 = metal_info[self.metal]["c"]
        self.mass = metal_info[self.metal]["mass"]
        self.a = np.arange(self.a0 + amin, self.a0 + amax, step)
        self.c = np.arange(self.c0 + cmin, self.c0 + cmax, step)

    def do_job(self):
        os.chdir(self.directory)
        if self.job == "run":
            self.run()

        elif self.job == "energy":
            self.get_energy()

    @staticmethod
    def get_lat(ai, ci):
        v1 = np.array([ai, 0., 0.])
        v2 = np.array([-ai/2., sqrt(3) * ai / 2., 0.])
        v3 = np.array([0., 0., ci])
        lattice = [v1, v2, v3]
        return lattice

    @staticmethod
    def get_cart(lat):
        pd1 = np.array([1./3, 2./3, 0.25])
        pd2 = np.array([2./3, 1./3, 0.75])
        pc1 = np.dot(pd1, lat)
        pc2 = np.dot(pd2, lat)
        cart = [pc1, pc2]
        return cart

    def write_dat(self, i, j):
        lat = self.get_lat(self.a[i], self.c[j])
        cart = self.get_cart(lat)

        f = open(self.metal + ".dat_a%d_c%d" % (i, j), "w")
        f.write("LAMMPS data file for lattice constants test\n")
        f.write("\n")
        f.write("2 atoms\n")
        f.write("1 atom types\n")
        f.write("\n")
        f.write("0. %.5f xlo xhi\n" % (lat[0][0]))
        f.write("0. %.5f ylo yhi\n" % (lat[1][1]))
        f.write("0. %.5f zlo zhi\n" % (lat[2][2]))
        f.write("%.5f 0. 0. xy xz yz\n" % (lat[1][0]))
        f.write("\n")
        f.write("Masses\n")
        f.write("\n")
        f.write("1 "+str(self.mass)+"  unified atomic mass units\n")
        f.write("\n")
        f.write("Atoms\n")
        f.write("\n")
        f.write("1 1 "+"%.5f   %.5f   %.5f\n" %
                (cart[0][0], cart[0][1], cart[0][2]))
        f.write("2 1 "+"%.5f   %.5f   %.5f" %
                (cart[1][0], cart[1][1], cart[1][2]))
        f.close()

    @staticmethod
    def rewrite_submit(i, j):  # create a new submission script
        data = ''
        with open('submit', 'r+') as f:
            for line in f.readlines():
                if line.find('mpirun') > -1:
                    line = 'mpirun lmp_knl  < input.lmp_a%d_c%d > out.lmp_a%d_c%.d\n' % (i, j, i, j)
                data += line
        with open('submit_a%d_c%d' % (i, j), 'w') as f:
            f.writelines(data)

    def rewrite_input(self, i, j):    # cerate a new input script
        data = ''
        with open('input.lmp', 'r+') as f:
            for line in f.readlines():
                if line.find('read_data') > -1:
                    line = '  read_data        ' + self.metal + \
                           '.dat_a%d_c%d\n' % (i, j)
                if line.find('dump') > -1:
                    line = '  dump              d1 all custom 100 coordinates.dump_a%d_c%d id type x y z\n' % (i, j)
                data += line
        with open('input.lmp_a%d_c%d' % (i, j), 'w') as f:
            f.writelines(data)

    def run(self):
        for i in range(len(self.a)):
            for j in range(len(self.c)):
                self.write_dat(i, j)
                self.rewrite_submit(i, j)
                self.rewrite_input(i, j)
                time.sleep(self.timer)
                os.system('sbatch submit_a%d_c%d' % (i, j))

    @staticmethod
    def minmatrix(mat):  # square matrix
        m = 0
        n = 0
        mmin = mat[0][0]
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if mat[i][j] < mmin:
                    mmin = mat[i][j]
                    m = i
                    n = j
        return mmin, [m, n]

    def get_energy(self):  # get energies in eV
        energy = []
        for i in range(len(self.a)):
            ea = []
            for j in range(len(self.c)):
                f0 = open("out.lmp_a%d_c%d" % (i, j), "r")
                data = f0.readlines()
                f0.close()
                for k in range(len(data)):
                    if data[k].find("Loop time") > -1:
                        ea += [float(data[k-1].split()[1])]
            energy += [ea]
        for i in range(len(energy)):
            print(energy[i])
        emin, indexmin = self.minmatrix(energy)
        print("a = ", self.a[indexmin[0]])
        print("c = ", self.c[indexmin[1]])
        print("min Energy = ", emin, "eV")

        f = open("strain-energy", "w")
        for i in range(len(self.a)):
            for j in range(len(self.c)):
                f.write("a= %.4f c=%.4f Energy= %.6f\n" % (self.a[i], self.c[j],
                                                           energy[i][j]))
        f.close()


def main(args):
    parser = argparse.ArgumentParser()

    # -d DIRECTORY -m METAL -j {run,energy} -t TIMER -amin AMIN -amax AMAX
    # -cmin CMIN -cmax CMAX -step STEP
    parser.add_argument("-d", "--directory", help="Working directory",
                        type=str, default=os.getcwd())
    parser.add_argument("-m", "--metal", help="Metal type",
                        type=str, default="Re")
    parser.add_argument("-j", "--job", help="Job type",
                        choices=['run', 'energy'], default="run")
    parser.add_argument("-t", "--timer", help="Job submission interval",
                        type=int, default=5)
    parser.add_argument("-amin", "--amin", help="a min",
                        type=float, default=-0.004)
    parser.add_argument("-amax", "--amax", help="a max",
                        type=float, default=0.005)
    parser.add_argument("-cmin", "--cmin", help="c min",
                        type=float, default=-0.02)
    parser.add_argument("-cmax", "--cmax", help="c max",
                        type=float, default=-0.01)
    parser.add_argument("-step", "--step", help="step",
                        type=float, default=0.002)

    args = parser.parse_args(args)

    print("Working dir: ", args.directory)
    print("Metal: ", args.metal)
    print("Job type: ", args.job)
    print("Timer: ", args.timer)

    job_instance = LatticeConstant(args.directory,
                                   metal=args.metal,
                                   job=args.job,
                                   timer=args.timer,
                                   amin=args.amin,
                                   amax=args.amax,
                                   cmin=args.cmin,
                                   cmax=args.cmax,
                                   step=args.step)
    print("a range: ", job_instance.a)
    print("c range: ", job_instance.c)
    job_instance.do_job()
    print("Job done.")


if __name__ == '__main__':
    main(sys.argv[1:])
