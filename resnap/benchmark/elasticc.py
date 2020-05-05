from resnap.benchmark.structure import Structure
from pymatgen.io.lammps.outputs import parse_lammps_log
import numpy as np
import argparse
import os
import sys
import shutil
import time
import math


class ElasticJob:

    def __init__(self, directory, job="make", timer=5, etamin=-0.008,
                 etamax=0.009, etastep=0.002):
        valid = {"make", "makerun", "getEF"}
        if job not in valid:
            raise ValueError("Job type must be of of %r." % valid)
        self.eta = np.arange(etamin, etamax, etastep)
        self.f_tensor = self.get_f_tensor()
        self.timer = timer
        self.job = job
        self.directory = directory

    def do_job(self):
        os.chdir(self.directory)
        if self.job == "getEF":
            self.get_data()
        elif self.job == 'make' or self.job == 'makenrun':
            for i in range(7):
                for j in range(len(self.eta)):
                    strdir = "s" + str(i + 1) + "_%.2f" % (self.eta[j] * 100)
                    os.mkdir(strdir)  # Make directory si_j
                    # copy relavent files to new directory
                    shutil.copy("Reunit.dat", strdir)
                    shutil.copy("Re_3.snapcoeff", strdir)
                    shutil.copy("Re_3.snapparam", strdir)
                    shutil.copy("input.lmp", strdir)
                    shutil.copy("submit", strdir)

                    os.chdir(strdir)

                    cell = Structure("LAMMPSdat", "Reunit.dat")
                    cell.addstrain(self.f_tensor[i][j])
                    cell.write_lammps_data("Reunit.dat", "Re")

                    # submit the job
                    if self.job == 'makenrun':
                        time.sleep(self.timer)
                        os.system('sbatch submit')

                    os.chdir('..')

    def get_data(self):
        energy = []
        for m in range(7):
            energy_i = []
            stress = []
            for n in range(len(self.eta)):
                strdir = "s"+str(m+1)+"_%.2f" % (self.eta[n]*100)
                os.chdir(strdir)
                df = parse_lammps_log('log.lammps')
                energy_ij = float(df[0].iloc[-1, :][["TotEng"]])
                pxx = float(df[0].iloc[-1, :][["Pxx"]])
                pyy = float(df[0].iloc[-1, :][["Pyy"]])
                pzz = float(df[0].iloc[-1, :][["Pzz"]])
                pxy = float(df[0].iloc[-1, :][["Pxy"]])
                pxz = float(df[0].iloc[-1, :][["Pxz"]])
                pyz = float(df[0].iloc[-1, :][["Pyz"]])
                energy_i.append(energy_ij)
                stress.append([pxx, pyy, pzz, pxy, pxz, pyz])
                os.chdir('..')

            self.write_stress(m, stress)
            energy.append(energy_i)
        self.write_energy(energy)

    def write_stress(self, m, stress):
        fs = open("stress_" + str(m + 1), "w")
        fs.write("eta range: ")
        for j in range(len(self.eta)):
            fs.write("%.5f    " % self.eta[j])
        fs.write("\n")
        fs.write("pxx    pyy    pzz    pxy    pxz    pyz\n")

        for j in range(len(self.eta)):
            for k in range(6):
                fs.write("%.5f    " % stress[j][k])
            fs.write("\n")
        fs.close()

    def write_energy(self, energy):
        fe = open("energy_data", "w")
        fe.write("eta range: \n")
        for m in range(len(self.eta)):
            fe.write("%.5f    " % self.eta[m])
        fe.write("\n")
        fe.write("energy (eV): \n")
        for m in range(len(energy)):
            for n in range(len(self.eta)):
                fe.write("%.5f    " % energy[m][n])
            fe.write("\n")
        fe.close()

    def get_f_tensor(self):

        f = []

        f1 = []
        for i in range(len(self.eta)):
            fi = np.eye(3)
            fi[0][0] = math.sqrt(2*self.eta[i]+1)
            f1 += [fi]
        f += [f1]

        f2 = []
        for i in range(len(self.eta)):
            fi = np.eye(3)
            fi[0][0] = math.sqrt(2*self.eta[i]+1)
            fi[1][1] = math.sqrt(2*self.eta[i]+1)
            f2 += [fi]
        f += [f2]

        f3 = []
        for i in range(len(self.eta)):
            fi = np.eye(3)
            fi[2][2] = math.sqrt(2*self.eta[i]+1)
            f3 += [fi]
        f += [f3]

        f4 = []
        for i in range(len(self.eta)):
            fi = np.eye(3)
            fi[1][1] = math.sqrt(2*self.eta[i]+1)
            fi[2][2] = math.sqrt(2*self.eta[i]+1)
            f4 += [fi]
        f += [f4]

        f5 = []
        for i in range(len(self.eta)):
            fi = np.eye(3)
            fi[2][2] = math.sqrt(1-4*(self.eta[i]**2))
            fi[1][2] = 2*self.eta[i]
            f5 += [fi]
        f += [f5]

        f6 = []
        for i in range(len(self.eta)):
            fi = np.eye(3)
            fi[2][2] = math.sqrt(1-4*(self.eta[i]**2))
            fi[0][2] = 2*self.eta[i]
            f6 += [fi]
        f += [f6]

        f7 = []
        for i in range(len(self.eta)):
            fi = np.eye(3)
            fi[1][1] = math.sqrt(1-4*(self.eta[i]**2))
            fi[0][1] = 2*self.eta[i]
            f7 += [fi]
        f += [f7]

        return f


def main(args):
    parser = argparse.ArgumentParser()

    # -d DIRECTORY -j {make,makerun,getEF} -t TIMER -min STAMIN -max ETAMAX
    # -step ETASTEP
    parser.add_argument("-d", "--directory", help="Working directory",
                        type=str, default=os.getcwd())
    parser.add_argument("-j", "--job", help="Job type",
                        choices=['make', 'makerun', 'getEF'], default="make")
    parser.add_argument("-t", "--timer", help="Job submission interval",
                        type=int, default=5)
    parser.add_argument("-min", "--etamin", help="eta min",
                        type=float, default=-0.008)
    parser.add_argument("-max", "--etamax", help="eta max",
                        type=float, default=0.009)
    parser.add_argument("-step", "--etastep", help="eta step",
                        type=float, default=0.002)

    args = parser.parse_args(args)

    print("Working dir: ", args.directory)
    print("Job type: ", args.job)
    print("Timer: ", args.timer)
    print("eta: ", args.etamin, args.etamax, args.etastep)

    job_instance = ElasticJob(args.directory,
                              job=args.job,
                              timer=args.timer,
                              etamin=args.etamin,
                              etamax=args.etamax,
                              etastep=args.etastep)
    job_instance.do_job()

    print("Job done.")


if __name__ == '__main__':
    main(sys.argv[1:])
