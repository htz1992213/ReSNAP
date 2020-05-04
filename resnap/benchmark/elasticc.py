from resnap.benchmark.structure import Structure
from pymatgen.io.lammps.outputs import parse_lammps_log
import numpy as np
import os
import shutil
import time
import math


class ElasticJob:

    def __init__(self, job="make", timer=5, etamin=-0.008, etamax=0.009,
                 etastep=0.002):
        self.f_tensor = self.get_f_tensor()
        self.timer = timer
        self.job = job
        self.eta = np.arange(etamin, etamax, etastep)

    def do_job(self):
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

                f = open('log.lammps', 'r')
                data = f.readlines()
                f.close()

                nums = list()
                for k in range(len(data)):
                    if data[k].find("Loop time") > -1:
                        nums = data[k-1].split()
                nums = [float(num) for num in nums]
                [step, energy_ij, pxx, pyy, pzz, pxy, pxz, pyz] = nums
                energy_i.append(energy_ij)
                stress.append([pxx, pyy, pzz, pxy, pxz, pyz])

                os.chdir('..')

            fs = open("stress_"+str(m+1), "w")
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

            energy.append(energy_i)

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
