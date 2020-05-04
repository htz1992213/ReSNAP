from resnap.benchmark.structure import Structure
import numpy as np
import os
import shutil
import time
import math


etamin = -0.008
etamax = 0.009
etastep = 0.002

job = 'make'  # 'make' 'makenrun', 'getEF'
timer = 5


eta = np.arange(etamin, etamax, etastep)


def get_tensor():
        
    F = []
    F1 = []
    for i in range(len(eta)):
        Fi = np.eye(3)
        Fi[0][0] = math.sqrt(2*eta[i]+1)
        F1 += [Fi]
    F += [F1]

    F2 = []
    for i in range(len(eta)):
        Fi = np.eye(3)
        Fi[0][0] = math.sqrt(2*eta[i]+1)
        Fi[1][1] = math.sqrt(2*eta[i]+1)
        F2 += [Fi]
    F += [F2]


    F3 = []
    for i in range(len(eta)):
        Fi = np.eye(3)
        Fi[2][2] = math.sqrt(2*eta[i]+1)
        F3 += [Fi]
    F += [F3]

    F4 = []
    for i in range(len(eta)):
        Fi = np.eye(3)
        Fi[1][1] = math.sqrt(2*eta[i]+1)
        Fi[2][2] = math.sqrt(2*eta[i]+1)
        F4 += [Fi]
    F += [F4]


    F5 = []
    for i in range(len(eta)):
        Fi = np.eye(3)
        Fi[2][2] = math.sqrt(1-4*(eta[i]**2))
        Fi[1][2] = 2*eta[i]
        F5 += [Fi]
    F += [F5]


    F6 = []
    for i in range(len(eta)):
        Fi = np.eye(3)
        Fi[2][2] = math.sqrt(1-4*(eta[i]**2))
        Fi[0][2] = 2*eta[i]
        F6 += [Fi]
    F += [F6]

    F7 = []
    for i in range(len(eta)):
        Fi = np.eye(3)
        Fi[1][1] = math.sqrt(1-4*(eta[i]**2))
        Fi[0][1] = 2*eta[i]
        F7 += [Fi]
    F += [F7]

    return F


tensor = get_tensor()


if job == 'make' or job == 'makenrun':
    for i in range(7):
        for j in range(len(eta)):
            strdir = "s"+str(i+1)+"_%.2f" % (eta[j]*100)
            os.mkdir(strdir)  # Make directory si_j
        # copy relavent files to new directory
            shutil.copy("Reunit.dat", strdir)
            shutil.copy("Re_3.snapcoeff", strdir)
            shutil.copy("Re_3.snapparam", strdir)
            shutil.copy("input.lmp", strdir)
            shutil.copy("submit", strdir)

            os.chdir(strdir)

            cell = Structure("LAMMPSdat", "Reunit.dat")
            cell.addstrain(tensor[i][j])
            cell.write_lammps_data("Reunit.dat", "Re")

        # submit the job
            if job == 'makenrun':
                time.sleep(timer)
                os.system('sbatch submit')

            os.chdir('..')


def getdata():
    energy = []
    for m in range(7):
        energy_i = []
        stress = []
        for n in range(len(eta)):
            strdir = "s"+str(m+1)+"_%.2f" % (eta[n]*100)
            os.chdir(strdir)

            f = open('log.lammps', 'r')
            data = f.readlines()
            f.close()

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
        for j in range(len(eta)):
            fs.write("%.5f    " % eta[j])
        fs.write("\n")
        fs.write("pxx    pyy    pzz    pxy    pxz    pyz\n")

        for j in range(len(eta)):
            for k in range(6):
                fs.write("%.5f    " % stress[j][k])
            fs.write("\n")
        fs.close()

        energy.append(energy_i)

    fe = open("energy_data", "w")
    fe.write("eta range: \n")
    for m in range(len(eta)):
        fe.write("%.5f    " % eta[m])
    fe.write("\n")
    fe.write("energy (eV): \n")
    for m in range(len(energy)):
        for n in range(len(eta)):
            fe.write("%.5f    " % energy[m][n])
        fe.write("\n")
    fe.close()


if job == 'getEF':
    getdata()
