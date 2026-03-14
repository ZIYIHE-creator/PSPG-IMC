# -*- coding: utf-8 -*-

"""The methods to compute labels for machine learning the heat capacity."""

from ase.io import read
from .utils import * #改动之后导入utils中的全部内容    .utils表示包内引用，即同一个cp_app的package下的其他module
import pandas as pd
import numpy as np
import phonopy
from phonopy.units import CP2KToTHz
import matplotlib.pyplot as plt



def compute_total_dos_structure(phonopy_params: str, unitfactor=CP2KToTHz, dx:float=0.5, fmax:float=100.0, freq_pitch:float=0.001, saveto: str=None):
    """Compute projected dos from phonopy parameter file

    :param phonopy_params: list of phonopy parameter files (output of DFT)
    :param cif: list of crystal structure in cif format
    :param temperatures: the target temperature 
    :param factor: the unit conversion factor
    :param dx: spacing to compute dos
    :param fmax: max frequency in dos calculations
    :param freq_pitch: pitch frequency in dos calculations
    """
    phonon = None
    phonon = phonopy.load(phonopy_params, factor=unitfactor)
    phonon.run_mesh([1,1,1], with_eigenvectors=True)
    # total dos
    phonon.run_total_dos(sigma=dx,freq_min=0,freq_max=fmax, freq_pitch=freq_pitch)
    dos_dict = phonon.get_total_dos_dict()
    totaldos = np.vstack([dos_dict["frequency_points"],dos_dict["total_dos"]]).T
    if saveto:
        np.savetxt(saveto,totaldos,header="sigma = {}".format(dx))

    return totaldos


def compute_projected_dos_structure(phonopy_params: str, unitfactor=CP2KToTHz, dx=0.5, fmax=100, freq_pitch=0.001, saveto: str=None):
    """Compute projected dos from phonopy parameter file

    :param phonopy_params: list of phonopy parameter files (output of DFT)
    :cif: list of crystal structure in cif format
    :temperatures: the target temperature 
    :factor: the unit conversion factor
    :dx: spacing to compute dos
    :fmax: max frequency in dos calculations
    :freq_pitch: pitch frequency in dos calculations
    :saveto: save the projected dos to a file
    """
    phonon = None
    phonon = phonopy.load(phonopy_params, factor=unitfactor)
    phonon.run_mesh([1,1,1], with_eigenvectors=True)
    mesh = phonon.get_mesh_dict()
    # pdos
    phonon.run_projected_dos(sigma=dx,freq_min=0,freq_max=fmax, freq_pitch=freq_pitch)
    pdos_dict = phonon.get_projected_dos_dict()
    pdos = np.vstack([pdos_dict["frequency_points"],pdos_dict["projected_dos"]]).T
    if saveto:
        np.savetxt(saveto, pdos, header="sigma = {}".format(dx))

    return pdos 

def compute_atomic_cv_dataset(phonopy_params: list[str], cifs: list[str], temperatures: list, verbos=False, saveto: str="labels.csv") -> pd.DataFrame:
    """Compute atomic contribution to total cv from phonopy parameter file

    :param phonopy_params: list of phonopy parameter files (output of DFT)
    :cifs: list of crystal structure in cif format
    :temperatures: the target temperature 
    """
    labels={}
    for phonopy_param,cif in zip(phonopy_params,cifs):
        if verbos:
            print(cif)
        pdos=compute_projected_dos_structure(phonopy_params=phonopy_param, saveto="%s_projected_dos.dat".format(cif.replace(".cif","")))
        pdos[:,0]*=th2cm#频率THz换算成cm-1
# 测试不同 sigma 和 freq_pitch 对积分的影响
#        for siga in [0.15,0.16, 0.17, 0.18,0.19,0.2]:
#            totaldos = compute_total_dos_structure(phonopy_param, dx=siga, freq_pitch=0.5)
#            total_integral = np.trapz(totaldos[:,1], totaldos[:,0])
#            print(f"siga={siga}, Integral={total_integral:.2f} (理论值=3N)")#通过迭代找到dx=0.18时积分值接近理论值
#        for freq_pitchs in [0.5,0.498, 0.496, 0.494,0.492,0.49]:
#            total_dos = compute_total_dos_structure(phonopy_param, dx=0.18, freq_pitch=freq_pitchs)
#            plt.plot(total_dos[:, 0], total_dos[:, 1])
#            plt.xlabel("Frequency (THz)")
#            plt.ylabel("DOS")
#            plt.legend()
#            plt.show()
#            total_integral = np.trapz(total_dos[:,1], total_dos[:,0])
#            print(f"freq_pitch={freq_pitchs}, Integral={total_integral:.2f} (理论值=3N)")#通过迭代找到freq_pitch=0.496时积分值接近理论值,但是减小freq_pitch最后会收敛到214.5？

        atoms=read(cif)#读取cif文件没有问题，还把相对坐标转换成了绝对坐标了
        num_atoms = atoms.get_global_number_of_atoms()
        if not num_atoms==pdos.shape[1]-1:
            print(atoms.get_global_number_of_atoms(),pdos.shape[0])
            print("Warning! number of atoms do not match in pdos and structure for %s"%cif)
            continue
        for atomidx in range(num_atoms):
            site_name="%s_%i"%(cif.replace(".cif",""),atomidx)
            for i,temperature in enumerate(temperatures):
                cv_site=cv_from_pdos_site(temperature,pdos,atomidx)
                if i==0:
                    labels[site_name]={"pCv_%05.2f"%temperature:cv_site}
                else:
                    labels[site_name]["pCv_%05.2f"%temperature]=cv_site

    df=pd.DataFrame.from_dict(labels).T
    df.to_csv(saveto)

    return df