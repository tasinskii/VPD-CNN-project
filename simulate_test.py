from ase.io import read
from abtem.waves import PlaneWave
from abtem.waves import Probe
from abtem.potentials import PotentialArray
from abtem.potentials import Potential
import matplotlib.pyplot as plt
import numpy as np

atoms = read("CIF_files/SrS (1).cif")
#print(atoms)
#params? ask
probe = Probe(energy=200e3, semiangle_cutoff=20, rolloff=.1, sampling=.05)
#print(atoms[1])
#print("\n")

pos = [] #pos vector for multislice
for a in atoms:
    print(a.position)
    pos.append(a.position[:2])#compile x,y coords of all atoms read from cif
print(pos)
#save 4D STEM DATA
probe_exit_wave = probe.multislice(positions= pos, potential=atoms)
probe_exit_wave.write('CIF_files/result-data/SrS-res-1.hdf5')
#probe_exit_wave[1].show(cmap='gray')
image = np.abs(probe_exit_wave.array[0]) ** 2
plt.imshow(image.T, extent=[0, probe_exit_wave.extent[0], 0, probe_exit_wave.extent[1]], origin='lower')

cbed_diffraction_pattern = probe_exit_wave.diffraction_pattern()
cbed_diffraction_pattern.write('CIF_files/result-data/SrS-diffpat.hdf5')
cbed_diffraction_pattern[1].show(cmap='inferno')


print("done")
input()