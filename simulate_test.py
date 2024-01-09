# notebook last tested with abtem version 1.0.0beta7
from abtem import __version__
print('current version:', __version__)
from abtem import *
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms

device = 'cpu'

atoms = read("CIF_files/SrPbS2_tetragonal.cif")

atoms *= (8, 8, 1)

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

#show_atoms(atoms, ax=ax1, title='Top view')
#show_atoms(atoms, ax=ax2, plane='xz', title='Side view')

#default params
#gpts = 512, infite projection, .5 slice thickness, kirk param, energy 200e3, semiangle 9.4, rolloff 0.05
potential = Potential(atoms, 
                      gpts=512, 
                      device=device, 
                      projection='infinite', 
                      slice_thickness=3.03, 
                      parametrization='kirkland', 
                      storage=device).build(pbar=True)


detector = PixelatedDetector(max_angle=30)

end = (potential.extent[0] / 8, potential.extent[1] / 8)

scan = GridScan(start=[0, 0], end=end, sampling=0.1)

probe = Probe(energy=200e3, semiangle_cutoff=27.1, device='cpu', rolloff=0.05)

probe.grid.match(potential)

measurements = [detector.allocate_measurement(probe) for i in range(150)]

for indices, positions in scan.generate_positions(max_batch=20, pbar=True):
    probes = probe.build(positions)
    
    for measurement in measurements:
        probes = probes.multislice(potential, pbar=False)
        
        measurement += detector.detect(probes).sum(0)

fig, axes = plt.subplots(15,10, figsize=(15,10))
#print(measurements[0])
for i, ax in enumerate(axes.ravel()):
    np.save("pacbed-results/SrPbS2/pacbed-" + str(i) + "-SrPbS2", measurements[i].array)
    #ax.imshow(measurements[i].array, cmap='inferno')
    #ax.axis('off')

#plt.tight_layout()

#plt.show()


#print("done")

