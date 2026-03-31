import numpy as np
import matplotlib.pyplot as plt

# Load in eigen values (they are pre-sorted)
evals = np.loadtxt("SYK_Simulation/spectralData/eigenValueData/evals_K6_J3.000000_seed10.csv", delimiter = ",")

plt.plot(np.arange(len(evals)), evals, marker='o', linestyle='none')
plt.xlabel("Eigenvalue index")
plt.ylabel("Energy")
plt.show()

spacings = np.loadtxt("SYK_Simulation/spectralData/eigenRepulsionData/erep_K6_J3.000000_seed10.csv", delimiter=",")

plt.hist(spacings, bins=30, density=True)
plt.xlabel("Level spacing s")
plt.ylabel("P(s)")
plt.title("Level spacing distribution")
plt.show()
