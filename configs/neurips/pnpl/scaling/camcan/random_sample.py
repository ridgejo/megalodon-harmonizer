import random
from pnpl.datasets.shafto2014 import constants

print("Total subjects", len(constants.SUBJECTS))
sample = random.sample(constants.SUBJECTS, k=152)
print(sample)