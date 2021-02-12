import torch
import numpy as np
import GPUtil
import IPython
import sys

# Set device
if torch.cuda.is_available():
	available_gpus = GPUtil.getAvailable(lilogo=2, maxLoad=0.5, maxMemory=0.009)
	device = torch.device("cuda:1")
else:
	device = torch.device("cpu")
# print("Running experiment on device: %s:%d" % (device.type, 0 if device.index is None else device.index))

# Setting read/write directories
def get_hostname():
	with open("/etc/hostname") as f:
		hostname=f.read()
	hostname=hostname.split('\n')[0]
	return hostname

hostname = get_hostname()
if hostname == "nsh1609server4":
	rawdir = "/your/data/load/path/here/"
	rltdir = "/your/output/path/here"

# fix seeds for deterministic runs
random_seed_number=1
np.random.seed(random_seed_number)
torch.manual_seed(random_seed_number)
# make cudnn backend deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False