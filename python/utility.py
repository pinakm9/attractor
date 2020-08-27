from time import time
import random
import numpy as np

def timer(func):
	"""
	Timing wrapper for a generic function.
	Prints the time spent inside the function to the output.
	"""
	def new_func(*args, **kwargs):
		start = time()
		val = func(*args,**kwargs)
		end = time()
		print('Time taken by {} is {} seconds'.format(func.__name__, end-start))
		return val
	return new_func



def random_color(as_str=True, alpha=0.5):
	rgb = [random.randint(0,255),
		   random.randint(0,255),
		   random.randint(0,255)]
	if as_str:
		return "rgba"+str(tuple(rgb+[alpha]))
	else:
		# Normalize & listify
		return list(np.array(rgb)/255) + [alpha]
