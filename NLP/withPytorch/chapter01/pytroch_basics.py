import torch

def describe(x):
    print("type: {}".format(x.type()))
    print("shape: {}".format(x.shape()))
    print("x: {}".format(x.format()))