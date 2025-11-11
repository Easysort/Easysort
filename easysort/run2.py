from easysort.registry import ResultRegistry, DataRegistry
import numpy as np
import json

def run():
    path = DataRegistry.LIST("argo")[10]
    array = np.array([1, 2, 3, 4, 5])
    json_data = {"array": array.tolist()}
    print(path)
    ResultRegistry.POST(path, "test", "test", json_data)
    ResultRegistry.POST(path, "test", "test", array)
    ResultRegistry.POST(path, "test", "test", b"hello")
