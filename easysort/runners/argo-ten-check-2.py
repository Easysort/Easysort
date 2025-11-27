

import os


class ArgoCounter:
    def __init__(self, folder: str):
        self.images = os.listdir(folder)
        

    def remove_duplicates(self):







if __name__ == "__main__":
    counter = ArgoCounter(folder="output")