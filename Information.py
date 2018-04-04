import os

from Globals import *

class Information():
    def __init__(self):
        pass
    
    @staticmethod
    def printTitle():
        os.system("cls")
        print("AI Video Frame Doubler, Ver {}\n".format(VERSION_CODE))
    
    @staticmethod
    def printHelp():
        print("Usage::")
        print(">> AVFD.py option targetFilename [learning suboptions...]")
        print("")
        print("Options::")
        print("--l : Learning Mode. AVFD learn from targetFilename.")
        print("--c : Convert Mode. AVFD makes new double frame file from targetFilename.")
        print("")
        print("Suboption::")
        print("--sf integer : Set begining frame for learning. Default is 0.")
        print("--ls integer : Set learning size. Default is infinite.")
        print("--el boolean : Toggle log writing. Default is False.")
        print("--rc integer : Set repeat count for learning. Default is 1.")
        print("")

if __name__ == "__main__":
    Information.printTitle()
    Information.printHelp()