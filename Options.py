import os

from Globals        import *
from Convert        import ConvertMode      as CMode
from Learn          import LearningMode     as LMode

class Options():
    __CLS_DIC_MODES = {MODE_CONVERT:CMode, MODE_LEARNING:LMode}
    __CLS_DIC_OPTIONS = {PAR_MODE:None, PAR_FILENAME:"", OPT_START_FRAME:"0", OPT_LEARNING_SIZE:"99999999", OPT_ENABLE_LOG:"False", OPT_REPEAT_COUNT:"1"}
    __CLS_DIC_OPTIONTYPE = {OPT_START_FRAME:str.isdecimal, OPT_LEARNING_SIZE:str.isdecimal, OPT_ENABLE_LOG:isboolean, OPT_REPEAT_COUNT:str.isdecimal}

    def __init__(self):
        pass

    @classmethod
    def __parseArgs(cls):
        argv = os.sys.argv[1:]

        if len(argv) < 2:
            return False
        
        mode = argv[0]
        if mode not in cls.__CLS_DIC_MODES:
            return False
        cls.__CLS_DIC_OPTIONS[PAR_MODE] = cls.__CLS_DIC_MODES[mode]

        cls.__CLS_DIC_OPTIONS[PAR_FILENAME] = argv[1]

        for argCode, argValue in zip(argv[2::2], argv[3::2]):
            if argCode not in cls.__CLS_DIC_OPTIONS:
                return False
            if cls.__CLS_DIC_OPTIONTYPE[argCode](argValue) == False:
                return False
            cls.__CLS_DIC_OPTIONS[argCode] = argValue

        return True
    
    @classmethod
    def getOptions(cls):
        if cls.__parseArgs() == False:
            return None
        
        return cls.__CLS_DIC_OPTIONS