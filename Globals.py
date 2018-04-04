VERSION_CODE        = "0.2.1"

MODE_CONVERT        = "--c"
MODE_LEARNING       = "--l"

OPT_START_FRAME     = "--sf"
OPT_LEARNING_SIZE   = "--ls"
OPT_ENABLE_LOG      = "--el"
OPT_REPEAT_COUNT    = "--rc"

PAR_MODE            = "Mode"
PAR_FILENAME        = "Filename"

NN_SESSION          = "SESS"
NN_X                = "X"
NN_Y                = "Y"
NN_KP               = "KP"
NN_COST             = "COST"
NN_OPTIMIZER        = "OPTIMIZER"
NN_STEP             = "GLOBAL STEP"
NN_MODEL            = "MODEL"
NN_SAVER            = "SAVER"
NN_MERGED           = "MERGED"
NN_WRITER           = "WRITER"

def isboolean(string):
    if string in ("True", "False"):
        return True
    return False
