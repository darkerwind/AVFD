class Process():
    __CLS_DICT_OPTIONS = {}
    __CLS_DICT_NN_VARIABLES = {}

    def __init__(self):
        pass
    
    @classmethod
    def setOptions(cls, dicOptions):
        '''
        Set up options dictionary for working.
        dicOptions argument must be dict type.
        '''
        if type(dicOptions) != dict:
            return False
        
        cls.__CLS_DICT_OPTIONS = dicOptions
        return True
    
    @classmethod
    def _getOptions(cls):
        '''
        Get options dictionary for working.
        '''
        return cls.__CLS_DICT_OPTIONS
    
    @classmethod
    def setNNVariables(cls, dicNNVariables):
        '''
        Set up neural network variables dictionary for working.
        dicNNVariables argument must be dict type.
        '''
        if type(dicNNVariables) != dict:
            return False
        
        cls.__CLS_DICT_NN_VARIABLES = dicNNVariables
        return True
    
    @classmethod
    def _getNNVariables(cls):
        '''
        Get neural network variables dictionary for working.
        '''
        return cls.__CLS_DICT_NN_VARIABLES
    
    @classmethod
    def run(cls):
        return True

if __name__ == "__main__":
    if Process.setOptions(None) == False:
        print("Check OK.")
    if Process.setOptions({}) == True:
        print("Check OK.")

    if Process.setNNVariables(None) == False:
        print("Check OK.")
    if Process.setNNVariables({}) == True:
        print("Check OK.")