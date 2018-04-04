from Globals        import *
from Information    import Information      as Info
from NN             import NeuralNetwork    as NNet
from Options        import Options          as Opts

def main():
    Info.printTitle()

    # get options from arguments.
    dicOptions = Opts.getOptions()
    if dicOptions == None:
        Info.printHelp()
        return False
    workClass = dicOptions[PAR_MODE]
    
    # get variables for NN.
    dicNNVariables = NNet.getNNVariables()

    # set up parameters.
    workClass.setOptions(dicOptions)
    workClass.setNNVariables(dicNNVariables)
    
    # run work-function.
    if workClass.run() == False:
        print("Fail.")
        return False

    print("Complete.")

    return True

if __name__ == "__main__":
    main()
