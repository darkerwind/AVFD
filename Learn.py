import os
import datetime     as dt
import numpy        as np
import cv2          as cv

from Globals        import *
from Process        import Process

class LearningMode(Process):
    __PATH_STRING = "Default"
    __LST_IMG_CACHE = []
    __IDX_IMG_CACHE = 0
    __CV_FILE = None

    def __init__(self):
        pass
    
    @classmethod
    def __getPath(cls):
        return "./"+cls.__PATH_STRING+"/"
    
    @classmethod
    def __makeSaveFolder(cls, name):
        needMake = True

        startIndex = name.rfind("/")
        if startIndex == -1:
            startIndex = 0
        else:
            startIndex += 1
        lastIndex = name.rfind(".")
        if lastIndex == -1:
            lastIndex = len(name)
        cls.__PATH_STRING = name[startIndex:lastIndex]

        sd = os.scandir()
        for entry in sd:
            if entry.name == cls.__PATH_STRING:
                needMake = False
                break
        
        if needMake == True:
            os.mkdir(cls.__PATH_STRING)
        
        return True
    
    @classmethod
    def __saveNeuralNetwork(cls, session, saver, step):
        saver.save(session, "./Model/1280x720.ckpt", global_step=step)

    @classmethod
    def __saveImage(cls, name, label, model):
        cv.imwrite(cls.__getPath() + name + "_L.jpg", label)
        cv.imwrite(cls.__getPath() + name + "_M.jpg", model)
        #np.savetxt("./"+savePath+"/"+str(curFrameCount)+"_M.txt", frameModel[0], fmt="%.4f", footer="\r")
        #np.savetxt("./"+savePath+"/"+str(curFrameCount)+"_M.txt", frameModel[1], fmt="%.4f", footer="\r")
        #np.savetxt("./"+savePath+"/"+str(curFrameCount)+"_M.txt", frameModel[2], fmt="%.4f", footer="\r")
        return
    
    @classmethod
    def __readImageFromCache(cls):
        img = cls.__LST_IMG_CACHE[cls.__IDX_IMG_CACHE]
        cls.__IDX_IMG_CACHE += 1
        return img
    
    @classmethod
    def __readImageFromFile(cls):
        img = cls.__CV_FILE.read()
        cls.__LST_IMG_CACHE.append(img)
        return img

    @classmethod
    def run(cls):
        '''
        Learn from target file.
        Before calling this method, must set up Options by calling setOptions method,
        snd must set up NN Variables by calling setNNVariables method.
        '''
        dicOptions  = cls._getOptions()
        dicNNVars   = cls._getNNVariables()

        srcFileName     = dicOptions[PAR_FILENAME]
        startFrame      = int(dicOptions[OPT_START_FRAME])
        learningSize    = int(dicOptions[OPT_LEARNING_SIZE])
        enableLog       = (dicOptions[OPT_ENABLE_LOG] == "True")

        cls.__makeSaveFolder(srcFileName)

        sess        = dicNNVars[NN_SESSION]
        cost        = dicNNVars[NN_COST]
        optimizer   = dicNNVars[NN_OPTIMIZER]
        model       = dicNNVars[NN_MODEL]
        X           = dicNNVars[NN_X]
        Y           = dicNNVars[NN_Y]
        KP          = dicNNVars[NN_KP]
        saver       = dicNNVars[NN_SAVER]
        globalStep  = dicNNVars[NN_STEP]
        merged      = dicNNVars[NN_MERGED]
        writer      = dicNNVars[NN_WRITER]

        src = cv.VideoCapture(srcFileName)
        cls.__CV_FILE = src

        totalFrameCount = int(src.get(cv.CAP_PROP_FRAME_COUNT))

        repeatCount = 0
        targetCount = int(dicOptions[OPT_REPEAT_COUNT])
        while(repeatCount < targetCount):
            repeatCount += 1
            print("Repeat {} / {}".format(repeatCount, targetCount))
            if targetCount == repeatCount:
                enableLog = True
            beginTime = dt.datetime.today()

            curFrameCount = 1
            if startFrame > 0:
                print("Learning is started from {}".format(startFrame))
                src.set(cv.CAP_PROP_POS_FRAMES, startFrame - 1)
                curFrameCount = startFrame
            else:
                src.set(cv.CAP_PROP_POS_FRAMES, 0)

            if len(cls.__LST_IMG_CACHE) != 0:
                fReadImg = cls.__readImageFromCache
                cls.__IDX_IMG_CACHE = 0
            else:
                fReadImg = cls.__readImageFromFile

            result, framePrev = fReadImg()
            if result == False:
                return False
            result, frameAnswer = fReadImg()
            if result == False:
                return False

            processedFrameCount = 1
            totalCost = 0
            while(src.isOpened()):
                result, frameNext = fReadImg()
                if result == False:
                    break
                
                print("Process : {} / {}".format(curFrameCount, totalFrameCount), end="\r", flush=True)
                
                layerData = np.array([[framePrev[0:,0:,0], frameNext[0:,0:,0], framePrev[0:,0:,1], frameNext[0:,0:,1], framePrev[0:,0:,2], frameNext[0:,0:,2]]])
                layerData = np.swapaxes(layerData, 1,2)
                layerData = np.swapaxes(layerData, 2,3)
                layerAnswer = np.array([frameAnswer])
                if np.all(layerAnswer == 0) or np.all(framePrev == 0) or np.all(frameNext == 0):
                    print("Skip {} Frame.                                      ".format(curFrameCount))
                    framePrev = frameAnswer
                    frameAnswer = frameNext
                    processedFrameCount += 1
                    curFrameCount += 1
                    continue
                _, costVal = sess.run([optimizer, cost], feed_dict={X:layerData, Y:layerAnswer, KP:0.8})
                totalCost += costVal

                if (processedFrameCount % 100) == 0:
                    cls.__saveNeuralNetwork(sess, saver, globalStep)
                    summary = sess.run(merged, feed_dict={X:layerData, Y:layerAnswer, KP:0.8})
                    writer.add_summary(summary, global_step=sess.run(globalStep))
                    writer.flush()
                    if enableLog == True:
                        frameModel = sess.run(model, feed_dict={X:layerData, Y:layerAnswer, KP:1})
                        if np.max(frameModel) > 255.0:
                            #print(np.max(frameModel))
                            frameModel = (frameModel / np.max(frameModel)) * 255.0
                        frameModel = frameModel.astype(np.uint8)[0]
                        cls.__saveImage(str(curFrameCount), frameAnswer, frameModel)
                    print("{} Cost AVG / CUR : {:.3f} / {:.3f}                                  ".format(processedFrameCount, totalCost / 100, costVal))
                    totalCost = 0

                if processedFrameCount >= learningSize:
                    print("Done {} / {}                                   ".format(processedFrameCount, learningSize))
                    break
                
                framePrev = frameAnswer
                frameAnswer = frameNext
                processedFrameCount += 1
                curFrameCount += 1
            
            print("Total Training Count {}".format(sess.run(globalStep)))

            if totalFrameCount == curFrameCount:
                cls.__saveNeuralNetwork(sess, saver, globalStep)

            print("Process : Done                           ")
            
            endTime = dt.datetime.today()
            print("Elapsed Time : {}\n".format(endTime - beginTime))
        
        cls.__CV_FILE = None
        src.release()
        
        # remove opencv2.
        cv.destroyAllWindows()

        return True