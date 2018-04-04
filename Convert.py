import os

import numpy as np
import cv2 as cv
import subprocess as sp

from Process import Process

class ConvertMode(Process):
    def __init__(self):
        pass
    
    @classmethod
    def __getInterFrame(cls, prevFrame, nextFrame):
        frame = ((prevFrame / 2) + (nextFrame / 2)).astype(np.uint8)
        return frame

    @classmethod
    def __makeVideo(cls, srcFileName):
        dstFileName = srcFileName[0:srcFileName.rfind(".")] + "_New" + srcFileName[srcFileName.rfind("."):]
        #print(srcFileName + " --> " + dstFileName)
        
        src = cv.VideoCapture(srcFileName)
        resolution = (int(src.get(cv.CAP_PROP_FRAME_WIDTH)), int(src.get(cv.CAP_PROP_FRAME_HEIGHT)))
        fps = src.get(cv.CAP_PROP_FPS)
        fcc = int(src.get(cv.CAP_PROP_FOURCC))

        dst = cv.VideoWriter(dstFileName, fcc, (fps * 2.0), resolution)

        result, framePrev = src.read()
        if result == False:
            return ""
        curFrameCount = 1
        totalFrameCount = int(src.get(cv.CAP_PROP_FRAME_COUNT))
        while(src.isOpened()):
            result, frameNext = src.read()
            if result == False:
                break
            
            print("Process : {} / {}".format(curFrameCount, totalFrameCount), end="\r", flush=True)

            dst.write(framePrev)
            dst.write(cls.__getInterFrame(framePrev, frameNext))

            framePrev = frameNext
            curFrameCount += 1
        print("Process : Done                           ")
        
        src.release()
        dst.release()

        return dstFileName

    @classmethod
    def __makeAudio(cls, srcFileName):
        dstFileName = srcFileName[0:srcFileName.rfind(".")] + "_New.mp3"
        #print(srcFileName + " --> " + dstFileName)

        command = ["./FFMPEG/bin/ffmpeg.exe", "-i", srcFileName, "-c:a", "copy", dstFileName]
        result = sp.run(command)
        if result.returncode != 0:
            return ""

        return dstFileName

    @classmethod
    def __mergeAV(cls, videoFileName, audioFileName):
        dstFileName = videoFileName[0:videoFileName.rfind(".")] + "_Final" + videoFileName[videoFileName.rfind("."):]

        command = ["./FFMPEG/bin/ffmpeg.exe", "-i", videoFileName, "-i", audioFileName, "-c", "copy", dstFileName]
        result = sp.run(command)

        os.remove(videoFileName)
        os.remove(audioFileName)

        return result.returncode

    @classmethod
    def run(cls):
        '''
        Convert target file to double framed new file.
        Before calling this method, must set up Options by calling setOptions method,
        snd must set up NN Variables by calling setNNVariables method.
        '''
        dicOptions = cls._getOptions()
        dicNNVars = cls._getNNVariables()

        srcFileName = dicOptions["FileName"]

        videoFileName = cls.__makeVideo(srcFileName)
        if videoFileName == "":
            return False

        audioFileName = cls.__makeAudio(srcFileName)
        if audioFileName == "":
            return False

        cls.__mergeAV(videoFileName, audioFileName)

        # remove opencv2.
        cv.destroyAllWindows()

        return True