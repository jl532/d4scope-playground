# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:33:36 2022

@author: jason
"""
import cv2
import numpy as np

class D4ArrayClass:
    def __init__(self):
        pass
    def importImage(self, method, filePath = "", array = ""):
        """
            import and initialize images into the D4ArrayClass. 
            We handle both old file uploads, and accept fresh arr images.
            
            provide the image path to one image.             
            eg. C:/Users/jason/CODE/Image labeller/images-cov/58A_SLSp_LT.tiff
            or 
            provide the numpy array for the . 16bit depth int.
            
        """
        self.original_image = [] # upload original 16b image
        self.image8b = [] # convert to 8bit for faster processing
        self.imageVerif = [] # convert mono to RGB for annotation
        
        if method == "fileUpload":
            print("file upload")
            self.original_image = cv2.imread(filePath,-1)
            self.image8b = cv2.normalize(self.original_image.copy(),
                                     np.zeros(shape=self.original_image.shape),
                                     0, 255,
                                     norm_type=cv2.NORM_MINMAX,
                                     dtype=cv2.CV_8U)
            self.imageVerif = cv2.cvtColor(self.image8b.copy(), 
                                           cv2.COLOR_GRAY2RGB)
        elif method == "array":
            print("array upload")
            
    def defArrayConfig(self, configDict = {}, configPath = ""):
        
        pass
    
    def patternMatch(self):
        pass
    
    def viewAllImages(self, timer = -1):
        cv2.imshow("original image", self.original_image)
        cv2.imshow("8 bit image", self.image8b)
        cv2.imshow("verification image", self.imageVerif)
        cv2.waitKey(timer)
        cv2.destroyAllWindows()
        
    def viewVerif(self, timer = -1):
        cv2.imshow("verification image", self.imageVerif)
        cv2.waitKey(timer)
        cv2.destroyAllWindows()
    
    def analyzeArray(self):
        pass
    
    
if __name__ == "__main__":
    """
        Whatever code to run in case this is the main script. will need to implement unit tests to make sure individual components in the functions works.
    """
    testObject = D4ArrayClass()
    #testObject.importImage("fileUpload",filePath = "C:/Users/jason/CODE/Image labeller/images-cov/58A_SLSp_LT.tiff")
    hmm = testObject.original_image
    print(hmm.shape)