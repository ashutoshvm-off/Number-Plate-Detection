import cv2
import numpy as np 
from skimage.filter import threshold_local
import tensorflow as tf 
from skimage import measure
import imutils 
import os

def sort_cont(charaacter_contours):
    i = 0
    boundingBoxes = [cv2.boundingReact(c) for c in character_counters]
    (character_counters, boundingBoxes) = zip(*sorted(zip(charaacter_contours, boundingBoxes), key = lambda b: b[1][i], reverse = False))
    return charaacter_contours

def segment_chars(plate_img, fixed_width):
    v = cv2.split(cv2.cvtcolor(plate_img, cv2.COLOR_BGR2HSV))[2]
    thresh = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    plate_img = imutils.resize(plate_img, width = fixed_width)
    thresh  = imutils.resize(thresh, width = fixed_width)
    bgr_thresh = cv2.cvtcolor(thresh, cv2,COLOR_GRAY2BGR)
    labels = measure.label(thresh, background = 0)
    charCandidates = np.zeros(thresh.shape, dtype = 'uint8')
    character = []
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype ='uint8')
        labelMask[labels == label] = 255

        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]

        if len(cnts) > 0:
            c = max(cnts, key = cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingReact(c)
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW*boxH)
            heightRatio = boxH / float(plate_img.shape[0])
            keepAAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = heightRatio > 0.5 and heightRatio < 0.95

            if keepAAspectRatio and keepSolidity and keepHeight and boxW > 14:
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

        contours, hier = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contours = sort_cont(contours)
            addapixel = 4
            for c in contours:
                (x, y, w, h) = cv2.boundingReact(c)
                if y > addapixel:
                    y = y - addapixel
                else:
                    y = 0
                if x > addapixel:
                    x = x - addapixel
                else:
                    x = 0
                temp = bgr_thresh[y:y + h + (addapixel * 2), x:x + w + (addapixel * 2)]
                character.append(temp)
            return characters
        else:
            return None

class plateFinder:
    def __init__(self, minPlateArea, maxPlateArea):
        self.min_area = minPlateArea
        self.max_area = maxPlateArea
        self.element_structure = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize =(22, 3))
    
    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.sobel(gray, cv2.cv_8U, 1, 0, ksize = 3)
        ret2, threshhold_img = cv2.threshhold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        element = self.element_structure
        morph_n_thresholded_img = threshhold_img.copy()
        cv2.morphologyEx(src = threshhold_img, op = cv2.MORPH_CLOSE, kernel = element, dst = morph_n_thresholded_img)
        return morph_n_thresholded_img

    def extract_counters(self, after_preprocess):
        contours, _ = cv2.findContours(after_preprocess, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)
        return contours

    def clean_plate(self, plate):
        gray = cv2.cvtcolor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            area = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x, y,w, h = cv2.boundingReact(max_cnt)
            react = cv2.minAreaReact(max_cnt)
            if not self.ratioCheck(max_cntArea, plate.shape[1], plate.shape[0]):
                return plate, False, None
            return plate, True, [x, y, w, h]

        else:
            return plate, False, None

    def check_plate(self, input_img, contours):
        min_rect = cv2.minAreaReact(contour)
        if self.validateRatio(min_rect): 
            x, y, w, h = cv2.boundingRect(contour) 
            after_validation_img = input_img[y:y + h, x:x + w] 
            after_clean_plate_img, plateFound, coordinates = self.clean_plate(after_validation_img) 
               
            if plateFound: 
                characters_on_plate = self.find_characters_on_plate(after_clean_plate_img) 
                   
                if (characters_on_plate is not None and len(characters_on_plate) == 8): 
                    x1, y1, w1, h1 = coordinates 
                    coordinates = x1 + x, y1 + y 
                    after_check_plate_img = after_clean_plate_img 
                    return after_check_plate_img, characters_on_plate, coordinates 
           
        return None, None, None
    
    def find_possible_plates(self, input_img):
        plates = [] 
        self.char_on_plate = [] 
        self.corresponding_area = [] 
        self.after_preprocess = self.preprocess(input_img) 
        possible_plate_contours = self.extract_contours(self.after_preprocess)
        for cnts in possible_plate_contours: 
            plate, characters_on_plate, coordinates = self.check_plate(input_img, cnts) 
               
            if plate is not None: 
                plates.append(plate) 
                self.char_on_plate.append(characters_on_plate) 
                self.corresponding_area.append(coordinates) 
   
        if (len(plates) > 0): 
            return plates 
           
        else: 
            return None

    def find_characters_on_plate(self, plate): 
   
        charactersFound = segment_chars(plate, 400) 
        if charactersFound: 
            return charactersFound 

    def ratioCheck(self, area, width, height): 
        min = self.min_area 
        max = self.max_cntArea
        ratioMin = 3
        ratioMax = 6
        ratio = float(width) / float(height) 

        if ratio < 1: 
            ratio = 1 / ratio 
           
        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax): 
            return False
           
        return True

    def preRatioCheck(self, area, width, height): 
        min = self.min_area 
        max = self.max_area 
        ratioMin = 2.5
        ratioMax = 7
        ratio = float(width) / float(height) 
           
        if ratio < 1: 
            ratio = 1 / ratio 
   
        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax): 
            return False
           
        return True

    def validateRatio(self, rect): 
        (x, y), (width, height), rect_angle = rect 
   
        if (width > height): 
            angle = -rect_angle 
        else: 
            angle = 90 + rect_angle 
   
        if angle > 15: 
            return False
           
        if (height == 0 or width == 0): 
            return False
   
        area = width * height 
           
        if not self.preRatioCheck(area, width, height): 
            return False
        else: 
            return True

class OCR:
    def __init__(self, modelFile, labelFile): 
        self.model_file = modelFile 
        self.label_file = labelFile 
        self.label = self.load_label(self.label_file) 
        self.graph = self.load_graph(self.model_file) 
        self.sess = tf.compat.v1.Session(graph=self.graph, config=tf.compat.v1.ConfigProto())

    def load_graph(self, modelFile): 
        graph = tf.Graph() 
        graph_def = tf.compat.v1.GraphDef() 
        with open(modelFile, "rb") as f: 
            graph_def.ParseFromString(f.read()) 
        with graph.as_default(): 
            tf.import_graph_def(graph_def) 
        return graph 

    def load_label(self, labelFile): 
        label = [] 
        proto_as_ascii_lines = tf.io.gfile.GFile(labelFile).readlines() 
        for l in proto_as_ascii_lines: 
            label.append(l.rstrip()) 
        return label

    def convert_tensor(self, image, imageSizeOuput): 
        image = cv2.resize(image, dsize =(imageSizeOuput, imageSizeOuput), interpolation = cv2.INTER_CUBIC) 
        np_image_data = np.asarray(image) 
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX) 
        np_final = np.expand_dims(np_image_data, axis = 0) 
        return np_final

    def label_image(self, tensor): 
        input_name = "import/input"
        output_name = "import/final_result"
        input_operation = self.graph.get_operation_by_name(input_name) 
        output_operation = self.graph.get_operation_by_name(output_name) 
        results = self.sess.run(output_operation.outputs[0], {input_operation.outputs[0]: tensor}) 
        results = np.squeeze(results) 
        labels = self.label 
        top = results.argsort()[-1:][::-1] 
        return labels[top[0]] 

    def label_image_list(self, listImages, imageSizeOuput): 
        plate = "" 
        for img in listImages: 
               
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
            plate = plate + self.label_image(self.convert_tensor(img, imageSizeOuput)) 
        return plate, len(plate) 

if __name__ == "__main__": 
    findPlate = PlateFinder(minPlateArea=4100, maxPlateArea=15000) 
    model = OCR(modelFile="model/binary_128_0.50_ver3.pb", labelFile="model/binary_128_0.50_labels_ver2.txt") 
    cap = cv2.VideoCapture('test.MOV') 
    while (cap.isOpened()): 
        ret, img = cap.read() 
        if ret == True: 
            cv2.imshow('original video', img) 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
            possible_plates = findPlate.find_possible_plates(img) 
            if possible_plates is not None: 
                   
                for i, p in enumerate(possible_plates): 
                    chars_on_plate = findPlate.char_on_plate[i] 
                    recognized_plate, _ = model.label_image_list(chars_on_plate, imageSizeOuput = 128) 
                    print(recognized_plate) 
                    cv2.imshow('plate', p) 
                    if cv2.waitKey(25) & 0xFF == ord('q'): 
                        break
        else: 
            break
               
    cap.release() 
    cv2.destroyAllWindows()