import pandas as pd

class LicensePlateIdentifier():

    def __init__(self, frame, alpr):
        self.frame = frame
        self.alpr = alpr
        pass

    def apply_alpr(self):
        self.results = self.alpr.recognize_ndarray(self.frame)

    def extract_output(self):
        i = 0
        for plate in self.results['results']:
            i += 1
            for candidate in plate['candidates']:
                if candidate['matches_template']:
                    plate = candidate['plate']
                    confidence = candidate['confidence']
                    break
        
        return plate, confidence

    def extract_plates(self):
        return self.results['results'], self.results['results'][0]['candidates']

    def get_plate_information(self, plate):
        dvla_memory_tag = [plate[0], plate[1]]
        age_identifier = [plate[2], plate[3]]
        random_letters = plate[4:]

        return "".join(dvla_memory_tag), "".join(age_identifier), random_letters
