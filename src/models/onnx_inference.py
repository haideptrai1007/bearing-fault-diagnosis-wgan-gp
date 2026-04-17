import onnxruntime as ort
import numpy as np

class ONNXModel:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, x):
        outputs = self.session.run([self.output_name], {self.input_name: x})
        return outputs[0]