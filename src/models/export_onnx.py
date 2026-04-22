import torch
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

    
class CalibDataReader(CalibrationDataReader):
    """
    Calibration Dataset
    """
    def __init__(self, data_loader, input_name='input'):
        self.data = []
        for images, _ in data_loader:
            for img in images:
                self.data.append({input_name: img.unsqueeze(0).numpy()})
            if len(self.data) >= 200:
                break
        self.index = 0

    def get_next(self):
        if self.index >= len(self.data):
            return None
        item = self.data[self.index]
        self.index += 1
        return item
    
def export_onnx_int8(model_instance, weight_path, onnx_fp32_path, onnx_int8_path, calib_loader):
    """
    Exporting int8 onnx model
    """
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_instance.load_state_dict(checkpoint['model_state_dict'])
    model_instance.eval()

    dummy = torch.randn(1, 1, 128, 128)
    torch.onnx.export(model_instance, dummy, onnx_fp32_path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                      opset_version=17)
    print(f"FP32 exported → {onnx_fp32_path}")

    reader = CalibDataReader(calib_loader)
    quantize_static(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        calibration_data_reader=reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    print(f"INT8 exported → {onnx_int8_path}")