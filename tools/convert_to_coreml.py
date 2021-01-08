import sys
sys.path.append('.')

from lib.core.model.centernet import CenterNet

import argparse

from train_config import config as cfg
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str,default=None, help='the thres for detect')
args = parser.parse_args()

model_path=args.model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch

import coremltools as ct

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
dummy_input = torch.randn(1, 3, cfg.DATA.hin, cfg.DATA.win, device='cpu')
model = CenterNet(inference=True,coreml=True)

### load your weights
model.eval()



state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=True)
trace = torch.jit.trace(model, dummy_input)

# Convert the model
mlmodel = ct.convert(
    trace,
    inputs=[ct.ImageType(name="__input", shape=dummy_input.shape)],
)
spec = mlmodel.get_spec()

# Edit the spec
ct.utils.rename_feature(spec, '__input', 'image')
ct.utils.rename_feature(spec, '2574', 'output')
# save out the updated model
mlmodel = ct.models.MLModel(spec)
print(mlmodel)



from coremltools.models.neural_network import quantization_utils
from coremltools.models.neural_network.quantization_utils import AdvancedQuantizedLayerSelector

selector = AdvancedQuantizedLayerSelector(
    skip_layer_types=['batchnorm', 'depthwiseConv'],
    minimum_conv_kernel_channels=4,
    minimum_conv_weight_count=4096
)

model_fp16 = quantization_utils.quantize_weights(mlmodel, nbits=8,quantization_mode='linear',selector=selector)



fp_16_file='./centernet.mlmodel'
model_fp16.save(fp_16_file)

