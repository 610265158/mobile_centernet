import sys
sys.path.append('.')
import torch
import torchvision


from lib.core.base_trainer.centernet import CenterNet

import argparse

from train_config import config as cfg
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str,default=None, help='the thres for detect')
args = parser.parse_args()

model_path=args.model




device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
dummy_input = torch.randn(1, 3, cfg.DATA.hin, cfg.DATA.win, device='cpu')
model = CenterNet(inference=True)

### load your weights
model.eval()

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=True)
# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.

torch.onnx.export(model,
                  dummy_input,
                  "centernet.onnx",
                  opset_version=11,
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  )
