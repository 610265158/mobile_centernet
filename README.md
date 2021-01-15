# mobilenetv3_centernet

## introduction

This is a pytorch implement mobilenet-centernet framework,
which can be easily deployeed on Android(MNN) and IOS(CoreML) mobile devices, end to end.

Purpose: Light detection algorithms that work on mobile devices is widely used, 
such as face detection.
So there is an easy project contains model training and model converter. 

** contact me if u have question 2120140200@mail.nankai.edu.cn **



## pretrained model , and preformance

### mscoco

no test time augmentation.

| model                     |input_size |map      | map@0.5|map@0.75|
| :------:                  |:------:   |:------:  |:------:  |:------:  |
|[mbv2_100-centernet_stride8](https://drive.google.com/drive/folders/1wYcoPsa5boK-TEeyuJbQuYdOoY6NbWc5?usp=sharing)  |512x512     | 0.224| 0.383|0.228  |
|[mbv2_100-centernet_stride4](https://drive.google.com/drive/folders/1CJdX4XXcmEGdHEAMWg1sJ4CH-h2VKTNj?usp=sharing)  |512x512     | 0.234| 0.385|0.242  |

## requirment

+ pytorch

+ tensorpack 

+ opencv

+ python 3.6

+ MNNConverter

+ coremltools

## useage

### MSCOCO

#### train
1. download mscoco data, then run `python prepare_coco_data.py --mscocodir ./mscoco`



3. then, modify in config=mb3_config in train_config.py,  then run:

   ```python train.py```
   
   and if u want to check the data when training, u could set vis in confifs/mscoco/mbv3_config.py as True



#### evaluation

```
python model_eval/custome_eval.py [--model [TRAINED_MODEL]] [--annFile [cocostyle annFile]]
                          [--imgDir [the images dir]] [--is_show [show the result]]

python model_eval/custome_eval.py --model model/detector.pb
                                --annFile ../mscoco/annotations/instances_val2017.json
                                --imgDir ../mscoco/val2017

ps, no test time augmentation is used.
```


### visualization


`python visualization/vis.py --model yout.pth --imgDir yourimgdir`

u can check th code in visualization to make it runable, it's simple.


### model convert for mobile device
I have carefully processed the postprocess, and it can works within the model, so it could be deployed end to end.

4.1 MNN
    
    convert to onnx first
    
    + 4.1.1 convert model to onnx
    
        `python tools/converter_to_coreml.py --model your.pth`
        
    + 4.1.2 convert onnx to mnn
        
        './MNNConvert -f ONNX --modelFile centernet.onnx --MNNModel centernet.mnn --bizCode biz --weightQuantBits  8`
    
    + 4.1.2 visualization with mnn python wrapper

        `python visualization/vis_with_mnn.py --mnn_model centernet.mnn --imgDir 'your image dir'`

4.2 coreml
    
    ##some bugs in coremltools now, convert carefully. try to find the answer in coremltools issue
    + 4.2.1 convert

        `python tools/converter_to_coreml.py --model your.pth`

    + 4.2.2 visualization with coreml python wrapper

        `python visualization/vis_with_coreml.py --coreml_model centernet.mlmodel --imgDir 'your image dir'`


