# YOLObile 
arXiv: [https://arxiv.org/abs/2009.05697](https://arxiv.org/abs/2009.05697)

## Introduction
The rapid development and wide utilization of object detection techniques have aroused attention on both accuracy and speed of object detectors. However, the current state-of-the-art object detection works are either accuracy-oriented using a large model but leading to high latency
or speed-oriented using a lightweight model but sacrificing accuracy. In this work, we propose YOLObile framework, a real-time object detection on mobile devices via compression-compilation co-design. A novel block-punched pruning scheme is proposed for any kernel size. To improve computational efficiency on mobile devices, a GPU-CPU collaborative  scheme is adopted along with advanced compiler-assisted optimizations. Experimental results indicate that our pruning scheme achieves 14x compression rate of YOLOv4 with 49.0 mAP. 
Under our YOLObile framework, we achieve 17 FPS inference speed using GPU on Samsung Galaxy S20. 
By incorporating our proposed GPU-CPU collaborative scheme, the inference speed is increased to 19.1 FPS, and outperforms the original YOLOv4 by 5x speedup.


![Image of YOLObile](figure/yolobilemap.png)

## Test our model
We release 8x and 14x pruning model in this repository and provide testing instructions.
### Environment Requirements:

Ubuntu 18.04

cuda version = 10.2


```shell
conda create --name yolotest  python==3.6.9
conda activate yolotest
```

install requirements:

```shell script
 pip install -r requirements.txt
```

### Download Coco Dataset: (18 GB)
```shell script
cd ../ && sh YOLObile/data/get_coco2014.sh
```
### Download Model Checkpoints:
Google Drive: [Google Drive Download](https://drive.google.com/drive/folders/1FcWdXcWc3vScV-guIrxWsWGhjQwPOEQW?usp=sharing)

Baidu Netdisk: [Baidu Netdisk Download](https://pan.baidu.com/s/1FMTOQF6ebH6OJWEAq9F0KQ) code: r3nk
### Check model Weight parameters & Flops:
```shell script
python check_compression.py
```
### Test model MAP:

```shell script
python test.py --img-size 320 --batch-size 64 --device 0 --cfg cfg/csdarknet53s-panet-spp.cfg --weights weights/best8x-514.pt --data data/coco2014.data

```
```
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|| 79/79 [00:
                 all     5e+03  3.51e+04     0.501     0.544     0.508     0.512
              person     5e+03  1.05e+04     0.643     0.697     0.698     0.669
             bicycle     5e+03       313     0.464     0.409     0.388     0.435
                 car     5e+03  1.64e+03     0.492     0.547     0.503     0.518
          motorcycle     5e+03       388     0.602     0.635     0.623     0.618
            airplane     5e+03       131     0.676     0.786     0.804     0.727
                 bus     5e+03       259      0.67     0.788     0.792     0.724
               train     5e+03       212     0.731     0.797     0.805     0.763
               truck     5e+03       352     0.414     0.526     0.475     0.463
                boat     5e+03       458     0.459     0.424     0.387     0.441
       traffic light     5e+03       516     0.425     0.413     0.357     0.419
        fire hydrant     5e+03        83     0.808     0.795     0.826     0.802
           stop sign     5e+03        84     0.633      0.69     0.722     0.661
       parking meter     5e+03        59      0.53     0.424     0.425     0.471
               bench     5e+03       471     0.411     0.308     0.292     0.352
                bird     5e+03       453     0.442     0.478     0.439      0.46
                 cat     5e+03       195     0.673       0.8     0.774     0.731
                 dog     5e+03       223     0.619     0.789      0.76     0.694
               horse     5e+03       305     0.673     0.764     0.766     0.716
               sheep     5e+03       306     0.562     0.719     0.699     0.631
                 cow     5e+03       376     0.559     0.637     0.612     0.595
            elephant     5e+03       283     0.744     0.883     0.894     0.807
                bear     5e+03        53     0.706      0.83     0.855     0.763
               zebra     5e+03       275     0.734     0.836      0.85     0.782
             giraffe     5e+03       170     0.849     0.853     0.876     0.851
            backpack     5e+03       384     0.355     0.331     0.273     0.342
            umbrella     5e+03       387     0.495     0.587     0.528     0.537
             handbag     5e+03       483     0.241     0.195      0.14     0.215
                 tie     5e+03       290     0.477     0.441     0.419     0.458
            suitcase     5e+03       309     0.436     0.608     0.533     0.508
             frisbee     5e+03       109     0.542      0.55     0.557     0.546
                skis     5e+03       281     0.484     0.381     0.358     0.426
           snowboard     5e+03        90     0.418     0.511      0.43      0.46
         sports ball     5e+03       233     0.568     0.403     0.421     0.472
                kite     5e+03       381     0.512     0.583     0.507     0.545
        baseball bat     5e+03       123     0.447     0.496     0.465      0.47
      baseball glove     5e+03       139     0.479      0.39     0.383      0.43
          skateboard     5e+03       215     0.617     0.614     0.633     0.616
           surfboard     5e+03       266     0.569     0.556     0.535     0.563
       tennis racket     5e+03       183     0.659     0.667     0.675     0.663
              bottle     5e+03       939      0.37     0.485     0.385     0.419
          wine glass     5e+03       363     0.517     0.399     0.403     0.451
                 cup     5e+03       891     0.449     0.444     0.424     0.447
                fork     5e+03       234     0.401     0.363     0.326     0.381
               knife     5e+03       290     0.299     0.217     0.182     0.252
               spoon     5e+03       253     0.278     0.229     0.165     0.251
                bowl     5e+03       617     0.417     0.502     0.416     0.456
              banana     5e+03       359     0.328      0.46     0.331     0.383
               apple     5e+03       158     0.212     0.381       0.2     0.273
            sandwich     5e+03       158     0.438     0.563     0.471     0.493
              orange     5e+03       185     0.294     0.395     0.272     0.337
            broccoli     5e+03       330     0.389     0.518      0.39     0.444
              carrot     5e+03       341     0.325     0.431      0.31     0.371
             hot dog     5e+03       160     0.488     0.412     0.397     0.447
               pizza     5e+03       223     0.604     0.682     0.687     0.641
               donut     5e+03       225     0.403      0.68     0.591     0.506
                cake     5e+03       236     0.443      0.53     0.511     0.483
               chair     5e+03  1.59e+03     0.436     0.462     0.398     0.449
               couch     5e+03       236     0.477     0.674     0.624     0.558
        potted plant     5e+03       429     0.435      0.45     0.377     0.442
                 bed     5e+03       195     0.608     0.723     0.687      0.66
        dining table     5e+03       633     0.458     0.517      0.45     0.485
              toilet     5e+03       179     0.713     0.838     0.795     0.771
                  tv     5e+03       257     0.559     0.794     0.766     0.656
              laptop     5e+03       236     0.571       0.7     0.714     0.629
               mouse     5e+03        95     0.473     0.653     0.588     0.548
              remote     5e+03       241     0.364     0.394     0.327     0.378
            keyboard     5e+03       117     0.465     0.692      0.65     0.557
          cell phone     5e+03       291     0.391     0.333     0.313      0.36
           microwave     5e+03        88     0.545     0.693     0.697      0.61
                oven     5e+03       142     0.497     0.599     0.524     0.543
             toaster     5e+03        11         0         0    0.0366         0
                sink     5e+03       211     0.486     0.583     0.511      0.53
        refrigerator     5e+03       107     0.532     0.724      0.69     0.613
                book     5e+03  1.03e+03     0.253     0.234     0.134     0.243
               clock     5e+03       290     0.636     0.624     0.621      0.63
                vase     5e+03       350     0.422     0.517     0.436     0.464
            scissors     5e+03        56     0.466     0.393     0.362     0.426
          teddy bear     5e+03       238     0.431     0.672     0.593     0.525
          hair drier     5e+03        11         1    0.0909     0.102     0.167
          toothbrush     5e+03        77      0.35     0.301     0.269     0.323
Speed: 3.6/1.4/5.0 ms inference/NMS/total per 320x320 image at batch-size 64

COCO mAP with pycocotools...
loading annotations into memory...
Done (t=3.87s)
creating index...
index created!
Loading and preparing results...
DONE (t=3.74s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=83.06s).
Accumulating evaluation results...
DONE (t=9.39s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.334
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.514
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.350
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.117
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.295
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.466
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.240
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.727
