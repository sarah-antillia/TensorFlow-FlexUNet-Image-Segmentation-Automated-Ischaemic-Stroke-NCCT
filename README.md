<h2>TensorFlow-FlexUNet-Image-Segmentation-Automated-Ischaemic-Stroke-NCCT (2025/09/21)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for AISD NCCT (Automated Ischaemic Stroke Dataset Non-Contrast CT) based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
 and a 512x512 pixels 
<a href="https://drive.google.com/file/d/18BRAvO9E1Hsr-lh89dc0vi5KmA4h2-fN/view?usp=sharing">AISD-ImageMask-Dataset.zip</a>, which was derived by us from <a href="https://github.com/GriffinLiang/AISD"
<b>AISD</b>
</a>
<br>
<br>
As demonstrated in <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel">
TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel</a>,
 our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to 
single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as 
a second category. In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
our dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/images/20324.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/masks/20324.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test_output/20324.png" width="320" height="320"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/images/20336.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/masks/20336.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test_output/20336.png" width="320" height="320"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/images/20540.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/masks/20540.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test_output/20540.png" width="320" height="320"></td>
</tr>
</table>

<hr>
<br>

<h3>1. Dataset Citation</h3>
The original dataset used here has been taken from the github repository:<br>
<a href="https://github.com/GriffinLiang/AISD">
<b>AISD</b>
</a>
<br><br>
<b>Description</b><br>
Acute ischemic stroke dataset contains 397 Non-Contrast-enhanced CT (NCCT) scans of acute ischemic stroke 
with the interval from symptom onset to CT less than 24 hours. The patients underwent diffusion-weighted MRI
 (DWI) within 24 hours after taking the CT. The slice thickness of NCCT is 5mm. 345 scans are used to train 
 and validate the model, and the remaining 52 scans are used for testing. Ischemic lesions are manually 
 contoured on NCCT by a doctor using MRI scans as the reference standard. Then a senior doctor double-reviews the labels.
<br>
<br>
<b>Download from Google Drive</b><br>
<a href="https://drive.google.com/file/d/157f9aE3ZhRSdIuIbP2PRG8ub9JJWvMGk/view?usp=share_link">image.zip</a><br>
<a href="https://drive.google.com/file/d/1d08fFpEvK4D6YTKfRlNuv_OlIxigZxl6/view?usp=share_link">mask.zip</a><br>
<br>
<b>Citation</b>
<pre style="font-size: 16px;"> 
@inproceedings{liang2021SymmetryEnhancedAN,
  title={Symmetry-Enhanced Attention Network for Acute Ischemic Infarct Segmentation with Non-Contrast CT Images},    
  author={Kongming Liang, Kai Han, Xiuli Li, Xiaoqing Cheng, Yiming Li, Yizhou Wang, and Yizhou Yu},    
  booktitle={MICCAI},    
  year={2021}    
}
</pre>

<br>
<b>License</b><br>
This dataset is made freely available to academic and non-academic entities for non-commercial purposes 
such as academic research, teaching, scientific publications, or personal experimentation.
<br>
<br>
<h3>
<a id="2">
2 AISD ImageMask Dataset
</a>
</h3>
 If you would like to train this AISD Segmentation model by yourself,
 please download  <a href="https://drive.google.com/file/d/18BRAvO9E1Hsr-lh89dc0vi5KmA4h2-fN/view?usp=sharing">AISD-ImageMask-Dataset.zip</a> on the google drive
, expand the downloaded and put it under <b>./dataset</b> folder to be.<br>
<pre>
./dataset
└─AISD
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
In order to generate this dataset,we excluded all black empty masks and their corresponding images,
 which were really irrelevant for training our segmentation model, from the original datasets <br>
<a href="https://drive.google.com/file/d/157f9aE3ZhRSdIuIbP2PRG8ub9JJWvMGk/view?usp=share_link">image.zip</a><br>
<a href="https://drive.google.com/file/d/1d08fFpEvK4D6YTKfRlNuv_OlIxigZxl6/view?usp=share_link">mask.zip</a><br>
in <a href="https://github.com/GriffinLiang/AISD">
<b>AISD</b>
</a><br>
Please note that commercial use of this dataset is prohibited.
<br><br>
<b>AISD Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/AISD/AISD_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br>
<br>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/AISD/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/AISD/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorFlowUNet Model
</h3>
 We have trained AISD TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/AISD/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/AISDand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small base_filters = 16, large base_kernels = (9,9), and large diation = (3,3) for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowUNet.py">TensorFlowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (3,3)

</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00005
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for AISD 1+1 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"

;     Background:black,  Ischaemic-Stroke: white 
rgb_map = {(0,0,0):0,(255, 255, 255):1,}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/AISD/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 23,24 25)</b><br>
<img src="./projects/TensorFlowFlexUNet/AISD/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 47,48,49)</b><br>
<img src="./projects/TensorFlowFlexUNet/AISD/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was stopped at epoch 49 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/AISD/asset/train_console_output_at_epoch49.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/AISD/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/AISD/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/AISD/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/AISD/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/AISD</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for AISD.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/AISD/asset/evaluate_console_output_at_epoch49.png" width="720" height="auto">
<br><br>Image-Segmentation-AISD

<a href="./projects/TensorFlowFlexUNet/AISD/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this AISD/test was not so low, but dice_coef high as shown below.
<br>
<pre>
categorical_crossentropy,0.0215
dice_coef_multiclass,0.9904
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/AISD</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for AISD.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/AISD/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/AISD/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
As shown below, this segmentation model failed to detect some Ischemia lesions.<br>

<img src="./projects/TensorFlowFlexUNet/AISD/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/images/20329.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/masks/20329.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test_output/20329.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/images/20348.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/masks/20348.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test_output/20348.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/images/20426.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/masks/20426.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test_output/20426.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/images/20603.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/masks/20603.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test_output/20603.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/images/20680.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/masks/20680.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test_output/20680.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/images/20429.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test/masks/20429.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/AISD/mini_test_output/20429.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Automated Segmentation of Ischemic Stroke Lesions in Non-Contrast Computed Tomography Images for Enhanced Treatment and Prognosis</b><br>
Toufiq Musah, Prince Ebenezer Adjei, Kojo Obed Otoo<br>

<a href="https://arxiv.org/html/2411.09402">
https://arxiv.org/html/2411.09402
</a>
<br>
<br>
<b>2. APIS: a paired CT-MRI dataset for ischemic stroke segmentation - methods and challenges </b><br>
Santiago Gómez, Edgar Rangel, Daniel Mantilla, Andrés Ortiz, Paul Camacho, Ezequiel de la Rosa, Joaquin Seia, <br>
Jan S. Kirschke, Yihao Li, Mostafa El Habib Daho & Fabio Martínez<br>

<a href="https://www.nature.com/articles/s41598-024-71273-x">
https://www.nature.com/articles/s41598-024-71273-x
</a>
<br>
<br>

<b>3. TensorFlow-FlexUNet-Image-Segmentation-Brain-Stroke-CT </b><br>
Toshiyuki Arai antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Brain-Stroke-CT">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Brain-Stroke-CT
</a>
<br>
<br>


