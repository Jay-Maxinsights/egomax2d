# Overview:
Create a new and better inference script for EgoMax2D model, other than inference/inference_heatmap_egomax2d.py

# Features:
1. Add preprocessing module
2. Decoupled load input, preprocess, inference, write overlay and writing result, so:
    * We can easily choose whether we want to overlay result or not
    * We can overlay the preprocess and inference so maximize the resource usage and minimize the timie usage  
3. Easy to tune batch size (right now its batch size 1)
4. Easy to monitor inference resource usage, including: time (total time, each stage time and per inference time), VRAM, RAM and disk        
5. The output format should be toon (same format as estimstion.toon)  
6. We will also add a test in tests/, Given to running result, we compare the estimation result: 
    * MPJPE in min, median, max and mean
    * Relative differences: Bit level differnece/GT (which is the result from inference/inference_heatmap_egomax2d.py),  in min, median, max and mean
  
# Reference: 
1. current inference script: inference/inference_heatmap_egomax2d.py 
2. Current preprocess script python scripts/max2D_Id_align.py --root data/EgoMax2D --apply and python scripts/build_egomax2d_cache.py --raw-root data/EgoMax2D --cached-root data/EgoMax2D_256.


# Requirements:
1. Do not change the original code, espicially inference/inference_heatmap_egomax2d.py. Everything we add a new script to it
2. Keep implementation simple and structure as possible 