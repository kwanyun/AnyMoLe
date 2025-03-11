## ___***AnyMoLe: Any Character Motion Inbetweening via Video Diffusion Prior***___
<!-- ![](./assets/logo_long.png#gh-light-mode-only){: width="50%"} -->
<!-- ![](./assets/logo_long_dark.png#gh-dark-mode-only=100x20) -->
### ! IMPORTANT ! Current version is not finalized. Full code will be released soon.


### How to run code (in short)
1.  Before start, using 2 sec context motion and keyframes, render frame
2. First, Inference video (ICAdapt+inference). Next, Train Scene-specific joint estimator
3. Last, DO Motion video mimicking 


Here, Motion video mimicking is a new terminology for adopting character motion from video generation model.

### Get data from context and key frames


### Step 1 : ICAdapt and inference.
    python run_anymole.py --frames_dir "$dir" --text_input --ICAdapt --interp--onlykey 750 --stage 2

### Step 2 : Train Scene-specific joint estimator.
    python pose3d_train.py
    python pose3d_inference.py
    
### Step 3 : Motion video mimicking
    python motion_video_mimicking.py
