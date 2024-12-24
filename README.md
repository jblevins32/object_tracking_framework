# Object Tracking Framework
This vision framework is for training and inference on custom PyTorch models with flexibility for various datasets, models, data augmentation. 

## TODO
- Fix dataloading structure to accomodate another set
- Fix data processing to implement k-folds
- Improve evaluation metrics with classic vision model metrics
- Move model blocks to blocks file to generalize the models
- Fix inference with new file names
- Allow inference to take in a series of images and show them as a video
- Allow inference to take in a video and and show result as an output video (real-time?)
- Generalize the loss function?

## How to Use:
- Training custom model:
  - Adjust hyperparameters and current model type selection in `config.yaml`
  - Run `train.py`
  - Select model and copy its path into inference.py to use a specific trained model
  - Run `inference.py`

## File Structure:
- `dataset`: All image files and corresponding labels for Training/Testing
  - `images`: Training and Testing image data
    - `training`
    - `testing`
  - `labels`: Labels for Training data, no labels for Testing data
- `figs`: Custom training output figures
- `trained_models`: Saved model state dictionaries
- `src`: Source code
  - `data_processing`: Downloading and processing of data
    - `data_downloader.py`: Download chosen data
    - `data_processing_kitti.py`: Process the KITTI data for training, validation, and testing
    - `data_processing.py`: Extracts KITTI data from the chosen dataset and sets it up for training 
  - `models`: PyTorch model architectures
    - `__blocks__.py`: PyTorch custom blocks for model building
    - `__init__.py`: imports all model files to the solver and dynamically chooses the model based on the config.yaml file_type parameter
    - `SimpleYOLO.py`: SimpleYOLO model implementation
    - `TinyYOLO.py`: TinyYOLO model implementation
    - `MidYOLO.py`: AttentionYOLO model implementation
    - `EncoderDecoderYOLO.py`: EncodeYOLO model implementation
  - `setup`: Helper functions
    - `globals.py`: Source the global directory
    - `load_args`: Load arguments from kwargs for training
  - `inference.py`: Run inference on the trained model
  - `loss.py`: Loss calculations
  - `process_output_img.py`: process inference images for viewing
  - `solver.py`: Core function for training the model
  - `train.py`: Train a model
- `config.yaml`: Model training hyperparameters
- `environment.yaml`: Conda environment

## Conda:
- Create environment with `conda env create -f environment.yaml`
- Still need to run `pip install torcheval` after activating conda env, pkg cannot be installed from conda env
