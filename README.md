# Molecular Property Prediction Framework

This repository contains the training and inference pipeline for a deep
learning model used for encapusulation rate prediction.

------------------------------------------------------------------------

## Training

To train the model:

``` bash
python train.py
```

The trained model will be saved as:

    final_model.pth

------------------------------------------------------------------------

## Inference

To run inference:

``` bash
python inference.py --folder Working_5_to_10 --output test_predictions_2.csv --model final_model.pth --config config.json --nbfix NBFIX_table
```

------------------------------------------------------------------------

## Arguments

  -----------------------------------------------------------------------
  Argument                         Description
  -------------------------------- --------------------------------------
  `--folder`                       Input folder containing molecular
                                   structures or simulation outputs

  `--output`                       Output CSV file containing predictions

  `--model`                        Path to trained model weights (`.pth`)

  `--config`                       Configuration file used during
                                   training

  `--nbfix`                        Path to NBFIX parameter table
                                   (optional)
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Output

The inference script generates a CSV file:

    compound,predicted_encapsulation
    sample_001,0.8421
    sample_002,0.6134

------------------------------------------------------------------------

## Requirements

-   Python 3.11+
-   PyTorch
-   NumPy
-   Pandas
-   Torch Geometric
  
Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Typical Workflow

1.  Adjust `config.json`
2.  Train the model
3.  Run inference on new data
