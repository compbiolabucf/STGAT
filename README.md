# STGAT
This repository represents a framework named STGAT which can be trained on Spatial Transcriptomics (ST) data and applied on TCGA data to generate spot-level gene expression for the TCGA data. The produced gene expression can be leveraged various downstream analysis including subtype classification, survival prediction etc. Moreover, comprehensive experimentation proves that considering only the tumor spots to compute the mean gene expression of a sample results in better performance.

## Required Python libraries
- Python (>= 3.9.7)
- Pytorch (>= 1.11) [with cudatoolkit (>= 11.3.1) if cuda is used]
- scikit-learn (>= 1.0.2)
- scipy (>= 1.7.1)
- pandas (>= 1.3.4)
- numpy (>= 1.20.3)
- scikit-image (>=0.19.2)

## Dataset
Sample dataset can be found at this link: https://drive.google.com/drive/folders/1j9lQnHBFW59LXH-cWhHoreaUbmQW2csy?usp=sharing
The data directory should look like the following:
- io_data
   - gene_names.csv
   - ST
      - wsi
      - coords
      - gene_exp
    - TCGA
      - tcga_exp.csv
      - wsi

where 'wsi' contains the whole slide images for ST and TCGA samples, 'coords' and 'gene_exp' contain the coordinates and gene expression of the spots for ST samples. The bulk gene expression of the TCGA samples should be in 'tcga_exp.csv' file.

## Training and testing the model
The SEG and GEP modules can be trained and tested by running 'main.py' file in the command line. The options can be used to modify the training and model parameters.
