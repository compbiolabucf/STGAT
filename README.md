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


## Sample dataset can be found at https://drive.google.com/drive/folders/1j9lQnHBFW59LXH-cWhHoreaUbmQW2csy?usp=sharing
If using own dataset, a folder named 'io_data' should be created in the same directory where 'main.py' is located. 
#io_data
  *[ST/](.\io_data\ST)
    *[wsi](.\io_data\ST\wsi)
    *[coords](.\io_data\ST\coords)
    *[gene_exp](.\io_data\ST\gene_exp)
  *[TCGA/](.\io_data\TCGA)
    *[wsi](.\io_data\TCGA\wsi)
    *[tcga_exp.csv](.\io_data\TCGA\tcga_exp.csv)
  *[gene_names.csv](.\io_data\gene_names.csv)
