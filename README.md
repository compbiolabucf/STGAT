# STGAT
This repository represents a Machine Learning framework named STGAT (Spatial Transcriptomics Graph Attention Network) which can be trained on Spatial Transcriptomics (ST) data and applied on TCGA data to generate spot-level gene expression for the TCGA samples. The produced gene expression can be leveraged various downstream analysis including subtype classification, survival prediction etc. Moreover, comprehensive experimentation proves that considering only the tumor spots to compute the mean gene expression of a sample results in better performance.

## Workflow
![alt text](https://github.com/compbiolabucf/STGAT/blob/main/STGAT_overall_diagram.pdf)

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
   - gene_names.csv (name of the genes to be predicted ('.csv' format) with column/Series name ['Gene'])
   - ST
      - wsi (Whole slide images of the ST samples in '.jpg/'.jpeg'/'.tiff' format)
      - coords (Files containing the coordinates of the spots of those samples ('.csv' format) with column names) 
      - gene_exp (Files which contains the gene expression of the spots for the ST samples ('.csv' format)
    - TCGA
      - tcga_exp.csv (Single '.csv' file contatining the bulk gene expression of the TCGA samples)
      - wsi (Whole slide images of the TCGA samples ('.svs' format))

Format for the files contatining the gene expression should be:
|         | Gene1  | Gene2 | Gene3 | Gene4 |
|---------|--------|-------|-------|-------|
| Sample1 |    -   |    -  |   -   |   -   |
| Sample2 |    -   |    -  |   -   |   -   |
| Sample3 |    -   |    -  |   -   |   -   |

'Sample' means spot in case spot level gene expression and TCGA sample name in case of TCGA bulk gene expression.

The coordinate files' format should be: 
|         |    X   |    Y  |
|---------|--------|-------|
| Spot1   |    -   |    -  |
| Spot2   |    -   |    -  |
| Spot3   |    -   |    -  |

## Training and testing the model
The SEG and GEP modules can be trained and tested by running 'main.py' file in the command line. The options can be used to modify the training and model parameters. For example,
```
python main.py --sp_dir io_data/ST/ --tcga_dir io_data/TCGA/
```
Codes for SEG and GEP modules can be found in the respective directories. 'split' directory contains the codes for splitting the patches from the ST and TCGA samples. For the TCGA samples, coordinate files are also generated. 
After successfull training of SEG and GEP, the directory structure should look like the following:
- SEG (containing SEG codes)
- GEP (containing GEP codes)
- io_data
   - gene_names.csv
   - ST
      - wsi
      - coords
      - gene_exp
      - patches (Containing split ST spots)
    - TCGA
      - tcga_exp.csv
      - wsi
      - patches (containing split TCGA spots)
      - coords (containing files having the coordinates of the generated spots from TCGA samples)
- prediction (Predicted spot-level gene expression of the test TCGA samples)
- results (Predicted spot-level gene expression of the test ST samples)
- saved_moodel (directory for saving modeles temporarily in each eporch)
- trained (directory for saving trained SEG and GEP modules)


NB: 
1. Sample data is provided only to let the user know about the format and processing of the input data. Results might be poor if trained with sample data.
2. 'SLP' part of the framework which will classifies the spots into tumor and non-tumor will be uploaded soon.
