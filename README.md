# STGAT
This repository represents a Machine Learning framework named STGAT (Spatial Transcriptomics Graph Attention Network) which can be trained on Spatial Transcriptomics (ST) data and applied on TCGA data to generate spot-level gene expression for the TCGA samples. The produced gene expression can be leveraged various downstream analysis including subtype classification, survival prediction etc. Moreover, comprehensive experimentation proves that considering only the tumor spots to compute the mean gene expression of a sample results in better performance.

## Workflow
![alt text](https://github.com/compbiolabucf/STGAT/blob/main/STGAT_overall_diagram.png)

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
      - gene_exp (Files which contains the gene expression of the spots for the ST samples ('.csv' format) with gene names as the column names)
      - clinical (Files containing the tumor label of the spots for the ST samples ('.csv' format)) 
    - TCGA
      - tcga_exp.csv (Single '.csv' file contatining the bulk gene expression of the TCGA samples with gene names as the column names)
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

The clinical tumor labels file for ST data formar should be:

|         |tumor_status|
|---------|------------|
| Spot1   |      -     |
| Spot2   |      -     |
| Spot3   |      -     |

## Training and testing the model
The SEG and GEP modules can be trained and tested by running 'main.py' file in the command line. The options can be used to modify the training and model parameters. For example,
```
python main.py --sp_dir io_data/ST/ --tcga_dir io_data/TCGA/
```
Codes for SEG, GEP and SLP modules can be found in the respective directories. 'split' directory contains the codes for splitting the patches from the ST and TCGA samples. For the TCGA samples, coordinate files clinical label files (using SLP) are also generated. 
After successfull training of SEG, GEP and SLP, the directory structure should look like the following:
- SEG (containing SEG codes)
- GEP (containing GEP codes)
- SLP (containing SLP codes)
- io_data
   - gene_names.csv
   - ST
      - wsi
      - coords
      - gene_exp
      - clinical
      - patches (Containing split ST spots)
    - TCGA
      - tcga_exp.csv
      - wsi
      - patches (containing split TCGA spots)
      - coords (containing files having the coordinates of the generated spots from TCGA samples)
      - clinical (contatining SLP generated files consisting of clinical tumor labels for the TCGA spots)
- prediction (contains predicted spot-level gene expression and mean gene expression of the tumor spots for the test TCGA samples)
- st_results (contains predicted spot-level gene expression of the test ST samples)
- saved_moodel (directory for saving modeles temporarily in each eporch)
- trained (directory for saving trained SEG, GEP and SLP modules)

## Applying trained model on TCGA samples
After completion of training of all the modules, 'apply_stgat.py' can be run to to apply STGAT framework on TCGA samples to generate corresponding spot-level gene expression profiles.
