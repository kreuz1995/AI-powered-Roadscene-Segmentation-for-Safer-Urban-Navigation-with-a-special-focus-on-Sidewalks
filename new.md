# Project Report
This project delves into the intricate realm of data engineering, specifically focusing on the enhancement and design of data pipelines using the Jayvee programming language. The primary objective is to optimize data processing for six distinct datasets within the domain of material science. Throughout this endeavor, we also explore and propose potential modifications or updates to the Jayvee language to better align with the intricate demands of this specialized field.

The foundational database for our project was constructed utilizing ChemDataExtractor version 2.0, a sophisticated 'chemistry-aware' software toolkit. This toolkit employs cutting-edge Natural Language Processing (NLP) and machine-learning techniques to extract valuable chemical data from scientific documents, forming the basis of our datasets. This integration of advanced technologies sets the stage for a comprehensive and nuanced analysis of material science datasets.

We divide our report into the following sections:

1. **Description of the datasets**:
      This section gives us an overview of the six material science datasets and our changes in the data pipelines in brief.
2. **Most common attributes**:
      We explore the prevalent data types and analyse them.
3. **Challenges faced**:
      We discuss about the most errors encountered overall on all the datasets.

## Description of the datasets

  **1. Dataset 1**: The first dataset is based on the paper cited as "Na, G. S., & Chang, H. (2022). A public database of thermoelectric materials and system-identified material representation for data-driven discovery. npj Computational Materials, 8(1), 214.".
                    It consists of 2 tables: ESTM and preds_sxgb. The ESTM dataset covers 880 unique thermoelectric materials and provides five experimentally measured thermoelectric properties: Seebeck coefficient, electrical conductivity, thermal conductivity, power factor, and figure of merit (ZT).
                    
