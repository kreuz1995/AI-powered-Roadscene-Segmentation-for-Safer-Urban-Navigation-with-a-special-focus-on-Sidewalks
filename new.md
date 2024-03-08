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

  **1. Dataset 1**: The first dataset is based on the paper cited as "Na, G. S., & Chang, H. (2022). A public database of thermoelectric materials and system-identified material representation for data-driven discovery. npj Computational Materials, 8(1), 214.". It consists of 2 tables: **ESTM** and **preds_sxgb**. The ESTM dataset covers 880 unique thermoelectric materials and provides five experimentally measured thermoelectric properties: Seebeck coefficient, electrical conductivity, thermal conductivity, power factor, and figure of merit (ZT). A machine learning approach is devised through which the ZT values for different materials from unexplored material groups were predicted and R2-score from 0.13 to 0.71 in an extrapolation problem. which is recorded in preds_sxgb.<br><br>Coming to the data engiineering pipeline, the earlier version had 5 sheets, but we removed the "results_extrapol_0.xlsx", "results_extrapol_1.xlsx" & "results_extrapol_2.xlsx", because they were the truncated versions of "preds_sxgb.xlsx" and no new data. So, we have kept only "estm.xlsx" & "preds_sxgb.xlsx" & removed the other redundant datasets. There are a total of 5205 rows in both the datasets, before and after the pipeline is executed. Other changes that we made were changing the datatype of various columns. Previously all of them were text, whereas, we changed them to appropriate datatypes, as we saw fit. We have also introduced constraints on specific columns like temperature & Thermal conductivity, where we mentioned the range of values that the columns can allow. <br><br>We have also used Transform blocks on the Reference column to standardize the DOI URLs, so that all of them start with 10.xx.xxx instead of http://xxx or https://xxxx.

  **2. Dataset 2**: The datasets used in this data engineering pipeline were taken from the paper Sierepeklis, O., & Cole, J. M. (2022). A thermoelectric materials database auto-generated from the scientific literature using ChemDataExtractor. Scientific Data, 9(1), 648. This is the first 
automatically generated database of thermoelectric materials and their properties from existing literature. The database was evaluated to have a precision of 82.25%. Here we have 2 datasets, one is the Main dataset
that contains all the properties of the thermoelectric-materials: "main_tedb.csv" and the other one contains the machine learning predictions of the ZT, Termal conductivity, Seebeck coefcient, Electrical conductivity
& Power factor.<br><br> The inf_tedb.csv has 18509 columns before this data pipeline & 18336 columns after the execution of the pipeline. The main_tedb.csv database had 19707 before & 14617 rows after the execution. The main difference in the count of the rows is because a lot of the values in the main_tedb dataset was misplaced in different columns. As a result, after standardisation, all those records were filtered out. For example, temperature value was in ZT column and vice-versa. The changes that we introduced are changing a lot of dataypes, from text to appropriate ones. All the datatypes were text before. Secondly, we introduced a lot of constraints according to the theory of the paper, like we created an allow list for the models as well as model types, the temperature and the Access types. And we also standardized the doi format in this pipeline: 10.xxxx/yyyy.

**3. Dataset 3**: This pipeline has been designed in accordance to the paper cited as "Kumar, P., Kabra, S., & Cole, J. M. (2022). auto-generating databases of Yield Strength and Grain Size using ChemDataExtractor.<br><br>The automatically-extracted data were organised into 4 databases:<br>
 * Database 1: YieldStrength_Database includes data about yield strength.
 * Database 2: GrainSize_Database includes data about grain size.
 * Database 3: EngineeringReady_YieldStrength_Database.
 * Database 4: Combined_YieldStrength_GrainSize_Database has a subset of columns from DB1 and DB2.
                    
                    
