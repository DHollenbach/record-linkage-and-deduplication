## A machine learning approach for master patient index record linkage and deduplication

### Abstract

The research emphasised the vital role of a Master Patient Index (MPI) solution in addressing the challenges public healthcare facilities face in eliminating duplicate patient records and improving record linkage. The study recognised that traditional MPI systems may have limitations in terms of efficiency and accuracy. To address this, the study focused on utilising machine learning techniques to enhance the effectiveness of MPI systems, aiming to support the growing record linkage healthcare ecosystem.

It was essential to highlight that integrating machine learning into MPI systems is crucial for optimising their capabilities. The study aimed to improve data linking and deduplication processes within MPI systems by leveraging machine learning techniques. This emphasis on machine learning represented a significant shift towards more sophisticated and intelligent healthcare technologies. Ultimately, the goal was to ensure safe and efficient patient care, benefiting individuals and the broader healthcare industry.

This research investigated the performance of five machine learning classification algorithms (random forests, extreme gradient boosting, logistic regression, stacking ensemble, and deep multilayer perceptron) for data linkage and deduplication on four datasets. These techniques improved data linking and deduplication for use in an MPI system.

The findings demonstrate the applicability of machine learning models for effective data linkage and deduplication of electronic health records. The random forest algorithm achieved the best performance (identifying duplicates correctly) based on accuracy, F1-Score, and AUC-score for three datasets (Electronic Practice-Based Research Network (ePBRN): Acc = 99.83%, F1-score = 81.09%, AUC = 99.98%; Freely Extensible Biomedical Record Linkage (FEBRL) 3: Acc = 99.55%, F1-score = 96.29%, AUC = 99.77%; Custom-synthetic: Acc = 99.98%, F1-score = 99.18%, AUC = 99.99%). In contrast, the experimentation on the FEBRL4 dataset revealed that the Multi-Layer Perceptron Artificial Neural Network (MLP-ANN) and logistic regression algorithms outperformed the random forest algorithm. The performance results for the MLP-ANN were (FEBRL4: Acc = 99.93%, F1-score = 96.95%, AUC = 99.97%). For the logistic regression algorithm, the results were (FEBRL4: Acc = 99.99%, F1 = 96.91%, AUC = 99.97%).

In conclusion, the results of this research have significant implications for the healthcare industry, as they are expected to enhance the utilisation of MPI systems and improve their effectiveness in the record linkage healthcare ecosystem. By improving patient record linking and deduplication, healthcare providers can ensure safer and more efficient care, ultimately benefiting patients and the industry.

Keywords: Machine Learning; Master Patient Index; Record Linking; Deduplication; Synthetic Dataset;

### Publications

1. Performing record linkage and deduplication in master patient index using machine learning classifiers, SAI Computing Conference, 2025, London, United Kingdom (In Press)

For a look at the code see below:
[notebook code](https://github.com/DHollenbach/record-linkage-and-deduplication/blob/main/duplicategenerator/mict_recordlinkage_2023_24_new.ipynb)
