# Description 

The data/ folder contains all datasets used throughout the project’s end-to-end machine learning workflow.
It includes raw data, cleaned datasets, engineering-ready files, and the final CSVs used during model training and testing.

This directory keeps the project’s data pipeline organized and ensures full transparency from *** raw → cleaned → processed → training-ready.***

![S3](../images/mflow_s3.png)

# Data Storage Strategy & Version Control

Although the most recent CSV datasets are stored directly in GitHub for convenience and transparency, we also maintain a separate data versioning approach using AWS S3.

This ensures:

- Long-term backups of all datasets

- Proper versioning of CSVs as they evolve

- Storage for large files that do not belong in GitHub (e.g., .pkl model objects)

- Reproducibility across all team members and graders


# S3 Data Storage & Backups

All final datasets, intermediate processed files, and large artifact files are also backed up in an S3 bucket:

👉 [Data_S3_Bucket](https://dagshub.com/jkhan2211/DiseaseFeatureClassifiers/src/main/s3:/DiseaseFeatureClassifiers)

From this bucket, we store:

- Raw and cleaned CSVs

- Dataset versions for reproducibility

- Large .pkl model experiment files (too large for GitHub)

- Saved model artifacts from training runs

This ensures the project maintains a reliable, scalable, and secure data management workflow.