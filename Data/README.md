# DATA Folder

This folder contains data samples, configurations, or loading scripts for the project.


## Dataset Attribution & Citation
This project utilizes the **PTB-XL** dataset (Version 1.0.1), a large publicly available 12-lead electrocardiography (ECG) dataset. 

When referencing or building upon this dataset for academic reports, please use the official citations provided by the authors:

### Primary Publication Citation
> Wagner, P., Strodthoff, N., Bousseljot, R.-D., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7, 154. DOI: [10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)

### PhysioNet Repository Citation
> Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.1). PhysioNet. DOI: [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)

---

## Accessing and Downloading the Files
Because this dataset is large (~1.7 GB), raw files are typically downloaded locally or fetched via script rather than stored directly inside Git. You can acquire version 1.0.1 using any of the following methods:

- **Direct Web Download:** Download the ZIP file (1.7 GB) directly from the [PhysioNet PTB-XL v1.0.1 Repository](https://physionet.org/content/ptb-xl/1.0.1/).
- **PhysioNet Python Package:** Run the command `physionet download ptb-xl --version 1.0.1` in your environment.
- **Terminal Command:** Use `wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.1/` in your command line.
- **AWS Command Line Tools:** Synchronize using `aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.1/ DESTINATION`.
