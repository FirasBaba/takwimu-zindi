# AI4D Takwimu Lab - Machine Translation Challenge - 3rd place

## Description
This repository contains the 3rd place solution for the AI4D Takwimu Lab - Machine Translation Challenge competition. The challenge focused on translating text from French into Fongbe or Ewe, two Niger–Congo languages that are part of the Gbe cluster. The competition aimed to address the communication barriers between local language speakers and modern society in West Africa.

**Competition Link:** [AI4D Takwimu Lab - Machine Translation Challenge](https://zindi.africa/competitions/ai4d-takwimu-lab-machine-translation-challenge)

## Problem Statement
Ewe and Fongbe are closely related tonal languages spoken in Togo, southeastern Ghana, and Benin. Despite being core to the economic and social life of major West African capital cities, they are rarely written, hindering access to critical facilities like education, banking, and healthcare for non-French/English speakers. The competition sought to develop a machine translation system to convert French text into Fongbe or Ewe, with the challenge of working efficiently with limited available data.

## Repository Files
- `config.py`: Configurations of the pipeline.
- `dataset.py`: PyTorch class for the customized dataset.
- `models.py`: Implementation of the T5 model.
- `predict.py`: Script to prepare the submission file.
- `prepare_data.py`: Handles data preparation, preprocessing, incorporates external data and split the cross-validation folds.
- `pretraining.py`: Pretrains the model on publicly available external data.
- `train.py`: Trains the final model using the competition data.
- `validation.py`: Measures the ROUGE score on the validation set.

## How to Use
1. **Clone the Repository:**
```
git clone https://github.com/FirasBaba/takwimu-zindi
```
2. **Navigate to the Repository:**
```
cd takwimu-zindi/working
```
3. **Run the Script:**
```
sh execute.sh
```

## Acknowledgments
- Special thanks to [Zindi](https://zindi.africa/) for hosting the competition.
- Acknowledgment to Takwimu Lab association and AI4D-Africa for providing the data.

Feel free to explore the code and contribute to the development of machine translation models for low-resourced West African languages. This initiative aims to break down communication barriers and enhance access to essential services in the region.
