[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/xnB1OI0j)
# DS542 Deep Learning for Data Science -- Spring 2025 Midterm Challenge

## Overview

This repository contains the code for the midterm challenge of the course DS542 Deep Learning for Data Science.

The challenge is in three parts:
1. **Part 1 -- Simple CNN:** Define a relatively simple CNN model and train it on the CIFAR-100 dataset to
    get a complete pipeline and establish baseline performance.
2. **Part 2 -- More Sophisticated CNN Models:** Use a more sophisticated model, including predefined models from torchvision
   to train and evaluate on CIFAR-100.
3. **Part 3 -- Transfer Learning from a Pretrained Model:** Pretrain a model, or use one of the pretrained models from torchvision, and
   fine-tune it on CIFAR-100. Try to beat the best benchmark performance on the leaderboard.

All your models should be built from linear and convoultional layers, as well as pooling, etc. We haven't covered Transformers yet,
so don't use Transformer architectures.

There is example starter template in `starter_code.py` which includes evaluation and submissions generation code. We suggest
you copy and revise that code for each of the three parts above. In other words, your repo should have (at least) the three
files, one for each part described above, as well as any supporting files.

For each part, submit the results to the Kaggle [leaderboard](https://www.kaggle.com/t/3551aa4f562f4b79b93204b11ae640b4).

Your best result needs beat the best benchmark performance of 0.397 on the leaderboard.

Use Weights and Biases experiment tracking tool to track your experiments. Create
a free student account at [WandB](https://wandb.ai). The starter code is already
instrumented for WandB, so it will start tracking experiments right away.

You can write your report using the WandB Reports UI if you wish.

## Data

You will start with the CIFAR-100 dataset, which is downloaded and installed the
first time your successfully run the sample code, `starter_code.py`.

It should install into the `data/cifar-100-python` directory.

We also have the challenge images in `data/ood-test` directory. Those are used
to make predictions on the challenge images with your model and produce the 
submission file.

## Setup

Fork this repository to your GitHub account and clone it to your local machine
or to the SCC.

On MacOS and Linux, you can create a virtual environment and install the
dependencies with the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Report

In addition to the code, we require a short report that includes:

* **AI Disclosure:** You are allowed to use AI assistance for this assignment, but you are required to:
    * Explain how you used AI, e.g. Copilot, Cursor, ChatGPT, etc.
    * Enumerate in detail which parts of the code were written by you and which were written with AI assistance.
    * Have detailed code comments explaining what every part of your code does. This can be in the codebase itself.
    * **Failure to disclose how you used AI may result in a score of 0 for the assignment.**
* **Model Description:** Detailed explanation of the chosen architecture, including justifications for design choices.
* **Hyperparameter Tuning:** Description of the hyperparameter search process and the final chosen values.
* **Regularization Techniques:** Explanation of the regularization methods used and their impact.
* **Data Augmentation Strategy:** Description of the data augmentation techniques used.
* **Results Analysis:** Discussion of the results, including strengths and weaknesses of the model, and potential areas for improvement.
* **Experiment Tracking Summary:**  Include screenshots or summaries from the experiment tracking tool.
  You can use the WandB Reports UI to create a report as well.

## Grading Rubric

The grading rubric is as follows:

* **Code Quality (30%):**
    * Correctness of implementation.
    * Readability and organization of code.
    * Use of PyTorch best practices.
    * Efficiency of data loading and processing.
* **Model Performance (40%):**
    * Performance on the primary evaluation metric.
    * Ranking on the leaderboard of at least above 0.397
    * List the leaderboard performance, identifier and username for the best scores for each of the three parts of the assignment.
* **Experiment Tracking and Report (30%):**
    * Comprehensive AI disclosure statement.
    * Completeness and clarity of the report.
    * Thoroughness of experiment tracking.
    * Justification of design choices.
    * Analysis of results.
    * Ablation study (if included).

## Bonus Points (Optional)

The top 10 students on the Private leaderboard will receive bonus points.


# Midterm Report

## AI Disclosure

I used ChatGPT-4o to initiate SimpleCNN models and downloading default parameters for ResNet18. 
The `SimpleCNN.__init__` is mainly generated by ChatGPT, afterwards I modified the size of input and output channels to make it smaller. 

The following code is generated by ChatGPT to resolve an Error when I tried to download the ResNet18 model and its pre-trained weights:
```
# safely download the pretrained model weights
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
from torchvision.models import ResNet18_Weights
```

## Repository Structure

```
.
├── code/                          # Main codebase: training, evaluation, and utilities
│   ├── eval_cifar100.py          # Evaluation script for CIFAR-100 dataset
│   ├── eval_ood.py               # Evaluation script for out-of-distribution data (test set)
│   ├── package_tests.ipynb       # Tests of package behavior
│   ├── starter_code.py           # main functions. Training & testing codes
│   └── utils.py                  # Helper functions: finding optimal batch sizes
│
├── models/                        # Trained model checkpoints
│   ├── best_model.pth             # The same as model_epoch_20.pth
│   ├── model_epoch_5.pth
│   └── ... (other saved epochs)
│
├── submissions/                   # Submission CSV files for evaluation
│   ├── sample_submission.csv
│   ├── submission_ood.csv         # The same as the 3rd submission (with best_model)
│   ├── submission_ood_1.csv       # 3 Submissions on Kaggle
│   ├── submission_ood_2.csv
│   └── submission_ood_3.csv
│
├── wandb_analysis/                # W&B visual analysis results
│   ├── Wandb_ResNet_1.png
│   └── Wandb_ResNet_2.png
│
├── .github/
├── .gitignore
├── requirements.txt              # List of Python dependencies
├── README.md                     # Overview and documentation, midterm report
└── midterm_challenge.code-workspace
```


## Model Description

For SimpleRNN, I wish to keep it "simple" so I did not put much parameters on it. ChatGPT gave me a network consisting of 2 convolutional layers and 2 fully-connected layers, and I think that was fair. Since this is a network processing images, adding fully-connected layers may be inefficient; I was thinking about adding another convolutional layer, but I noticed that the image size is small (32*32). I realized that adding another layer may break the characteristics in this simple CNN model, so I decided not to do it. Instead, I trimmed the model by decreasing the layer sizes of the convolutional layer, hoping it achieves similar accuracy while maintaining lighter size. 

I chose ResNet18 as the pretrained model also because it is light. Our project is also aiming for a light model, so I would prefer training a lighter model for prediction. 

One problem with the light model is that when it comes to fine-tuning the model, I have fewer chances to do; I decided to freeze all parameters (layers) but only the last layer of the ResNet18. 

## Hyperparameter Tuning

For `device` and `num_workers`: I am using an MacBook Pro (M2 Pro chip) which has 6 performance cores and 4 efficiency cores, so `num_workers = 6` should be suitable. The torch version and my macOS and chip also supports `mps`, so I checked that `device = "mps"` for training and testing models. These significantly accelerates the model training. 

For SimpleCNN, the code suggest me to use the biggest `batch_size = 512`, which my computer is capable to hold. For ResNet18, since I have to upscale the image to 224*224, I halved the batch size so `batch_size = 256`. In training steps for ResNet18, I noticed that both memory usage and pressure are around 80%, suggesting that they extensively used my computer (unified) memories and I should not increase the batch size. 

`epochs` and `learning_rate`: I was focusing on increasing `epochs` (from 5 up to 25) and modifying the structure of the network, so I chose to keep `learning_rate = 0.1`. Since the ResNet18 model seems got stuck on performance, decreasing learning rate may be helpful; for example, letting `learning_rate = 0.05` and further increasing the `epochs` to around 40 to 50. 

## Regularization Techniques

I did not use any regularization techniques for the first two parts. For Part 3 I used L2 regularization with `weight_decay = 1e-4` (which is $\lambda$). I would prefer using dropouts other than L2 regularizations if I want regularization on SimpleCNN and ResNet18. 

## Data Augmentation Strategy

I did not use any data augmentation. 

## Results Analysis

SimpleCNN: the performance (accuracy) is around 1%, which is terrible. It is nearly the performance of random guessing, as there are 100 classes. I think I would not embrace the need for a "light model" in designing that simple model. Besides, adding a convolutional layer or changing the padding and stride on convolutional layers may be helpful. 

ResNet18: Again, the strength of the model is that it is light. Increasing `epochs` number may be helpful, but since it turns out that it got stuck on accuracy right now, a better improvement may be decreasing the learning rate and increasing epochs. I don't think adjusting batch size would be really helpful. Though its weakness is also that it's light, indicating that there are smaller spaces for fine-tuning. Therefore, if we want a higher accuracy (e.g. >60%), I would like to choose a more complex model such as ResNet50. However, I believe that we can exploit the potential of ResNet18 to beat the 40% accuracy threshold. 

## Experiment Tracking Summary

[Link to Wandb results of ResNet18](https://wandb.ai/yg779xw-boston-university/-sp25-ds542-challenge/runs/mps9lbc8?nw=nwuseryg779xw)

If the link does not work, See `Wandb_ResNet_1.png` and `Wandb_ResNet_2.png` in the directory. 
