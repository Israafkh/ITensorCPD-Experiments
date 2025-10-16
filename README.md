## ITensorCPD-Experiments

## Overview
This repository contains scripts and experiments for the ITensorCPD project 

## Dataset
The dataset is not included in this repository. To run the experiments you need to download the dataset and place it in the following folder:

## Well dataset
The Well dataset contains a large-scale collection of machine learning datasets containing numerical simulations of a wide variety of spatiotemporal physical systems.

To install, follow the instructions at:  
[https://polymathic-ai.org/the_well/](https://polymathic-ai.org/the_well/)

Then you can download the two dataset "active_matter" and "gray_scott_reaction_diffusion"

```bash
the-well-download --base-path path/to/repo/test/datasets --dataset active_matter --split train
the-well-download --base-path path/to/repo/test/datasets --dataset gray_scott_reaction_diffusion --split train
