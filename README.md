## ITensorCPD-Experiments

## Overview
This repository contains scripts and experiments for the ITensorCPD project 

## Dataset
The dataset is not included in this repository. To run the experiments you need to download the dataset and place it in the following folder:

## Well dataset
This project uses the **Well dataset**. It contains a large-scale collection of machine learning datasets containing numerical simulations of a wide variety of spatiotemporal physical systems.

```bibtex
@article{ohana2024well,
  title={The well: a large-scale collection of diverse physics simulations for machine learning},
  author={Ohana, Ruben and McCabe, Michael and Meyer, Lucas and Morel, Rudy and Agocs, Fruzsina and Beneitez, Miguel and Berger, Marsha and Burkhart, Blakesly and Dalziel, Stuart and Fielding, Drummond and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={44989--45037},
  year={2024}
}

To install, follow the instructions at:  
[https://polymathic-ai.org/the_well/](https://polymathic-ai.org/the_well/)

Then you can download the two dataset "active_matter" and "gray_scott_reaction_diffusion"

```bash
the-well-download --base-path path/to/repo/test/datasets --dataset active_matter --split train
the-well-download --base-path path/to/repo/test/datasets --dataset gray_scott_reaction_diffusion --split train
