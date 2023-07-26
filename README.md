# **Using Topic Modelling to Identifying Challenges Faced by Developers in Scientific Workflow Management Systems**

## Introduction

This repository contains code and data for a project that utilizes topic modeling to identify challenges faced by developers in scientific workflow management systems. The project aims to gain insights into the common issues and difficulties that developers encounter while working with scientific workflow management systems.

## Overview

In scientific research, managing complex workflows is crucial for reproducibility and efficiency. Scientific workflow management systems provide tools and infrastructure to create, execute, and analyze complex computational workflows. However, these systems can present various challenges and obstacles for developers.

The goal of this project is to use topic modeling techniques, specifically BERTopic, to analyze a dataset of developer discussions, posts, or documentation related to scientific workflow management systems. By extracting and clustering topics from the text data, we can gain valuable insights into the key challenges faced by developers in this domain.

## Topic Modeling with BERTopic

Prior to this project, an analysis has been conducted using LDA (Latent Dirichlet Allocation) techniques and unigram/bigram for text categorization. The current exploration involves using BERTopic, a powerful topic modeling library, to replicate and potentially enhance the previous analysis.

The core of this project revolves around using BERTopic, a state-of-the-art topic modeling library, to analyze and cluster textual data related to scientific workflow management systems. BERTopic leverages pre-trained language models to extract meaningful topics from the text data, providing insights into the challenges faced by developers in this domain.

## Contents

The repository includes the following:

- Previous analysis: Jupyter Notebooks containing the previous analysis using LDA and unigram/bigram techniques

- Dataset: dataset containing developer discussions relating to scientific workflow management systems scraped from Stack Overflow and GitHub Issues

- Results: intermediate and final results obtained during the topic modeling process, including topic clusters, trained models, and visualizations

## Dependencies

The project requires the following Python libraries:

- pandas
- nltk
- bertopic
- beautifulsoup4

You can create a virtual environment using the provided `requirements.txt` file:

```bash
python3 -m venv BERTopicVenv
source BERTopicVenv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the notebook interactively in your preferred IDE.
