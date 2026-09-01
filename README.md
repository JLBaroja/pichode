# PICHODE
** Pigeon Choice in Dynamic Environments**

PICHODE is a behavioral dataset for studying choice and adaptation in pigeons exposed to concurrent variable-interval (VI–VI) reinforcement schedules.

Six pigeons were exposed to concurrent VI–VI in which the two response alternatives differed in their rates of reinforcement. The experiment included both stable sessions in which the same pair of reward rates remained unchanged for the whole session, and dynamic sessions in which the relative reinforcement rates of the alternatives changed without an explicit signal to the pigeons. In total, the experiment comprises 180 daily sessions for each bird.

This repository provides a clean, reproducible reconstruction of the experimental data originally collected and analyzed as part of the **JAPE/JAPEDE (Just Another Pigeon Experiment in Dynamic Environments)** project.

The original project repository is preserved at:

https://github.com/JLBaroja/JAPE

PICHODE does not replace or modify that historical repository. Instead, it starts from the original experimental records and MED-PC control programs and reorganizes them into a clean data-focused project. The goal is to preserve the raw experimental record while providing reproducible code for constructing properly documented datasets.

## Repository organization

The raw experimental materials are divided into two stages:

* `data/raw/autoshaping/` — data and MED-PC programs from autoshaping and operant-box training.
* `data/raw/concurrent/` — data and MED-PC programs from the main concurrent VI–VI choice experiment.

Within each of those stages:

* `data/` contains the original behavioral records.
* `programs/` contains the MED-PC programs used to control the experimental pigeon chambers.

