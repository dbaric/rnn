# Recurrent Neural Networks (RNNs) – Working with Sequential Data

Unlike feedforward networks, which treat each input independently, a **recurrent neural network (RNN)** maintains a **hidden state** that is updated at each time step as it processes the sequence. That state carries information from earlier steps, so the network can use context and temporal dependencies when making predictions.

This repository gives a short theoretical background and then focuses on **practical notebook examples** using RNNs on real data: next-value prediction, imputation, sequence labeling, sequence classification, and anomaly detection.

Data is scraped from publicly available ZSE (Zagreb Stock Exchange) [endpoints](https://zse.hr/hr/indeks/365?isin=HRZB00ICBEX6&tab=index_history).

A detailed seminar write-up (**in Croatian**) is in [SEMINAR.md](SEMINAR.md).

> This is a university assignment - the code is (partly) vibe coded, just experimenting and learning concepts, not production-ready.

## Highlights

**Predicting next value** — forecasting the next value in the series from historical index data.

![Next value prediction](figures/predict-next-value.png)

**Imputing** — filling missing data points for non-trading days.

![Regional accident profiles](figures/imputation.png)

**Sequence labeling** — assessing whether a period is a "work run".

![Sequence labelling](figures/sequence-labeling.png)
