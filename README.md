# Deep learning based state estimation: incorporating Transformer and LSTM to Kalman Filter with EM algorithm

## Overview

- Kalman Filter requires the true parameters of the model and solves optimal state estimation recursively. Expectation Maximization (EM) algorithm is applicable for estimating the parameters of the model that are not available before Kalman filtering, which is EM-KF algorithm.
- To improve the preciseness of EM-KF algorithm, the author presents a state estimation method by combining the Long-Short Term Memory network (LSTM), Transformer and EM-KF algorithm in the framework of Encoder-Decoder in Sequence to Sequence (seq2seq). 
- Simulation on a linear mobile robot model demonstrates that the new method is more accurate.
- It is **strongly recommended to read the [slides](slides.pdf) in this repository first**, for understanding the details w.r.t. theoretical analysis and experiment in our method.

## Usage

```
python main.py
```

## Requirements

The code has been tested running under Python3, with package PyTorch, NumPy, Matplotlib, PyKalman and their dependencies installed.

## Methodology

We proposed encoder-decoder framework in seq2seq for state estimation, that state estimation is equivalent to encode and decode observation.

1. Previous works incorporating LSTM to KF, are adopting LSTM encoder and KF
   decoder. We proposed LSTM-KF adopting LSTM encoder and EM-KF decoder.
2. Before EM-KF decoder, replace LSTM encoder by Transformer encoder, we call this
   Transformer-KF.
3. Integrating Transformer and LSTM, we call this TL-KF.

Integrating Transformer and LSTM to encode observation before filtering, makes it easier for EM algorithm to estimate parameters.

## Conclusions

1. Combining Transformer and LSTM as an encoder-decoder framework for observation, can depict state more effectively, attenuate noise interference, and weaken the assumption of Markov property of states, and conditional independence of observations. This can enhance the preciseness and robustness of state estimation.
2. Transformer, based on multi-head self attention and residual connection, can capture long-term dependency, while LSTM-encoder can model time-series. TL-KF, a combination of Transformer, LSTM and EM-KF, is precise for state estimation in systems with unknown parameters.
3. Kalman smoother can ameliorate Kalman filter, but in TL-KF, filtering is precise enough. Therefore, after offline training for parameter estimation, KF for online estimation can be adopted.

