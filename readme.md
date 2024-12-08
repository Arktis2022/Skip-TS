### [Anomaly Detection for Medical Images Using Teacher-Student Model with Skip Connections and Multi-scale Anomaly Consistency](https://ieeexplore.ieee.org/document/10540605)

M Liu, Y Jiao, J Lu, H Chen - IEEE Transactions on Instrumentation and Measurement,
Anomaly Detection for Medical Images Using Teacher–Student Model With Skip Connections and Multiscale Anomaly Consistency, 2024

Check the paper in https://ieeexplore.ieee.org/abstract/document/10540605


Usage:
    Run main.py to train and test the Skip_ST model.

Introduction of the code:
    1. dataset.py for loading the training and testing dataset.
    2. encoder.py for the encoder of the model which is pretrained
    3. decoder.py which is the opposite of the structure of the encoder
    4. loss_function.py as Figure 2(b) showed in the paper
    5. eval_func.py for some useful functions
    6. main.py for training and testing the model
