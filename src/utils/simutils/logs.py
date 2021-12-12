import sys
import numpy as np


class BatchLogs:
    """
    A class to log data in batches for ML applications
    """
    def __init__(self):
        self.metric_dict = {}

    def append(self, metrics, data):
        if not isinstance(metrics, list):
            sys.exit('Please specify a list of metrics to log')

        for i, metric in enumerate(metrics):
            data[i] = np.array(data[i])
            if metric not in self.metric_dict:
                self.metric_dict[metric] = []
            self.metric_dict[metric].append(data[i])

    def append_tensor(self, metrics, data):
        if not isinstance(metrics, list):
            sys.exit('Please specify a list of metrics to log')

        for i, metric in enumerate(metrics):
            data[i] = np.array(data[i].detach().cpu().item())
            if metric not in self.metric_dict:
                self.metric_dict[metric] = []
            self.metric_dict[metric].append(data[i])

    def flatten(self):
        for metric in self.metric_dict:
            self.metric_dict[metric] = np.mean(self.metric_dict[metric])

    def fetch(self, metric):
        return self.metric_dict[metric]
