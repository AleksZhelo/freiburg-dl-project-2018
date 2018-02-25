import numpy as np

class StandardScaler:
    def __init__(self):
        self.variables = {}

    def fit(self, dict_list):
        n = len(dict_list)
        for key in dict_list[0]:
            values = [dict_list[i][key] for i in range(n)]
            self.variables[key] = (np.mean(values), np.std(values))
    
    def transform(self, dict_list):
        n = len(dict_list)
        for i in range(n):
            for key in self.variables:
                dict_list[i][key] = (dict_list[i][key] - self.variables[key][0]) / self.variables[key][1]
        return dict_list
    
    def fit_transform(self, dict_list):
        self.fit(dict_list)
        return self.transform(dict_list)