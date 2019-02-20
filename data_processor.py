import math
import numpy as np
import pandas as pd

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None
        self.denormalization_vals = []

    def get_test_data(self, seq_len, normalise, day_pred=1):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len - day_pred):
            data_windows.append(self.data_test[i:i+seq_len + day_pred - 1])
            self.denormalization_vals.append(data_windows[i][0][0])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-day_pred]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len, normalise, day_pred=1):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len - day_pred):
            # print("%i" %i)
            x, y = self._next_window(i, seq_len, normalise, day_pred)
            # print(x.shape)
            # print(y.shape)
            # assert False
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise, day_pred=1):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len - day_pred):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len - day_pred):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                # print("%i" % b)
                x, y = self._next_window(i, seq_len, normalise, day_pred)
                # print(x.shape)
                # print(y.shape)
                # assert False
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            # print(np.array(x_batch).shape)
            # assert False
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise, day_pred=1):
        '''Generates the next data window from the given index location i'''
        # print("Running next_window")
        window = self.data_train[i:i+seq_len + day_pred - 1]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-day_pred] #this is the previous 50 data points (including all features
        y = window[-1, [0]] #starting from the 51st sample point
        # print(window.shape)
        # print(x.shape)
        # print(y.shape)
        # assert False
        # print(y)
        # print(x)
        # assert False
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                if col_i in [0,1,2,3]: #price data
                    normalised_col = [((float(window[p, col_i]) / float(window[0, col_i]) - 1)) for p in range(len(window[:, col_i]))]
                    #normalised_col = [((float(p) - float(window[0, col_i]))/np.max(window[:, col_i])) for p in window[:, col_i]]
                    #normalised_col = [((float(p) )) for p in window[:, col_i]]
                    normalised_window.append(normalised_col)
                elif col_i == 4: #volume data
                    normalised_col = [(((float(p) - np.average(self.data_train[:, col_i])) / np.max(self.data_train[:, col_i]))) for p in window[:, col_i]]
                    normalised_window.append(normalised_col)
                elif col_i in [5, 6]: #MACD data
                    normalised_col = [(float(p) / np.max(self.data_train[:, col_i])) for p in window[:, col_i]]
                    normalised_window.append(normalised_col)
                elif col_i == 7: #RSI data
                    normalised_col = [((float(p) - 50) / 50) for p in window[:, col_i]]
                    normalised_window.append(normalised_col)
                elif col_i in [8,9]:
                    # normalised_col = [((float(p) - float(window[0, col_i])) / np.max(window[:, col_i])) for p in
                    #                   window[:, col_i]]
                    normalised_col = [((float(p) / float(window[0, col_i]) - 1)) for p in window[:, col_i]]
                    normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)