import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

np.random.seed(1)
torch.manual_seed(1)


class LSTM_apm(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, max_layer_size=30):
        super(LSTM_apm, self).__init__()

        # Defining hidden_size
        self.hidden_dim = hidden_dim

        # Defining the cells that are in the layer
        self.lstm_cells = {}
        for i in range(1, max_layer_size + 1):
            self.lstm_cells[i] = nn.LSTMCell(input_size, hidden_dim)
        # Fully connected layer
        self.out_fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, timestamps):
        batch_size = x.size(0)
        hidden = [torch.zeros(1, self.hidden_dim)]
        c_out = torch.zeros(1, self.hidden_dim)
        for i in range(1, batch_size + 1):
            x_cell = torch.unsqueeze(x[i - 1], dim=0)
            h_out, c_out = self.lstm_cells[i].forward(x_cell, (hidden[i - 1], c_out))
            hidden.append(h_out)
            # Decay with increasing time difference
            if i != batch_size:  # Final one does not need to update c_out
                c_out = 2.0 * torch.sigmoid(- (timestamps[i] - timestamps[i - 1]) / 60.0) * c_out
        hid_mat = torch.stack(hidden[1:], dim=0).squeeze()
        return torch.sigmoid(self.out_fc.forward(hid_mat))


class activity_prediction_model:
    """Activity prediction
    
        You may add extra keyword arguments for any function, but they must have default values 
        set so the code is callable with no arguments specified.
    
    """

    def __init__(self, lr=0.01, reg=0.01, hidden_size=100):
        self.lr = lr
        self.reg = reg
        self.lstm_model: LSTM_apm = LSTM_apm(211, 5, hidden_size)
        self.means = None
        self.stds = None

    def fit(self, df, mode='train', train_batch_per_user=3):
        """Train the model using the given Pandas dataframe df as input. The dataframe
        has a hierarchical index where the outer index (ID) is over individuals,
        and the inner index (Time) is over time points. Many features are available.
        There are five binary label columns: 
        
        ['label:LYING_DOWN', 'label:SITTING', 'label:FIX_walking', 'label:TALKING', 'label:OR_standing']

        The dataframe contains both missing feature values and missing label values
        indicated by nans. Your goal is to design and fit a probabilistic forecasting 
        model that when given a dataframe containing a sequence of incomplete observations 
        and a time stamp t, outputs the probability that each label is active (e.g., 
        equal to 1) at time t.

        Arguments:
            df: A Pandas data frame containing the feature and label data
        """
        self.load_model()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.lr, weight_decay=self.reg)
        if mode is 'train':
            def train_callback(history, pred_record):
                optimizer.zero_grad()
                predictions = self.cal_class_prob(history, pred_record[0])
                loss = self.cross_log_loss(pred_record, predictions)
                loss.backward()
                optimizer.step()
                return predictions.data.numpy(), loss.item()

            train_loss_points = self.iteration_construct(df, train_callback, train_batch_per_user)
            print('Average train loss=', sum(train_loss_points) / len(train_loss_points))
            return train_loss_points
        else:
            def val_callback(history, pred_record):
                with torch.no_grad():
                    predictions = self.cal_class_prob(history, pred_record[0])
                    loss = self.cross_log_loss(pred_record, predictions)
                    return predictions.data.numpy(), loss.item()

            val_loss_points = self.iteration_construct(df, val_callback, train_batch_per_user)
            print('Average validation loss=', sum(val_loss_points) / len(val_loss_points))
            return val_loss_points

    def iteration_construct(self, df, iteration_fn, num_batch_per_user, debug_low=False, debug_high=True):
        loss_epochs = []
        for user, user_frame in df.groupby(level=0):
            batch_sizes = np.random.choice(range(2, min(user_frame.shape[0], 32)), num_batch_per_user)
            loss_user = 0.0
            for batch_size in batch_sizes:
                loss_epoch = 0.0
                # Include 1 sample for forecast and rest for observation
                num_iters = user_frame.shape[0] // batch_size
                for iter_num in range(num_iters):  # 1  epoch
                    batch = sample(user_frame, iter_num, batch_size)
                    history, pred_record = batch.iloc[:-1], batch.iloc[-1]
                    predictions, loss = iteration_fn(history, pred_record)
                    loss_epoch = loss_epoch + loss
                    if debug_low is True:
                        print('user %d, batch_size %d, iter %d/%d loss=%f predictions=' % (
                            user, batch_size, iter_num, num_iters, loss), predictions)
                avg_loss_epoch = loss_epoch / num_iters
                loss_epochs.append(avg_loss_epoch)
                loss_user += avg_loss_epoch
                if debug_high is True:
                    print(
                        'user %d, batch_size %d, iters = %d, Average Loss:%f' % (
                            user, batch_size, num_iters, avg_loss_epoch))
            if debug_high is True:
                print(
                    'user %d, batches %d,  Average Loss:%f' % (
                        user, num_batch_per_user, loss_user / num_batch_per_user))
        return loss_epochs

    def cross_log_loss(self, pred_record, predictions):
        return -torch.sum(torch.tensor(pred_record[-5:]) * torch.log(predictions + 1e-8))

    def get_params(self):
        return {
            'params': self.lstm_model.state_dict(),
            'mean': self.means,
            'std': self.stds
        }

    def set_params(self, params):
        self.lstm_model.load_state_dict(params['params'])
        self.means = params['mean']
        self.stds = params['std']
        self.lstm_model.eval()

    def cal_class_prob(self, df, t):
        label_data = torch.tensor(df[df.columns[df.columns.str.startswith('label')]].reset_index(drop=True).values)
        timestamps = torch.tensor(df['timestamp'].reset_index(drop=True))
        x = torch.tensor(df[get_ex_cols(df, [])].reset_index(drop=True).values)  # nxd
        # Add delta time from prediction as a feature
        p_pos = self.lstm_model.forward(x.float(), timestamps)
        p_all = (p_pos * (label_data == 1.0) + (torch.tensor(1.0) - p_pos) * (label_data == 0.0))
        weights = torch.tensor(2.0) * torch.sigmoid(- (torch.tensor(t) - timestamps) / 60.0)
        weights = torch.unsqueeze(weights, dim=1)
        return torch.sum(p_all * weights, dim=0) / (torch.sum(weights) + 1e-8)  # n.c

    def forecast(self, df, t, prepro=True):
        """Given the feature data and labels in the dataframe df, output the log probability
        that each labels is active (e.g., equal to 1) at time t. Note that df may contain
        missing label and/or feature values. Assume that the rows in df are in time order,
        and that all rows are for data before time t for a single individual. Any number of
        rows of data may be provided as input, including just one row. Further, the gaps
        between timestamps for successive rows may not be uniform. t can also be any time
        after the last observation in df. There are five labels to predict:

        ['label:LYING_DOWN', 'label:SITTING', 'label:FIX_walking', 'label:TALKING', 'label:OR_standing']

        Arguments:
            df: a Pandas data frame containing the feature and label data for multiple time
            points before time t for a single individual.
            t: a unix timestamp indicating the time to issue a forecast for

        Returns:
            pred: a python dictionary containing the predicted log probability that each label is
            active (e.g., equal to 1) at time t. The keys used in the dictionary are the label
            column names listed above. The values are the corresponding log probabilities.

        """
        with torch.no_grad():
            if prepro is True:
                df, _, _ = preprocess(df, self.means, self.stds, drop=False)
            class_probs = self.cal_class_prob(df, t)
            return {'label:LYING_DOWN': math.log(class_probs[0].item() + 1e-8),
                    'label:SITTING': math.log(class_probs[1].item() + 1e-8),
                    'label:FIX_walking': math.log(class_probs[2].item() + 1e-8),
                    'label:TALKING': math.log(class_probs[3].item() + 1e-8),
                    'label:OR_standing': math.log(class_probs[4].item() + 1e-8)}

    def save_model(self):
        """A function that saves the parameters for your model. You may save your model parameters
           in any format. You must upload your code and your saved model parameters to Gradescope.
           Your model must be loadable using the load_model() function on Gradescope. Note:
           your code will be loaded as a module from the parent directory of the code directory using:
           from code.activity_prediction import activity_prediction_model. You need to take this into
           account when developing a method to save/load your model.

        Arguments:
            None
        """
        torch.save(self.get_params(), 'code/lstm_model.pt')

    def load_model(self):
        """A function that loads parameters for your model, as created using save_model().
           You may save your model parameters in any format. You must upload your code and your
           saved model parameters to Gradescope. Following a call to load_model(), forecast()
           must also be runnable with the loaded parameters. Note: your code will be loaded as
           a module from the parent directory of the code directory using:
           from code.activity_prediction import activity_prediction_model. You need to take this into
           account when developing a method to save/load your model

        Arguments:
            None
        """
        if os.stat('code/lstm_model.pt').st_size == 0:
            return
        params = torch.load('code/lstm_model.pt')
        self.set_params(params)


def preprocess(df, means=None, stds=None, drop=True):
    # Columns that are redundant or have extremely low data
    df = df.drop(columns=['discrete:wifi_status:missing', 'discrete:wifi_status:is_reachable_via_wwan',
                          'discrete:wifi_status:missing', 'discrete:wifi_status:is_not_reachable',
                          'discrete:on_the_phone:missing', 'discrete:on_the_phone:is_True',
                          'discrete:on_the_phone:is_False', 'discrete:battery_state:is_not_charging',
                          'discrete:battery_state:is_unknown', 'discrete:battery_state:missing',
                          'discrete:battery_plugged:is_wireless', 'discrete:app_state:is_inactive',
                          'lf_measurements:proximity', 'lf_measurements:relative_humidity',
                          'lf_measurements:temperature_ambient'], errors='ignore')
    # Standard Scale the real variabled features
    cols_to_n = get_ex_cols(df, ['discrete'])
    if means is not None and stds is not None:
        df[cols_to_n] = (df[cols_to_n] - means) / stds
    else:
        scalar = StandardScaler()
        df[cols_to_n] = scalar.fit_transform(df[cols_to_n])
        means = scalar.mean_, scalar.scale_
    # Imputed missing sensor values to zero.
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)
    # All entries with null labels- ignore.
    if drop is True:
        df[cols_to_n] = imp_mean.fit_transform(df[cols_to_n])
        df.dropna(inplace=True)
    else:
        df[:] = imp_mean.fit_transform(df[:])
    return df, means, stds


def get_ex_cols(df, prefixes, colname='timestamp'):
    cols = df.columns.str.startswith('label')
    for pre in prefixes:
        cols = cols + df.columns.str.startswith(pre)
    exclude_cols = df.columns[cols]
    exclude_cols = exclude_cols.insert(0, colname).values
    return df.columns.values[~np.isin(df.columns.values, exclude_cols)]


def sample(df, iter_num, batch_size):
    timeIndexes = df.index.get_level_values(1)
    return df.iloc[iter_num * batch_size:min((iter_num + 1) * batch_size, timeIndexes.shape[0])]


def avg_loss_for_test(df_test, model: activity_prediction_model):
    def bce_loss(pred_record, predictions):
        return -torch.sum(torch.tensor(pred_record[-5:]) * (torch.tensor(list(predictions.values())) + 1e-8))

    loss_epochs = []
    for user, user_frame in df_test.groupby(level=0):
        batch_sizes = [2]
        loss_user = 0.0
        for batch_size in batch_sizes:
            loss_epoch = 0.0
            # Include 1 sample for forecast and rest for observation
            num_iters = user_frame.shape[0] // batch_size
            if num_iters == 0:
                continue
            for iter_num in range(num_iters):  # 1  epoch
                batch = sample(user_frame, iter_num, batch_size)
                history, pred_record = batch.iloc[:-1], batch.iloc[-1]
                predictions = model.forecast(history, pred_record[0])
                loss = bce_loss(pred_record, predictions)
                if loss!=loss and predictions==predictions:
                    continue
                print('user %d, batch_size %d, iter %d/%d loss=%f predictions=' % (
                    user, batch_size, iter_num, num_iters, loss.item()), predictions)
                loss_epoch = loss_epoch + loss
            avg_loss_epoch = loss_epoch / num_iters
            loss_epochs.append(avg_loss_epoch)
            loss_user += avg_loss_epoch
            print('user %d, batch_size %d, iters = %d, Average Loss:%f' % (
                user, batch_size, num_iters, avg_loss_epoch))
        print('user %d, Average Loss:%f' % (user, loss_user / 4))
    return sum(loss_epochs) / len(loss_epochs)


def split_train_test(df, choice='random', split=None):
    idx = pd.IndexSlice
    n_users = np.unique(df.index.get_level_values(0).values).shape[0]
    test = n_users // 5
    if choice is 'random':
        test_inds = np.random.choice(np.arange(n_users), size=test)
    else:
        test_inds = np.arange(test * split, test * (split + 1))
    train_mask = np.array([False] * n_users)
    train_mask[test_inds] = True
    train_mask = ~train_mask
    train_inds = np.arange(n_users)[train_mask]
    test_df = df.loc[idx[test_inds, :], idx[:]]
    train_df = df.loc[idx[train_inds, :], idx[:]]
    return train_df, test_df


def main():
    # Load the training data set
    df = pd.read_pickle("data/train_data.pkl", compression='gzip')

    df_training, df_test = split_train_test(df)
    df_training, means, stds = preprocess(df_training)

    # Account for correction
    params = torch.load('code/lstm_model.pt')
    if 'params' not in params:
        torch.save(
            {
                'params': params,
                'mean': means,
                'std': stds
            }, 'code/lstm_model.pt')
        params = torch.load('code/lstm_model.pt')
        assert 'params' in params

    print(np.unique(df_test.index.get_level_values(0).values))

    do_hyperparam = False
    if do_hyperparam is True:
        # Hyperparamter search
        lr = [0.0001, 0.001, 0.01, 0.1]
        reg = [0.01, 0.1]
        hid_dim = [50, 100]

        hyperparams = [(l, r, h) for l in lr for r in reg for h in hid_dim]
        apm = None
        best_loss = 166777554333.0
        best_params = None
        for params in hyperparams:
            avg_val_loss = 0.0
            # Split for 15 train and 5 val users
            model = None
            for split in range(4):
                train_df, val_df = split_train_test(df_training, split=split, choice='sequence')

                model = activity_prediction_model(lr=params[0], reg=params[1], hidden_size=params[2])

                model.fit(train_df, train_batch_per_user=3)

                val_loss_list = model.fit(df, mode='val', train_batch_per_user=3)
                avg_val_loss += sum(val_loss_list) / len(val_loss_list)
            avg_val_loss = avg_val_loss / 5
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                apm = model
                best_params = params
            print('Average loss for hyperparam ', params, 'is ', avg_val_loss)
        print('BEST: params ', best_params, ' . Min loss: ', best_loss)

        apm.save_model()
        apm.load_model()

        # finally on the entire data
        model = activity_prediction_model(lr=best_params[0], reg=best_params[1], hidden_size=best_params[2])
        model.fit(df_training, train_batch_per_user=10)

        # Test result
        apm.save_model()

    # Test result
    apm = activity_prediction_model()
    apm.load_model()

    # Get a sample of data
    example = df.loc[[0]][:10]

    # Get a timestamp 5 minutes past the end of the example
    t = example["timestamp"][-1] + 5 * 60

    # Compute a forecast
    f = apm.forecast(example, t)
    print(f)

    # Test on held out set
    print('Total average test loss', avg_loss_for_test(df_test, apm))


if __name__ == '__main__':
    main()
