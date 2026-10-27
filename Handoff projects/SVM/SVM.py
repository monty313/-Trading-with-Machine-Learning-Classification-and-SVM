# Machine learning libraries
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit

# Import talib
import talib as ta

import warnings

# Import Pipeline
from sklearn.pipeline import Pipeline

# Import numpy and pandas
import pandas as pd
import numpy as np

# The strategy requires context.lookback number of days. 

def initialize(context):
    
    # Define Symbols
    context.security = symbol("BAC")

    # The lookback for the indicators
    context.lookback = 10

    # The lookback for historical data
    context.data_lookback = 990

    # The variable to store the randomised cross vaidation search
    context.rcv = None

    # The variable to store the classifier
    context.cls = None

    # Instantiate the StandardScaler
    context.ss1 = StandardScaler()

    # The variable to store the model accuracy
    context.accuracy = 0

    # The parameters for SAR
    context.acceleration = 0.2
    context.maximum = 0.2

    # The variable to store the pipeline steps
    context.steps = [('scaler', StandardScaler()), ('svc', SVC())]

    # The variable to store the Pipeline function
    context.pipeline = Pipeline(context.steps)

    # Test variables for 'c' and 'g' parameters for the SVC
    context.c = [10, 100, 1000, 10000]
    context.g = [1e-2, 1e-1, 1e0]

    # Intialize the parameters for the SVC
    context.parameters = {'svc__C': context.c,
                          'svc__gamma': context.g,
                          'svc__kernel': ['rbf']
                          }

    # The train-test split
    context.split_ratio = 0.8

    # The flag variable is used to check whether to retrain the model or not
    context.retrain_flag = True

    # Schedule the retrain_model function every week
    schedule_function(
                        retrain_model,
                        date_rule=date_rules.week_start(),
                        time_rule=time_rules.market_open()
                     )

    # Schedule the rebalance function to run every hour daily
    for i in range(1, 390, 60):
        hours = int(i/60)
        minutes = i%60
        schedule_function(
                            rebalance,
                            date_rule=date_rules.every_day(),
                            time_rule=time_rules.market_open(hours=hours, minutes=minutes)
                        )
                        

    # Schedule the end_of_day_squareoff function every day
    schedule_function(
                        end_of_day_squareoff,
                        date_rule=date_rules.every_day(),
                        time_rule=time_rules.market_close(minutes=5)
                     )


def retrain_model(context, data):
    """
        A function to retrain the classification model. This function is called by
        the schedule_function in the initialize function.
    """
    context.retrain_flag = True


def end_of_day_squareoff(context, data):
    """
        A function to square-off trades at the end of the day. This function is
        called by the schedule_function in the initialize function.
    """
    order_target_percent(context.security, 0)

    
def rebalance(context, data):
    """
        A function to rebalance the portfolio. This function is called by the
        schedule_function in the initialize function.
    """

    # Fetch lookback no. days data for the security
    Df = data.history(
        context.security,
        ['open', 'high', 'low', 'close', 'volume'],
        context.data_lookback,
        '1m')

    print(f"Getting data {context.security}")
    print(Df)

    # Calculate the RSI
    Df['RSI'] = ta.RSI(
                        np.array(Df['close'].shift(1)),
                        timeperiod=context.lookback
                      )

    # Calculate the SMA
    Df['SMA'] = Df['close'].shift(1).rolling(window=context.lookback).mean()

    # Calculate the correlation
    Df['Corr'] = Df['close'].shift(1).rolling(
        window=context.lookback).corr(Df['SMA'].shift(1))

    # Create a column by name, SAR and assign the SAR calculation to it
    Df['SAR'] = ta.SAR(
                        np.array(Df['high'].shift(1)),
                        np.array(Df['low'].shift(1)),
                        context.acceleration,
                        context.maximum
                      )

    # Create a column by name, ADX and assign the ADX calculation to it
    Df['ADX'] = ta.ADX(
                        np.array(Df['high'].shift(1)),
                        np.array(Df['low'].shift(1)),
                        np.array(Df['open']),
                        timeperiod=context.lookback
                      )

    # Create columns 'high', 'low' and 'close' with previous minute's OHLC data
    Df['Prev_High'] = Df['high'].shift(1)
    Df['Prev_Low'] = Df['low'].shift(1)
    Df['Prev_Close'] = Df['close'].shift(1)

    # Create columns 'OO' with the difference between the current minute's
    # open and last minute's open
    Df['OO'] = Df['open']-Df['open'].shift(1)

    # Create columns 'OC' with the difference between the current minute's
    # open and last minute's close
    Df['OC'] = Df['open']-Df['Prev_Close']

    # Create a column 'Ret' with the calculation of returns
    Df['Ret'] = (Df['close']-Df['close'].shift(1))/Df['close']

    # Create n columns and assign
    for i in range(1, context.lookback):
        Df['return%i' % i] = Df['Ret'].shift(i)
        

    # Change the value of 'Corr' to -1 if it is less than -1
    Df.loc[Df['Corr'] < -1, 'Corr'] = -1

    # Change the value of 'Corr' to 1 if it is greater than 1
    Df.loc[Df['Corr'] > 1, 'Corr'] = 1
    
    # print(Df.head(25)) 
    bef_len=len(Df)
    
    # Drop the NaN values
    Df = Df.dropna()
    
    #see the number of nan rows being dropped
    aft_len=len(Df)
    nan=bef_len - aft_len
    print(f'Number of rows dropped with NaN values= {nan}')
    
    # Create a variable split which is 80% of the length of the Dataframe
    t = .8
    split = int(t*len(Df))
    print(f'Value of split= {split}')
    
    # Create a column by name, 'Signal' and initialize with 0
    Df['Signal'] = 0
        
    # Assign a value of 1 to 'Signal' column for the quantile with the highest returns
    Df.loc[Df['Ret'] > Df['Ret'][:split].quantile(q=0.66), 'Signal'] = 1

    # Assign a value of -1 to 'Signal' column for the quantile with the lowest returns
    Df.loc[Df['Ret'] < Df['Ret'][:split].quantile(q=0.34), 'Signal'] = -1
    
    X = Df.drop(['close', 'Signal', 'high', 'low', 'volume', 'Ret'], axis=1)

    # Create a variable which contains all the 'Signal' values
    y = Df['Signal']
    
    # Print dataframe with signals
    print(Df)

    if len(X) <= 0:
       print("Data missing so not processing further")
       return

    if context.rcv is None or context.cls is None or context.retrain_flag:

        context.retrain_flag = False
        
        # Call the RandomizedSearchCV function and pass the parameters
        context.rcv = RandomizedSearchCV(
                                        context.pipeline,
                                        context.parameters,
                                        cv=TimeSeriesSplit(n_splits=2)
                                        )

        # Call the 'fit' method of rcv and pass the train data to it
        context.rcv.fit(X.iloc[:split], y.iloc[:split])

        # Call the 'best_params_' method to obtain the best parameters of C
        best_C = context.rcv.best_params_['svc__C']

        # Call the 'best_params_' method to obtain the best parameters of kernel
        best_kernel = context.rcv.best_params_['svc__kernel']

        # Call the 'best_params_' method to obtain the best parameters of gamma
        best_gamma = context.rcv.best_params_['svc__gamma']

        # Create a new SVC classifier
        context.cls = SVC(C=best_C, kernel=best_kernel, gamma=best_gamma)

        # Pass the scaled train data to the SVC classifier
        context.cls.fit(context.ss1.fit_transform(X.iloc[:split]),
                        y.iloc[:split])
        warnings.filterwarnings("ignore")

    # Pass the test data to the predict function and store the values into 'y_predict'
    y_predict = context.cls.predict(context.ss1.transform(X.iloc[split:]))

    # Initiate a column by name, 'Pred_Signal' and assign 0 to it
    Df['Pred_Signal'] = 0
    
    # Save the predicted values for the train data
    Df.iloc[:split, Df.columns.get_loc('Pred_Signal')] = pd.Series(
        context.cls.predict(context.ss1.transform(X.iloc[:split])).tolist())

    # Save the predicted values for the test data
    Df.iloc[split:, Df.columns.get_loc('Pred_Signal')] = y_predict

    Df['True_Signal'] = 0
    Df.loc[Df['Ret'].shift(1) > 0, 'True_Signal'] = 1
    Df.loc[Df['Ret'].shift(1) < 0, 'True_Signal'] = -1

    # Calculate the accuracy
    context.accuracy = context.cls.score(
                                        X.iloc[split:],
                                        Df['True_Signal'][split:]
                                        )

    # Trading signal
    signal = list(Df['Pred_Signal'])[-1]
    
    # Print the entire dataframe and accuracy before placing order
    print(Df)
    print(f"Accuracy= {context.accuracy}")

    # Place the orders
    if context.accuracy > 0.5:

        if signal == 1:
            print("{} Going long in {}".format(get_datetime(), context.security.symbol))
            order_target_percent(context.security, 0.0003)

        elif signal == -1:
            print("{} Going short in {}".format(get_datetime(), context.security.symbol))
            order_target_percent(context.security, -0.0003)

        else:
            order_target_percent(context.security, 0)

    else:
        order_target_percent(context.security, 0)