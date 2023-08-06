# TimeGPT {#timegpt}

> Unlock the power of accurate predictions and confidently navigate uncertainty. Reduce uncertainty and resource limitations. With TimeGPT, you can effortlessly access state-of-the-art models to make data-driven decisions. Whether you’re a bank forecasting market trends or a startup predicting product demand, TimeGPT democratizes access to cutting-edge predictive insights, eliminating the need for a dedicated team of machine learning engineers.

## Introduction {#introduction}

Nixtla’s TimeGPT is a generative pre-trained model trained to forecast time series data. The inputs to TimeGPT are time series data, and the model generates forecast outputs based on these. The input involves providing the historical data and potentially defining parameters such as the forecast horizon. TimeGPT can be used across a plethora of tasks including demand forecasting, anomaly detection, financial forecasting, and more.

The TimeGPT model “reads” time series data much like the way humans read a sentence – from left to right. It looks at a chunk of past data, which we can think of as “tokens”, and predicts what comes next. This prediction is based on patterns the model identifies in past data, much like how a human would predict the end of a sentence based on the beginning.

The TimeGPT API provides an interface to this powerful model, allowing users to leverage its forecasting capabilities to predict future events based on past data. With this API, users can not only forecast future events but also delve into various time series-related tasks, such as what-if scenarios, anomaly detection, and more.

![figure](./img/timegpt-arch.png)

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
%load_ext autoreload
%autoreload 2
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

<details>
<summary>Code</summary>

``` python
import inspect
import json
import requests
from typing import Dict, List, Optional

import pandas as pd
```

</details>

:::

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
import logging
import os

from dotenv import load_dotenv
from fastcore.test import test_eq, test_fail
from nbdev.showdoc import show_doc


load_dotenv()
logging.getLogger('statsforecast').setLevel(logging.ERROR)
```

</details>

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

<details>
<summary>Code</summary>

``` python
class TimeGPT:
    """
    A class used to interact with the TimeGPT API.
    """

    def __init__(self, token: str):
        """
        Constructs all the necessary attributes for the TimeGPT object.

        Parameters
        ----------
        token : str
            The authorization token to interact with the TimeGPT API.
        """
        self.token = token
        self.api_url = 'https://dashboard.nixtla.io/api'
        self.weights_x: pd.DataFrame = None

    @property
    def request_headers(self):
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.token}"
        }
        return headers
        
    def _parse_response(self, response) -> Dict:
        """Parses responde."""
        response.raise_for_status()
        try:
            resp = response.json()
        except Exception as e:
            raise Exception(response)
        return resp

    def _input_size(self, freq: str):
        response_input_size = requests.post(
            f'{self.api_url}/timegpt_input_size',
            json={'freq': freq}, 
            headers=self.request_headers,
        )
        response_input_size = self._parse_response(response_input_size)
        return response_input_size['data']

    def _validate_inputs(
            self,
            df: pd.DataFrame,
            X_df: pd.DataFrame,
            id_col: str,
            time_col: str,
            target_col: str,
        ):
        renamer = {
            id_col: 'unique_id',
            time_col: 'ds',
            target_col: 'y'
        }
        df = df.rename(columns=renamer)
        drop_uid = False
        if 'unique_id' not in df.columns:
            # Insert unique_id column
            df = df.assign(unique_id='ts_0')
            drop_uid = True
        if X_df is not None:
            X_df = X_df.rename(columns=renamer)
            if 'unique_id' not in df.columns:
                X_df = X_df.assign(unique_id='ts_0')
        return df, X_df, drop_uid

    def _validate_outputs(
            self,
            fcst_df: pd.DataFrame,
            id_col: str,
            time_col: str,
            target_col: str,
            drop_uid: bool,
        ):
        renamer = {
            'unique_id': id_col,
            'ds': time_col,
            'target_col': target_col,
        }
        if drop_uid:
            fcst_df = fcst_df.drop(columns='unique_id')
        fcst_df = fcst_df.rename(columns=renamer)
        return fcst_df

    def _preprocess_inputs(
            self, 
            df: pd.DataFrame, 
            h: int,
            freq: str,
            X_df: Optional[pd.DataFrame] = None,
        ):
        input_size = self._input_size(freq)
        y_cols = ['unique_id', 'ds', 'y']
        y = df[y_cols].groupby('unique_id').tail(input_size + h)
        to_dict_args = {'orient': 'split'}
        if 'index' in inspect.signature(pd.DataFrame.to_dict).parameters:
            to_dict_args['index'] = False
        y = y.to_dict(**to_dict_args)
        x_cols = df.drop(columns=y_cols).columns.to_list()
        if len(x_cols) == 0:
            x = None
        else:
            x = pd.concat([df[['unique_id', 'ds'] + x_cols].groupby('unique_id').tail(input_size + h), X_df])
            x = x.sort_values(['unique_id', 'ds'])
            x = x.to_dict(**to_dict_args)
        return y, x, x_cols

    def _multi_series(
            self,
            df: pd.DataFrame,
            h: int,
            freq: str,
            X_df: Optional[pd.DataFrame] = None,
            level: Optional[List[int]] = None,
            finetune_steps: int = 0,
            clean_ex_first: bool = True,
        ):
        y, x, x_cols = self._preprocess_inputs(df=df, h=h, freq=freq, X_df=X_df)
        payload = dict(
            y=y,
            x=x,
            fh=h,
            freq=freq,
            level=level,
            finetune_steps=finetune_steps,
            clean_ex_first=clean_ex_first,
        )
        response_timegpt = requests.post(
            f'{self.api_url}/timegpt_multi_series',
            json=payload, 
            headers=self.request_headers,
        )
        response_timegpt = self._parse_response(response_timegpt)
        if 'weights_x' in response_timegpt['data']:
            self.weights_x = pd.DataFrame({
                'features': x_cols,
                'weights': response_timegpt['data']['weights_x'],
            })
        return pd.DataFrame(**response_timegpt['data']['forecast'])

    def forecast(
            self,
            df: pd.DataFrame,
            h: int,
            freq: str,    
            id_col: str = 'unique_id',
            time_col: str = 'ds',
            target_col: str = 'y',
            X_df: Optional[pd.DataFrame] = None,
            level: Optional[List[int]] = None,
            finetune_steps: int = 0,
            clean_ex_first: bool = True,
        ):
        """Forecast your time series using TimeGPT.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame on which the function will operate. Expected to contain at least the following columns:
            - time_col:
                Column name in `df` that contains the time indices of the time series. This is typically a datetime
                column with regular intervals, e.g., hourly, daily, monthly data points.
            - target_col:
                Column name in `df` that contains the target variable of the time series, i.e., the variable we 
                wish to predict or analyze.
            Additionally, you can pass multiple time series (stacked in the dataframe) considering an additional column:
            - id_col:
                Column name in `df` that identifies unique time series. Each unique value in this column
                corresponds to a unique time series.
        h : int
            Forecast horizon.
        freq : str
            Frequency of the data.
            See [pandas' available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
        id_col : str (default='unique_id')
            Column that identifies each serie.
        time_col : str (default='ds')
            Column that identifies each timestep, its values can be timestamps or integers.
        target_col : str (default='y')
            Column that contains the target.
        X_df : pandas.DataFrame, optional (default=None)
            DataFrame with [`unique_id`, `ds`] columns and `df`'s future exogenous.
        level : List[float], optional (default=None)
            Confidence levels between 0 and 100 for prediction intervals.
        finetune_steps : int (default=0)
            Number of steps used to finetune TimeGPT in the
            new data.
        clean_ex_first : bool (default=True)
            Clean exogenous signal before making forecasts
            using TimeGPT.
        
        Returns
        -------
        fcsts_df : pandas.DataFrame
            DataFrame with TimeGPT forecasts for point predictions and probabilistic
            predictions (if level is not None).
        """
        df, X_df, drop_uid = self._validate_inputs(
            df=df,
            X_df=X_df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        fcst_df = self._multi_series(
            df=df, 
            h=h,
            freq=freq,
            X_df=X_df,
            level=level, 
            finetune_steps=finetune_steps,
            clean_ex_first=clean_ex_first,
        )
        fcst_df = self._validate_outputs(
            fcst_df=fcst_df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            drop_uid=drop_uid,
        )
        return fcst_df
```

</details>

:::

## Usage {#usage}

<details>
<summary>Code</summary>

``` python
show_doc(TimeGPT.__init__, title_level=3, name='TimeGPT')
```

</details>

You can instantiate the `TimeGPT` class providing your credentials.

<details>
<summary>Code</summary>

``` python
timegpt = TimeGPT(token=os.environ['TIMEGPT_TOKEN'])
```

</details>

::: {.cell 0=‘h’ 1=‘i’ 2=‘d’ 3=‘e’}

<details>
<summary>Code</summary>

``` python
# test input_size
test_eq(
    timegpt._input_size('D'),
    28,
)
```

</details>

:::

<details>
<summary>Code</summary>

``` python
show_doc(TimeGPT.forecast, title_level=4)
```

</details>

Now you can start to make forecasts! Let’s import an example:

<details>
<summary>Code</summary>

``` python
df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv')
df.head()
```

</details>

Let’s plot this series

<details>
<summary>Code</summary>

``` python
df.set_index('timestamp').plot(figsize=(20, 10))
```

</details>

Now we can forecast this dataset. We observe that this dataset has monthly frequency. We have to pass the right pandas frequency to `TimeGPT` to have the right forecasts. In this case ‘MS’. Let’s forecast the next 12 observations. In this case we also have to define:

-   `time_col`: Column that identifies the datestamp column.
-   `target_col`: The variable that we want to forecast.

<details>
<summary>Code</summary>

``` python
timegpt_fcst_df = timegpt.forecast(df=df, h=12, freq='MS', time_col='timestamp', target_col='value')
timegpt_fcst_df.head()
```

</details>
<details>
<summary>Code</summary>

``` python
pd.concat([df, timegpt_fcst_df]).set_index('timestamp').plot(figsize=(20, 10))
```

</details>

You can also produce a larger forecast horizon:

<details>
<summary>Code</summary>

``` python
timegpt_fcst_df = timegpt.forecast(df=df, h=36, freq='MS', time_col='timestamp', target_col='value')
timegpt_fcst_df.head()
```

</details>
<details>
<summary>Code</summary>

``` python
pd.concat([df, timegpt_fcst_df]).set_index('timestamp').plot(figsize=(20, 10))
```

</details>

### Prediction Intervals {#prediction-intervals}

Prediction intervals provide a measure of the uncertainty in the forecasted values. In time series forecasting, a prediction interval gives an estimated range within which a future observation will fall, based on the level of confidence or uncertainty you set. This level of uncertainty is crucial for making informed decisions, risk assessments, and planning.

For instance, a 95% prediction interval means that 95 out of 100 times, the actual future value will fall within the estimated range. Therefore, a wider interval indicates greater uncertainty about the forecast, while a narrower interval suggests higher confidence.

When using TimeGPT for time series forecasting, you have the option to set the level of prediction intervals according to your requirements. TimeGPT uses conformal prediction to calibrate the intervals.

Here’s how you could do it:

<details>
<summary>Code</summary>

``` python
timegpt_fcst_pred_int_df = timegpt.forecast(
    df=df, h=12, freq='MS', level=[80, 90], 
    time_col='timestamp', target_col='value',
)
timegpt_fcst_pred_int_df.head()
```

</details>
<details>
<summary>Code</summary>

``` python
import matplotlib.pyplot as plt
```

</details>
<details>
<summary>Code</summary>

``` python
history_with_fcst_df = pd.concat([df, timegpt_fcst_pred_int_df])
ax = history_with_fcst_df[['timestamp', 'value', 'TimeGPT']].set_index('timestamp').plot(figsize=(20, 10))
for level, alpha in zip([80, 90], [0.4, 0.2]):
    plt.fill_between(
        history_with_fcst_df['timestamp'], 
        history_with_fcst_df[f'TimeGPT-lo-{level}'], 
        history_with_fcst_df[f'TimeGPT-hi-{level}'], 
        color='orange', 
        alpha=alpha,
        label=f'TimeGPT-level-{level}]'
    )
plt.legend()
plt.show()
```

</details>

It’s essential to note that the choice of prediction interval level depends on your specific use case. For high-stakes predictions, you might want a wider interval to account for more uncertainty. For less critical forecasts, a narrower interval might be acceptable.

### Finetuning {#finetuning}

Fine-tuning is a process of further training a pre-existing model (like TimeGPT) on a specific task or dataset. This allows you to leverage the general language understanding capabilities of the pre-trained model and adapt it to your specific use case.

In TimeGPT, you can use the `finetune_steps` argument to specify the number of additional training steps the model should undergo on your time series data. This helps in refining the model’s understanding and prediction of your data patterns.

Here’s an example of how to fine-tune TimeGPT:

<details>
<summary>Code</summary>

``` python
timegpt_fcst_finetune_df = timegpt.forecast(
    df=df, h=12, freq='MS', finetune_steps=10,
    time_col='timestamp', target_col='value',
)
```

</details>
<details>
<summary>Code</summary>

``` python
pd.concat([df, timegpt_fcst_finetune_df]).set_index('timestamp').plot(figsize=(20, 10))
```

</details>

In this code, `finetune_steps: 10` means the model will go through 10 iterations of training on your time series data.

Keep in mind that fine-tuning can be a bit of trial and error. You might need to adjust the number of `finetune_steps` based on your specific needs and the complexity of your data. It’s recommended to monitor the model’s performance during fine-tuning and adjust as needed. Be aware that more `finetune_steps` may lead to longer training times and could potentially lead to overfitting if not managed properly.

Remember, fine-tuning is a powerful feature, but it should be used thoughtfully and carefully.

### Multiple Series {#multiple-series}

TimeGPT provides a robust solution for multi-series forecasting, which involves analyzing multiple data series concurrently, rather than a single one. The tool can be fine-tuned using a broad collection of series, enabling you to tailor the model to suit your specific needs or tasks.

The following dataset contains prices of different electricity markets. Let see how can we forecast them.

<details>
<summary>Code</summary>

``` python
df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short.csv')
df.head()
```

</details>

Let’s plot this series using [`StatsForecast`](https://github.com/Nixtla/statsforecast):

<details>
<summary>Code</summary>

``` python
from statsforecast import StatsForecast as sf
```

</details>
<details>
<summary>Code</summary>

``` python
sf.plot(df, engine='matplotlib')
```

</details>

We just have to pass the dataframe to create forecasts for all the time series at once.

<details>
<summary>Code</summary>

``` python
timegpt_fcst_multiseries_df = timegpt.forecast(df=df, h=24, freq='H', level=[80, 90])
timegpt_fcst_multiseries_df.head()
```

</details>
<details>
<summary>Code</summary>

``` python
sf.plot(df, timegpt_fcst_multiseries_df, max_insample_length=365, level=[80, 90], engine='matplotlib')
```

</details>

### Exogenous variables {#exogenous-variables}

Exogenous variables or external factors are crucial in time series forecasting as they provide additional information that might influence the prediction. These variables could include holiday markers, marketing spending, weather data, or any other external data that correlate with the time series data you are forecasting.

For example, if you’re forecasting ice cream sales, temperature data could serve as a useful exogenous variable. On hotter days, ice cream sales may increase.

To incorporate exogenous variables in TimeGPT, you’ll need to pair each point in your time series data with the corresponding external data.

Let’s see an example.

<details>
<summary>Code</summary>

``` python
df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short-with-ex-vars.csv')
df.head()
```

</details>

To produce forecasts we have to add the future values of the exogenous variables. Let’s read this dataset. In this case we want to predict 24 steps ahead, therefore each unique id will have 24 observations.

<details>
<summary>Code</summary>

``` python
future_ex_vars_df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short-future-ex-vars.csv')
future_ex_vars_df.head()
```

</details>

Let’s call the `forecast` method, adding this information:

<details>
<summary>Code</summary>

``` python
timegpt_fcst_ex_vars_df = timegpt.forecast(df=df, X_df=future_ex_vars_df, h=24, freq='H', level=[80, 90])
timegpt_fcst_ex_vars_df.head()
```

</details>
<details>
<summary>Code</summary>

``` python
sf.plot(df[['unique_id', 'ds', 'y']], timegpt_fcst_ex_vars_df, max_insample_length=365, level=[80, 90], engine='matplotlib')
```

</details>

We also can get the importance of the features.

<details>
<summary>Code</summary>

``` python
timegpt.weights_x.plot.barh(x='features', y='weights')
```

</details>
