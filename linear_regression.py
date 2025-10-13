import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import date, timedelta
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from sampler import Sampler

np.set_printoptions(legacy='1.25')

class Linear_Regression():
    def __init__(self,
                 symbol_list,
                 start = str(date.today() - timedelta(59)),
                 end = str(date.today() - timedelta(1)),
                 interval = '1d',
                 optional_df=None):
        if optional_df is not None:
            self.df = optional_df
        else:
            df = yf.download(symbol_list, start, end, interval = interval, multi_level_index=False, ignore_tz=True)
            self.df = df
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = [' '.join(col).strip() for col in self.df.columns.values]
        self.symbol_list = symbol_list
        self.df['Day Count'] = np.arange(1, len(self.df) + 1)
        self.ticker_x = self.symbol_list[0]
        self.ticker_y = self.symbol_list[1]
        # Always use returns for models due to stationarity
        self.df[f'{self.symbol_list[0]} Return'] = np.log(self.df[f'Close {self.symbol_list[0]}']) # returns for x
        self.df[f'{self.symbol_list[1]} Return'] = np.log(self.df[f'Close {self.symbol_list[1]}']) # returns for y
        self.df.dropna(inplace = True)
        #sampler goes here
        sampler = Sampler(self.df)
        self.df = sampler.sampled_df
        self.x_prices = self.df[f'Close {self.symbol_list[0]}'] # price series of first symbol
        self.y_prices = self.df[f'Close {self.symbol_list[1]}'] # price series of second symbol
        self.x_returns = self.df[f'{self.symbol_list[0]} Return'] # return series of first symbol
        self.y_returns = self.df[f'{self.symbol_list[1]} Return'] # return
        self.interval = interval

        #Preliminary Calculations
        self.mean_x = self.df[f'{self.symbol_list[0]} Return'].mean() # Xbar
        self.mean_y = self.df[f'{self.symbol_list[1]} Return'].mean() # Ybar
        self.idx = self.df.index.get_loc(self.df.index[0])
        self.last_idx = self.df.index.get_loc(self.df.index[-1])
        def SSxy_calculation(return_array_1, return_array_2, last_idx):
            SSxy = 0
            for return_idx in range(0, last_idx):
                SSxy += (return_array_1.iloc[return_idx] - self.mean_x) * (return_array_2.iloc[return_idx] - self.mean_y)
            return SSxy
        self.ssxy = SSxy_calculation(self.df[f'{self.symbol_list[0]} Return'], self.df[f'{self.symbol_list[1]} Return'], self.last_idx) # SSxy
        def SSxx_calculation(return_array, mean_value, last_idx):
            #Works as SSyy also
            SSxx = 0
            for return_idx in range(0, last_idx):
                SSxx += ((return_array.iloc[return_idx] - mean_value)**2)
            return SSxx
        self.ssxx = SSxx_calculation(self.df[f'{self.symbol_list[0]} Return'], self.mean_x, self.last_idx) # SSxx
        self.ssyy = SSxx_calculation(self.df[f'{self.symbol_list[1]} Return'], self.mean_y, self.last_idx) # SSyy
        self.slope = self.ssxy / self.ssxx
        self.sse = self.ssyy - (self.slope * self.ssxy)
        self.y_int = self.mean_y - (self.slope * self.mean_x)
        self.degrees_of_freedom = 2
        self.n = self.df.index.size
        self.var = self.sse / (self.n - self.degrees_of_freedom)
        self.standard_deviation = math.sqrt(self.var)
        self.standard_error_of_slope = self.standard_deviation / (math.sqrt(self.ssxx))
        #r is the measure of strength in a linear relationship
        self.r = self.ssxy / math.sqrt(self.ssxx * self.ssyy)
        #r2 is the amount that x contributes to predicting y
        self.r2 = (self.ssyy - self.sse) / self.ssyy
        print(f"SSxy = {round(self.ssxy,4)} | SSxx = {round(self.ssxx,4)} | SSyy = {round(self.ssyy,4)} | Slope = {round(self.slope,4)} | Y-int = {round(self.y_int,4)}")
        print('-' * 50)
        print(f"Variance: {round(self.var,4)} | Standard Deviation: {round(self.standard_deviation,4)} | N: {self.n}")
        print('-' * 50)
        print(f"Coefficient of Correlation(r): {round(self.r,4)} | Coefficient of Determination(R^2): {round(self.r2,4)}")
        print('-' * 50)
        print(f"Model: Y = {round(self.y_int,4)} + {round(self.slope,4)} * X")

    def hypothesis_test(self, significance_level=.05):
        # Null Hypothesis is Slope = 0
        # Alternative Hypothesis is Slope != 0
        t = self.slope / self.standard_error_of_slope
        critical_value = stats.t.ppf(1-(significance_level / 2), self.n - self.degrees_of_freedom)
        if abs(t) > critical_value:
            print(f'We REJECT the Null Hypothesis')
            print('-' * 50)
            print(f'The Test Statistic {round(t,4)} falls in the rejection region of {round(critical_value,4)}')
            print(f"We can conclude that the slope of the linear regression between {self.symbol_list[0]} and {self.symbol_list[1]} is not 0")
            print("Thus, we can predict the value of Y based on a given X, and conclude that X contributes to Y")
        else:
            print(f"We CANNOT reject the Null Hypothesis at this signficance level")
            print(f'The Test Statistic {round(t,4)} does not fall into the rejection region of {round(critical_value,4)}')
            print(f"More than likely no linear relationship exists between {self.symbol_list[0]} and {self.symbol_list[1]}")

    def slope_confidence_interval(self, confidence_level=.95):
        critical_value = stats.t.ppf(1-((1-confidence_level) / 2), self.n - self.degrees_of_freedom)
        upper_level = self.slope + (critical_value * self.standard_error_of_slope)
        lower_level = self.slope - (critical_value * self.standard_error_of_slope)
        interval = [round(lower_level,4), round(upper_level,4)]
        print(f'{confidence_level * 100}% Confidence Level: Slope lies between {interval}')
        return interval
    
    def find_y(self, x, confidence_level=.95, find_mean=False):
        #y is also called Y hat here
        y = self.y_int + self.slope * x
        critical_value = stats.t.ppf(1-((1-confidence_level) / 2), self.n - self.degrees_of_freedom)
        print(f'Given X: {x} | Expect Y: {round(y,4)}')
        if find_mean is True:
            #Whenever we run the model on X, we'd expect Y to lie within this interval
            upper_level = y + critical_value * self.standard_deviation * math.sqrt((1/self.n) + ((x - self.mean_x)**2 / self.ssxx))
            lower_level = y - critical_value * self.standard_deviation * math.sqrt((1/self.n) + ((x - self.mean_x)**2 / self.ssxx))
            interval = [round(lower_level * 100,4), round(upper_level * 100,4)]
            print (f'{confidence_level * 100}% Confidence Interval of Y Mean: If {self.ticker_x} moves {x * 100}%, we expect the mean of {self.ticker_y} to move {round(y,4) * 100}% between the interval {interval}')
        else:
            # Prediction interval for a single test of Y
            upper_level = y + critical_value * self.standard_deviation * math.sqrt(1 + (1/self.n) + ((x - self.mean_x)**2 / self.ssxx))
            lower_level = y - critical_value * self.standard_deviation * math.sqrt(1 + (1/self.n) + ((x - self.mean_x)**2 / self.ssxx))
            interval = [round(lower_level * 100,4), round(upper_level * 100,4)]
            print (f'{confidence_level * 100}% Prediction Interval: If {self.ticker_x} moves {x * 100}%, we expect {self.ticker_y} to move {round(y,4) * 100}% between the interval {interval}')

    def scatterplot(self):
        # Calculate regression line
        x = self.x_returns
        y = self.y_returns
        y_hat = self.y_int + self.slope * x

        # Calculate residuals and standard deviation
        std = self.standard_deviation

        # Calculate upper and lower bands
        upper_band = y_hat + 1 * std
        lower_band = y_hat - 1 * std
        upper_band2 = y_hat + 2 * std
        lower_band2 = y_hat - 2 * std

        # Create scatter plot
        fig = px.scatter(x=x, y=y, trendline='ols', opacity=0.5, trendline_color_override='white', labels={'x': self.ticker_x, 'y': self.ticker_y})
        fig.add_traces([
            go.Scatter(x=x, y=upper_band, mode='lines', name=f'+1σ', line=dict(color='skyblue', dash='dash')),
            go.Scatter(x=x, y=lower_band, mode='lines', name=f'-1σ', line=dict(color='skyblue', dash='dash')),
            go.Scatter(x=x, y=upper_band2, mode='lines', name=f'+2σ', line=dict(color='purple', dash='dash')),
            go.Scatter(x=x, y=lower_band2, mode='lines', name=f'-2σ', line=dict(color='purple', dash='dash'))
        ])
        fig.update_layout(title=f'Scatterplot of {self.ticker_x} vs {self.ticker_y}', template='plotly_dark')
        fig.show()
