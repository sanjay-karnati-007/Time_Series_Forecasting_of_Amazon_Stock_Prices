# Time_Series_Forecasting_of_Amazon_Stock_Prices

Project analyzes Amazon Stock data using Python. Feature Extraction is performed and ARIMA and Fourier series models are made. LSTM is used with multiple features to predict stock prices.

I predicted different models of the time series OHLCV data, performed feature extraction, hyper parameter tuning and trained on ARIMA, Fourier, LSTM. Then I focused on stock price movement instead of stock prices because I found that it's more accurate to predict them.
Below i give an overview of what I have learnt.
I took historical Amazon data from Yahoo.Api and then performed feature generation on it, ARIMA model, Fourier model. Then performed LSTM on said model.
Put up the data in time series form and split between train and test seen below.

![image](https://github.com/user-attachments/assets/89d65cdf-9d6a-43fa-a64d-9b283cfc1234)


Amazon data peaks around 2015 and after. Most of this data is in the testing set and training set does not have peak value data.
I have predicted this could be a problem while dealing with conventional LSTM.
### FEATURE GENERATION:
Following technical indicators were generated other than OHLCV data:

1. Bollinger bands: Bollinger Bands is used to define the prevailing high and low prices in a market to characterize the trading band of a financial instrument or commodity. Bollinger Bands are a volatility indicator. Bands are consists of Moving Average (MA) line, a upper band and lower band. The upper and lower bands are simply MA adding and subtracting standard deviation.
 
2.	EMA: Exponential moving average is a better version of a simple moving average that doesnt have SMAs lag. Moving averages just average out the data for a given time so we know how the company's closing price are trending for a given amount of days. example for 4 days is price was 22,23 ,45,1
(the company crashed on 4th day) the average would be 23. Now 23 is a below average value so it gives us an idea that 45 was indeed just a fluke and that infact the company was always making losses
EMA is calculated as:
EMA(t)EMA(t0)=(1−α)EMA(t−1)+α p(t)=p(t0)
where	α=1L+1 and length of window is α=2M
I used the ewm(exponential weighted mean ) function to calculate ema.
3.	Momentum: Momentum is perhaps the simplest and easiest oscillator (financial analysis tool) to understand and use. It is the measurement of the speed or velocity of price changes, or the rate of change in price movement for a particular asset.
The formula for momentum is: Momentum=V−Vx	where:
V=Latest price Vx=Closing price x=Number of days ago
while generating them other features that got generated were: 20SD(Standard Deviation of 20days) , UpperBand, Lower band, moving average of 7 days and 21 days and exponential moving average of 26 and 12 days.

![image](https://github.com/user-attachments/assets/026a2b9d-7b1d-4adc-9b62-c891b441e3fd)


Plot of technical indictaors over days where they started peaking. They also peak around the days where Amazon saw a huge growth.

Then I genrated ARIMA and Fourier models and decided to see if they can be used as features.
## ARIMA:

![image](https://github.com/user-attachments/assets/ee283284-af5d-4c53-a90f-208dd83689db)

- SUMMARY OF THE ARIMA MODEL
1.	A good starting point for the AR parameter of the model may be 5 which we did.
2.	From the summary of the ARIMA we can see that most P-values are greater than 0.05 other than the last two.The model should be great!
3.	The difference between AIC and BIC is low so this indicates this is a good model
4.	Running the example, we can see that there is a positive correlation with the first 0-to-500 lags that is perhaps significant for the first 250 lags in the autocorrelation below

![image](https://github.com/user-attachments/assets/8f5bbaef-d142-41fe-8ea7-ff217f25cc4e)


ARIMA prediction plot is pretty good:

![image](https://github.com/user-attachments/assets/71ed73e7-05cc-4148-8a9c-85b24b996461)

FOURIER MODEL:
Use Fourier Tranform in the spectral domain and reconvert it into time domain and plot with multiple components. The component which is closest to real values can be plotted.

![image](https://github.com/user-attachments/assets/6d92ada0-5b09-4302-ba28-58a8c327dfc3)

In our case it is 100 components.
Normalise the values ie do not keep spectral component values and generate the prediction data Fourier gives results very very close to Closing price data as seen in plot:

![image](https://github.com/user-attachments/assets/2cf629fc-1f34-4985-8123-eb36d61bc6e0)

SIMPLE MOVING AVERAGE:

![image](https://github.com/user-attachments/assets/859bdba8-7889-42be-8be3-a2a949c3ea71)


I used the simple moving average by creating a lookback window and then ran it on the data. I was able to get a good model just as expected from SMA.
 
EXPONENTIAL MOVING AVERAGE:

![image](https://github.com/user-attachments/assets/98fc2459-51bf-401a-b58e-392deba4434b)

EMA is a great model for this dataset. Ideally the pattern of the True data should have been followed in the prediction model. I coded	from range to 1 to N-1 and put all the averaged values in the running mean. I used dense as 0.5 and then multiply it with the running average.
It predicts the predicted values after performing EMA on the dataset formula of which has been given above.

FEATURE IMPORTANCE USING XGBOOST:
Using XGBoost I found which features would make be the best for prediction. These are plotted below.

![image](https://github.com/user-attachments/assets/ccbc1852-f518-40cd-a4bc-5a2c94ab3111)

 
 
As seen above, Open Adj CLose, EMA and high and Low are great indicators. Others are ma7, ma21 and 12ema.
LSTM model to predict stock prices using 1 feature.
I ran Open training data for 100 epochs and tried to predict Open with it. This is more like a regression problem so the metrics I used were mse and mean absolute error, not accuracy. The results weren't all that great since there was some overfitting and I normalized the data and did not perform hyperparamater tuning.
MAE was: 0.167
This means the average difference between input and ouput for all 2300 datapoints is 0.167.
However the value is for the days here so the MAE here is pretty bad.(2350 length of dataset will be denominator. Difference between actual and prediced values should be so small that such a large denominator dividing the difference should put MAE in rage of 10^-3 ie 0.00then digits. Since MAE is 167.something*10^-3(0.167) difference is high.
The prediction plot looked like this:

![image](https://github.com/user-attachments/assets/0662cd74-946d-4b9d-8460-b2beb181d13f)

 
Not great.

LSTM MODEL USING MULTIPLE FEATURES:
So I tried more features(5 features) and I tried to predict closing prices with them. I encoded and normalized the 5 features and the prediction plot looked something like this: Again this was a regression problem so the metrics I used where mse and mae and not accuracy.

![image](https://github.com/user-attachments/assets/537e7e55-116f-4387-9d98-d701e4f1b9e0)


MAE here was 0.016 during training and the plot continues the trend of the training data but I am not satisfied.
MAE calculation is sum(Y-X)/total data points = (2000-1980)/2350
here 2000 is the day with highest closing price. 1980 is the day with highest closing price in training set and 2350 will be the length of the data,
My mistakes with the 1st LSTM models were:
 
•	Encoding and normalization lost a lot of precious data. We have only 2000 days of data and 1500 days of training data where every data point matters. So no extensive encoding and normalization.

•	I tried to predict stock prices as a regression problem but di not do any hyperparameter tuning. Adding linearity in a neural network will produce flat results.

•	Instead of feeding direct stock prices I should feed in stock price movements by creating a window that keeps all similar stock price values in one window and feeds in each window as a datapoint.

LSTM PREDICTING STOCK PRICE MOVEMENT
I created a next 'Vanila' LSTM model to predict only Closing stock prices and got a good predction model:

-Ran a window over close data to convert closing prices into closing price movements.

-Normalized the data and split into train test and validation.

-Ran the model and saw to that it does not overfit.

-Got a mae of 0.0076 lowest yet in any model

-Plotted a good prediction model:


![image](https://github.com/user-attachments/assets/7f38f2f8-d3ce-4502-9a30-522b6fd044b0)

## Connect with Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/sanjay-karnati/)

---




