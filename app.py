import pandas as pd
from pandas_datareader import data as web
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import yfinance as fyn


from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

from flask import Flask, render_template



crypto = 'BTC'
currency = 'USD'

start = dt.datetime(2016,1,1)
end = dt.datetime.now()

fyn.pdr_override()
data = web.get_data_yahoo(f'{crypto}-{currency}', start, end).reset_index()

# print(data)
# print(data.columns)
# print((data['Date'][0].to_pydatetime().strftime('%Y-%m-%d')))

new_data = []

for i in range(0, len(data)):
    new_data.append((data['Date'][i].to_pydatetime().strftime('%Y-%m-%d'), data['Close'][i]))


# print(new_data)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

days = 60

x_train, y_train = [], []

for x in range (days, len(scaled_data)):
    x_train.append(scaled_data[x-days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=1, batch_size=32)



test_start = dt.datetime(2016,1,1)
test_end = dt.datetime.now()

test_data = web.get_data_yahoo(f'{crypto}-{currency}', test_start, test_end).reset_index()
actual_prices = test_data['Close'].values
test_date = test_data['Date'].values


total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)


x_test = []

for x in range(days, len(model_inputs)):
    x_test.append(model_inputs[x-days:x, 0])


x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

prediction_prices = [item for sublist in prediction_prices for item in sublist]

# print(prediction_prices)

test_predictions = []

for i in range(0, len(test_data)):
    test_predictions.append((test_data['Date'][i].to_pydatetime().strftime('%Y-%m-%d'), str(prediction_prices[i])))

# print(test_predictions)


plt.plot(actual_prices, color='black', label='Actual Prices')
plt.plot(prediction_prices, color='green', label='Predicted Prices')
plt.title(f'{crypto} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()




app = Flask(__name__)

@app.route("/")
def home():
    actual_data = new_data
    predicted_data = test_predictions

    actual_labels = [row[0] for row in actual_data]
    actual_values = [row[1] for row in actual_data]

    predicted_labels = [row[0] for row in predicted_data]
    predicted_values = [row[1] for row in predicted_data]


    return render_template("graph.html", actual_labels=actual_labels, actual_values=actual_values, predicted_labels=predicted_labels, predicted_values=predicted_values, )


