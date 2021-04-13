# Stock Prediction AI
Analytics Club project 2020. 

News data collected for various companies is stored here: [link](https://drive.google.com/drive/folders/1xLPhpsjQh6rEAgi7kjowIzIK1xrM4ZeS?usp=sharing)


## Processed data
Processed data is available here: [link](https://drive.google.com/drive/folders/1x5rlR0Fy03Z634OhwoVNlFnQeBz8411e)

Price data is available as a CSV file with high, low and adjusted close price of each available stock combined into a single dataframe. `news_aligned.pkl` is a dictionary with keys as individual stocks and values as a lists, each containing a list of all news articles available for a particular stock on a given day. Each dictionary value has same number of lists in it as number of days in `combined_prices.csv`, with a one-to-one day to news mapping. 
