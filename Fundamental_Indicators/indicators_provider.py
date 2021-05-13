class Company:
   def __init__(self, symbol):
      self.symbol = symbol
      self.fundamental_indicators = {}
def to_float(val):
    if val == 0:
        return float(0)

    val = str(val).upper()
    
    if '%' in val:
        return round(float(val[:-1]), 4)

    m = {'K': 3, 'M': 6, 'B': 9, 'T': 12}

    for key in m.keys():
        if key in val:
            multiplier = m.get(val[-1])
            return round(float(val[:-1]) * (10 ** multiplier), 4)
    return round(float(val), 4)
def get_statatistics(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}/key-statistics?p={symbol}"
    dataframes = pandas.read_html(url)
    return pandas.concat(dataframes[1:])
def get_data_item(result, dataframe, columns):
    for column_to_find, column_to_name in columns.items():
        try:
            result[column_to_name] = list((dataframe.loc[dataframe[0] == column_to_find].to_dict()[1]).values())[0]
        except Exception as ex:
            result[column_to_name] = 'NA'

def get_last_data_item(result, dataframe, columns):
    data = dataframe.iloc[:, :2]
    data.columns = ["Column", "Last"]

    for column_to_find, column_to_name in columns.items():
        try:
            val = data[data.Column.str.contains(column_to_find, case=False, regex=True)].iloc[0, 1]
            float_val = to_float(val)
            result[column_to_name] = float_val
        except Exception as ex:
            result[column_to_name] = "NA"