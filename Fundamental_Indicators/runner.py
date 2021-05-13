import asyncio
import pandas
import yahoo_fin.stock_info as si
async def get_fundamental_indicators_for_company(config, company):
    company.fundmantal_indicators = {}

 # Statistics Valuation
 keys = {
    'Market Cap (intraday) 5': 'MarketCap',
    'Price/Sales (ttm)': 'PS',
    'Trailing P/E': 'PE',
    'PEG Ratio (5 yr expected) 1': 'PEG',
    'Price/Book (mrq)': 'PB'
 }
 data = si.get_stats_valuation(company.symbol)
 get_data_item(company.fundmantal_indicators, data, keys)

 # Income statement and Balance sheet
 data = get_statatistics(company.symbol)

 get_data_item(company.fundmantal_indicators, data,
              {
                  'Profit Margin': 'ProfitMargin',
                  'Operating Margin (ttm)': 'OperMargin',
                  'Current Ratio (mrq)': 'CurrentRatio',
                  'Payout Ratio 4': 'DivPayoutRatio'
              })

 get_last_data_item(company.fundmantal_indicators, data,
           {
               'Return on assets': 'ROA',
               'Return on equity': 'ROE',
               'Total cash per share': 'Cash/Share',
               'Book value per share': 'Book/Share',
               'Total debt/equity': 'Debt/Equity'
           })
		   
import fundamental_indicators_provider
config = {}
company = Company()
# Note: 
You might want to create an event loop and run within the loop:
loop = asyncio.get_event_loop()
loop.run_until_complete(fundamental_indicators_provider.get_fundamental_indicators_for_company(config, company))
print(company.fundmantal_indicators)