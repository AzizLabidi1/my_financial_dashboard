# -*- coding: utf-8 -*-
###############################################################################
# MY FINANCIAL DASHBOARD - v3
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st

#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret
    
#==============================================================================
# Header
#==============================================================================

def render_header():
    """
    This function render the header of the dashboard with the following items:
        - Title
        - Dashboard description
        - 3 selection boxes to select: Ticker, Start Date, End Date
        - Update button
    """
    
    # Add dashboard title and description
    st.title("⭐ MY FINANCIAL DASHBOARD ⭐")
    col1, col2 = st.columns([1,5])
    col1.write("Data source:")
    col2.image('./img/yahoo_finance.png', width=200)
    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    
    # Add the selection boxes
    col1, col2, col3, col4 = st.columns(4)  # Create 4 columns
    # Ticker name
    global ticker  # Set this variable as global, so the functions in all of the tabs can read it
    ticker = col1.selectbox("Ticker", ticker_list)
    # Begin and end dates
    global start_date, end_date  # Set this variable as global, so all functions can read it
    start_date = col2.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date = col3.date_input("End date", datetime.today().date())
    # Add an "Update" button
    if col4.button("Update"):
       # Trigger an update 
       yf.download(ticker, start_date, end_date)
    
#==============================================================================
# Tab 1
#==============================================================================

def render_tab1():
    """
    This function render the Tab 1 - Summary of the dashboard.
    """
       
    
    # Get the company information
    @st.cache_data
    def GetCompanyInfo(ticker):
        """
        This function get the company information from Yahoo Finance.
        """
        return  YFinance(ticker).info
    
    # If the ticker is already selected
    if ticker != '':
        # Get the company information in list format
        info = GetCompanyInfo(ticker)
        
        
        # Display the company profile information
        st.write('**1. Company Profile:**')
        
        # Fetch the company profile information
        company_profile_info = {
            "Address": info.get("address1", 'Not provided'),
            "city": info.get("city", 'Not provided') + ", " + info.get("state", 'Not provided') + ", " + info.get("zip", 'Not provided'),
            "country": info.get("country", 'Not provided'),
            "Phone": info.get("phone", 'Not provided'),
            "Website": info.get("website", 'Not provided'),
            "Industry": info.get("industry", 'Not provided'),
            "Sector(s)": info.get("sector", 'Not provided'),
            "Full Time Employees":info.get("fullTimeEmployees", 'Not provided')
        }
        
        # format the company profile information
        left_keys = ["Address", "city", "country", "Phone", "Website"]
        right_keys = ["Sector(s)", "Industry", "Full Time Employees"]
        
        # Display the formatted company profile information
        col1, col2 = st.columns(2)
        
        # Display left keys and values on the left side
        for key in left_keys:
            value = company_profile_info[key]
            col1.write(f"{value}")
        
        # Display right keys and values on the right side
        for key in right_keys:
            value = company_profile_info[key]
            col2.write(f"**{key}:** {value}")

        
        # Show the company description using markdown + HTML
        st.write('**2. Company Description:**')
        st.markdown('<div style="text-align: justify;">' + \
                    info['longBusinessSummary'] + \
                    '</div><br>',
                    unsafe_allow_html=True)
        
        # Display the stakeholders information
        st.write('**3. Stakeholders:**')
    
        if 'companyOfficers' in info:
            officers_data = []
            for officer in info['companyOfficers']:
                name = officer.get('name', 'N/A')
                title = officer.get('title', 'N/A')
                pay = officer.get('totalPay', {}).get('fmt', 'N/A')
                exercised = officer.get('exercisedValue').get('fmt') if officer.get('exercisedValue').get('fmt') is not None else 'N/A'
                year_born = officer.get('yearBorn', 'N/A')
                officers_data.append([name, title, pay, exercised, year_born])
        
            columns = ['Name', 'Title', 'Pay', 'Exercised', 'Year Born']
            officers_df = pd.DataFrame(officers_data, columns=columns)
            
            st.write(officers_df)
        else:
            st.write("No company officers' information available.")
        
        # Show some statistics as a DataFrame
        st.write('**3. Key Statistics:**')
        info_keys = {
                        'previousClose': 'Previous Close',
                        'open': 'Open',
                        'bid': 'Bid',
                        'ask': 'Ask',
                        'dayLow': "Low",
                        'dayHigh': "High",
                        'fiftyTwoWeekLow': '52 Week Low',
                        'fiftyTwoWeekHigh': '52 Week High',
                        'volume': 'Volume',
                        'averageVolume': 'Avg. Volume',
                        'marketCap': 'Market Cap',
                        'beta': 'Beta (5Y Monthly)',
                        'peRatio': 'PE Ratio (TTM)',
                        'trailingEps': 'EPS (TTM)',
                        'dividendYield': 'Forward Dividend & Yield',
                        'targetMeanPrice':"1y Target Est"
                    }
        # Create two columns layout
        col1, col2 = st.columns(2)
        
        # Display the table in the first column
        company_stats_left = {}
        for key in list(info_keys.keys())[:len(info_keys) // 2]:
            company_stats_left.update({info_keys[key]: info.get(key, 'Not provided')})
        company_stats_left = pd.DataFrame({'Value': pd.Series(company_stats_left)})
        col1.table(company_stats_left)
        
        # Display the table in the second column
        company_stats_right = {}
        for key in list(info_keys.keys())[len(info_keys) // 2:]:
            company_stats_right.update({info_keys[key]: info.get(key, 'Not provided')})
        company_stats_right = pd.DataFrame({'Value': pd.Series(company_stats_right)})
        # Format the 'Forward Dividend & Yield' column
        dividend_rate = info.get('dividendRate', 'Not provided')
        dividend_yield = info.get('dividendYield', 'Not provided')
        dividend_yield_formatted = f'{dividend_rate} ({dividend_yield * 100:.2f}%)'
        company_stats_right['Value'].loc[company_stats_right.index == 'Forward Dividend & Yield'] = dividend_yield_formatted
        col2.table(company_stats_right)
        
        # Add chart
        st.write('**4. Stock Prices Chart:**')
        
        #Create three columns layout
        col1, col2, col3 = st.columns(3)
       
        chart_type = col1.selectbox("Select Chart Type", ["Candlestick Chart", "Line Chart"], key="tab1_chart_type")
        duration = col2.selectbox('Select Duration', ['None', '1mo', '3mo', '6mo', 'YTD', '1y', '3y', '5y', 'max'], key="tab1_duration")
        time_interval = col3.selectbox("Select Time Interval", ["1d", "1mo", "3mo"], key="tab1_time_interval")
        
        @st.cache_data
        def GetStockData(ticker, time_interval, start_date, end_date):
            stock_df = yf.Ticker(ticker).history(interval=time_interval, start=start_date, end=end_date)
            stock_df.reset_index(inplace=True)
            stock_df['Date'] = stock_df['Date'].dt.date
            return stock_df
        
        # To differentiate between selecting the duration and using start/end dates
        if duration == 'None':
            duration = None
            

        if ticker:
            stock_price = None
            if duration:
                historical_data = yf.Ticker(ticker).history(period=duration, interval=time_interval)
                stock_price = pd.DataFrame(historical_data)
                stock_price.reset_index(inplace=True)
            else:
                stock_price = GetStockData(ticker, time_interval, start_date, end_date)
             
            if stock_price is not None:
                            
                fig = go.Figure()

                if chart_type == "Candlestick Chart":
                    fig.add_trace(go.Candlestick(
                        x=stock_price['Date'],
                        open=stock_price['Open'],
                        high=stock_price['High'],
                        low=stock_price['Low'],
                        close=stock_price['Close'],
                        name='CandleStick',
                        yaxis='y1'
                    ))
                else:  # Line Chart
                    fig.add_trace(go.Scatter(
                        x=stock_price['Date'],
                        y=stock_price['Close'], 
                        mode='lines', 
                        name='Close Price',
                        yaxis= 'y1'
                        ))
                    
                # Add volume bar chart
                fig.add_trace(go.Bar(x=stock_price['Date'],
                                     y=stock_price['Volume'],
                                     name='Volume',
                                     yaxis='y2',
                                     marker=dict(color='rgba(255, 50, 100, 0.5)')
                                     )
                              )
                
                # Update layout based on time interval selected
                fig.update_layout(
                    title=f'{ticker} Stock Price',
                    xaxis_title='Date',
                    xaxis_rangeslider_visible=False,
                    yaxis=dict(title='Close Price', side='left'),
                    yaxis2=dict(title='Volume', side='right', overlaying='y'),
                    annotations=[
                        dict(
                            text="Select 'CandleStick'/'Close Price' or Volume to display one of them or select both.",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=1.1,
                            y=1.3,
                            bordercolor="#c7c7c7",
                            font=dict(size=10),
                            )
                                ],
                    legend=dict(
                            x=0.5,  
                            y=1.25
                               )
                                )

                st.plotly_chart(fig, use_container_width=True)
       
        
        
#==============================================================================
# Tab 2
#==============================================================================

def render_tab2():
    """
    This function render the Tab 2 - Chart of the dashboard.
    """
  
    st.write('**Stock Prices Chart**')
    #Create three columns layout
    col1, col2, col3 = st.columns(3)
    
    chart_type = col1.selectbox("Select Chart Type", ["Candlestick Chart", "Line Chart"], key="tab2_chart_type")
    duration = col2.selectbox('Select Duration', ['None', '1mo', '3mo', '6mo', 'YTD', '1y', '3y', '5y', 'max'], key="tab2_duration")
    time_interval = col3.selectbox("Select Time Interval", ["1d", "1mo", "3mo"], key="tab2_time_interval")
    
    @st.cache_data
    def GetStockData(ticker, time_interval, start_date, end_date):
        stock_df = yf.Ticker(ticker).history(interval=time_interval, start=start_date, end=end_date)
        stock_df.reset_index(inplace=True)
        stock_df['Date'] = stock_df['Date'].dt.date
        return stock_df
    
    # To differentiate between selecting the duration and using start/end dates
    if duration == 'None':
        duration = None
        

    if ticker:
        stock_price = None
        if duration:
            historical_data = yf.Ticker(ticker).history(period=duration, interval=time_interval)
            stock_price = pd.DataFrame(historical_data)
            stock_price.reset_index(inplace=True)
        else:
            stock_price = GetStockData(ticker, time_interval, start_date, end_date)

         # Show data table
        show_data = st.checkbox("Show Data Table")
    
        if show_data:
             st.write('**Stock Price Data**')
             st.dataframe(stock_price, hide_index=True, use_container_width=True)
         
        if stock_price is not None:
                        
            fig = go.Figure()

            if chart_type == "Candlestick Chart":
                fig.add_trace(go.Candlestick(
                    x=stock_price['Date'],
                    open=stock_price['Open'],
                    high=stock_price['High'],
                    low=stock_price['Low'],
                    close=stock_price['Close'],
                    name='CandleStick',
                    yaxis='y1'
                ))
            else:  # Line Chart
                fig.add_trace(go.Scatter(
                    x=stock_price['Date'],
                    y=stock_price['Close'], 
                    mode='lines', 
                    name='Close Price',
                    yaxis= 'y1'
                    ))
                
            # Add volume bar chart
            fig.add_trace(go.Bar(x=stock_price['Date'],
                                 y=stock_price['Volume'],
                                 name='Volume',
                                 yaxis='y2',
                                 marker=dict(color='rgba(255, 50, 100, 0.5)')
                                 )
                          )
            
            # Update layout based on time interval selected
            fig.update_layout(
                title=f'{ticker} Stock Price',
                xaxis_title='Date',
                xaxis_rangeslider_visible=False,
                yaxis=dict(title='Close Price', side='left'),
                yaxis2=dict(title='Volume', side='right', overlaying='y'),
                annotations=[
                    dict(
                        text="Select 'CandleStick'/'Close Price' or Volume to display one of them or select both.",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=1.1,
                        y=1.3,
                        bordercolor="#c7c7c7",
                        font=dict(size=10),
                        )
                            ],
                legend=dict(
                        x=0.5,  
                        y=1.25
                           )
                            )


            st.plotly_chart(fig, use_container_width=True)
            

#==============================================================================
# Tab 3
#==============================================================================

def render_tab3():
    """
    This function render the Tab 3 - Financials of the dashboard.
    """
    
    #Create two columns layout
    col1, col2 = st.columns(2)
 
    # Create a selection box for financial statements
    financial_statement = col1.selectbox('Select a Financial Statement:', ['Income Statement', 'Balance Sheet', 'Cash Flow'])
 
    # Create a selection box for the period
    period = col2.selectbox('Select a Period:', ['Yearly', 'Quarterly'])
    
    if ticker != '':
        # Fetch financial data 
        if financial_statement == 'Income Statement':
            financial_data = yf.Ticker(ticker).get_income_stmt(freq=period.lower())
        elif financial_statement == 'Balance Sheet':
            financial_data = yf.Ticker(ticker).get_balance_sheet(freq=period.lower())
        elif financial_statement == 'Cash Flow':
            financial_data = yf.Ticker(ticker).get_cash_flow(freq=period.lower())
     
        # Display financial data
        st.write(f'**{ticker} {financial_statement} ({period}):**')
        st.write(financial_data)
        

#==============================================================================
# Tab 4
#==============================================================================

def render_tab4():
    """
    This function render the Tab 4 - Monte Carlo Simulation.
    """
    st.write('**Monte Carlo Simulation**')
    
    #Create three columns layout
    col1, col2 = st.columns(2)
    
    n_simulations = col1.selectbox("Select number of simulations", [200, 500, 1000])
    time_horizon = col2.selectbox("Select time horizon (days)", [30, 60, 90])
    
    @st.cache_data
    def GetStockData(ticker, start_date, end_date):
        stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        return stock_df
    
    
    if ticker: 
        stock_price= GetStockData(ticker, start_date, end_date)
        
        if stock_price is not None:
            
            daily_returns = stock_price['Close'].pct_change()
            daily_volatility = daily_returns.std()
            simulated_df = pd.DataFrame()
            
            for r in range(0, n_simulations):
            
                stock_price_list = []
                current_price = stock_price['Close'].iloc[-1]
            
                for i in range(0, time_horizon):
            
                    # Generate daily return
                    daily_return = np.random.normal(0, daily_volatility, 1)[0]
            
                    # Calculate the stock price of next day
                    future_price = current_price * (1 + daily_return)
            
                    # Save the results
                    stock_price_list.append(future_price)
            
                    # Change the current price
                    current_price = future_price
                    
                # Store the simulation results
                simulated_col = pd.Series(stock_price_list)
                simulated_col.name = "Sim" + str(r)
                simulated_df = pd.concat([simulated_df, simulated_col], axis=1) 
                
            # Display the plot in Streamlit
            st.line_chart(simulated_df)
            
            st.write(f"**Last price: {stock_price['Close'].iloc[-1]}**")
            st.write(f"**Price at 5th percentile: {np.percentile(simulated_df.iloc[-1, ], 5)}**")
            
            ValueatRisk = np.percentile(simulated_df.iloc[-1, ], 5) - stock_price['Close'].iloc[-1]
            st.write(f"**Value at Risk at 95% confidence interval: {ValueatRisk}**")
        

#==============================================================================
# Tab 5
#==============================================================================

def render_tab5():
    """
    This function render the Tab 5 - Stocks Comparison.
    """
    
    # Get user input for tickers
    stock_tickers = st.text_input("Enter multiple tickers (separated by a space):")

    if stock_tickers:
        stocks = stock_tickers.split()
        data = yf.download(stocks, start_date, end_date)  

        if not data.empty:
            st.write("#### Table Stock Prices Comparison")
            st.write(data["Close"])
            
            st.write("#### Chart Stock Prices Comparison")
            # Display a line chart for comparison
            st.line_chart(data["Close"])



#==============================================================================
# Main body
#==============================================================================
      
# Render the header
render_header()
# Render the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Chart", "Financials", "Monte Carlo Simulation","Stocks Comparison"])
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()
with tab5:
    render_tab5()
    
# Customize the dashboard with CSS
st.markdown(
    """
    <style>
        .stApp {
            background: #F0F8FF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    
###############################################################################
# END
###############################################################################