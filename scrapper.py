from os import write
from numpy import full
from pandas_datareader import data as pdr
import pandas as pd
import csv
from datetime import datetime
import time
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin
from pyrate_limiter import Duration, Rate, Limiter, InMemoryBucket
import time
import requests_cache
from pathlib import Path
from curl_cffi import requests as request 

# import warnings
# warnings.simplefilter(action="ignore", category=FutureWarning)

import yfinance as yf

# yf.pdr_override()

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    # print("Hello World")
    pass

session = CachedLimiterSession(
    limiter=Limiter(Rate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
    bucket_class=InMemoryBucket,
    backend=SQLiteCache("yfinance.cache")
)


current_datetime = datetime.now()
print(current_datetime)

import requests_cache

session = requests_cache.CachedSession('yfinance.cache')
session.headers['User-agent'] = 'my-program/1.0'

Places = ['SZ']
Placess=['SS']
## we start with the place first, let say shanghai 

import requests
import yfinance as yf
import urllib.parse  # Add this import statement

###Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.13 Safari/537.36
##Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.13 Safari/537.36
####Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36
###Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2503.0 Safari/537.36
####Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.155 Safari/537.36
###Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36


class YFinance:
    user_agent_key = "User-Agent"
    # user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
    #                     "AppleWebKit/537.36 (KHTML, like Gecko) "
    #                     "Chrome/58.0.3029.110 Safari/537.36")

    # user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
    #                 "AppleWebKit/537.36 (KHTML, like Gecko) "
    #                 "Chrome/43.0.2357.134 Safari/537.36")

    # user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) " 
    #                     "AppleWebKit/537.36 (KHTML, like Gecko) " 
    #                     "Chrome/44.0.2403.155 Safari/537.36")

    user_agent_value = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None
        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com", ## fc.yahoo.com
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

    def get_history(self, period="5y"):
        # Obtain cookies and crumb for downloading historical data
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)

        # Fetch historical stock data using yfinance
        stock = yf.Ticker(self.yahoo_ticker)
        print("gotten history")
        return stock.history(period=period)

    @property
    def info(self):
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("summaryDetail,"
                         "financialData,"
                         "quoteType,"
                         "assetProfile,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"  # Use urllib.parse.quote_plus
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")  # Use urllib.parse.quote_plus

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
    

class scrapper():

    stock_price_shenzhen = r"stock_list/list_of_shenzhen.txt"
    stock_price_shanghai = r"stock_list/list_of_shanghai.txt"
    stock_price_america = r"stock_list/nasdaqlisted.txt"
    list_for_shanghai =[]
    list_for_shenzhen =[]
    list_for_america = []
    area = ""

    def __init__(self,area,spectific_id=None,full_scale= False):
        self.area = area
        self.spectific_id=spectific_id
        if full_scale:
            print("Full scale scrapping")
            self.America(full_scale=True)
        if spectific_id is not None:
            print("Specific ID provided: " + spectific_id)
            self.scrap_specific_id_from_list([spectific_id], area)
        else:
            if area == "America":
                self.America()
            elif area == "SS":
                self.shanghai()
            elif area == "SZ":
                self.shenzhen()
            elif area == "China":
                self.China()

    def run(self, *args):
        print(*args)

    def scrap_specific_id_from_list(self, index_list, area):
        self.write_to_file(index_list, area)
    
    def write_to_file(self,index_list,area,full_scale=False):
        for indexes in index_list:
            try:
                print("Now working on "+indexes)
                session = request.Session(impersonate="chrome")

                filename = 'stock_data/'+str(area)+"/"+indexes+'.txt'
                pd.read_csv(filename)
                refreshing = False
                if Path(filename).is_file(): ## check if the destinated files exist or not 
                    print("Checking the last updated date")
                    last_updated_date = pd.read_csv(filename, header=None, quoting=csv.QUOTE_NONNUMERIC).iloc[-1, 0]
                    current_dataframe = pd.read_csv(filename, header=None, quoting=csv.QUOTE_NONNUMERIC,names=['Date','Close','High','Low','Open','Volume'],on_bad_lines='skip',date_format="%Y-%m-%d")
                    last_updated_date = pd.to_datetime(last_updated_date.split(" ")[0], format='%Y-%m-%d')
                    if last_updated_date.strftime('%Y-%m-%d') == current_datetime.strftime("%Y-%m-%d"):
                        print("Already updated")
                        continue
                    else:
                        print((current_datetime-last_updated_date).days)
                        df = pd.DataFrame()
                        print(current_datetime)
                        print(last_updated_date)

                        days_difference = (current_datetime - last_updated_date).days
                        if full_scale:
                            days_difference = 9999  # Force refresh for full scale

                        match (days_difference) :
                            # case 0|1:
                            #     print("Already updated and should be ready to use")
                            #     continue
                            case 0|1|2:
                                # if current_datetime.hour < 20 and (current_datetime-last_updated_date).days == 1:
                                #     continue
                                # yf.download
                                # df = yf.download(indexes, start=last_updated_date.strftime('%Y-%m-%d'), auto_adjust=True, session=session, progress=False, threads=True)
                                # print(df)
                                ticker = yf.Ticker(indexes, session=session)
                                df = ticker.history(period='1d', auto_adjust=True)
                                # print(df)
                                print("Updated for 1 day")
                            case 3|4|5:
                                ticker = yf.Ticker(indexes, session=session)
                                # df = yf.download(indexes, start=last_updated_date.strftime('%Y-%m-%d'), auto_adjust=True, session=session, progress=False, threads=True)

                                df = ticker.history(period='5d', auto_adjust=True)
                                # df = pd.concat([current_dataframe, df]).drop_duplicates(inplace=True).reset_index(drop=True)
                                # print(df)
                                print("Updated for 5 days")
                            case _:
                                print("Refreshing for 5 years")
                                ticker = yf.Ticker(indexes, session=session)
                                refreshing = True
                                df = ticker.history(period='5y', auto_adjust=True)
                                # print("retreived the data for 5 years")
                                # print(df)
                                print(df)

                
                
                # print(df)

                # a= YFinance(indexes)
                # df = YFinance.get_history(a)

                # a= yf.Ticker(indexes) ## oringal one
                # df = a.history(period='5y',auto_adjust=True) ## original one

                # print(indexes)
                # df = yf.download(indexes, start="2020-03-07",auto_adjust=True,progress=False, threads=False)
                # print(df)

                # df = pdr.get_data_yahoo(indexes,start="2019-1-1",end= current_datetime)
                if df.empty:
                    print("Something is wrong. The dataset failed to fetch.")
                    time.sleep(1)
                    continue
                else:
                    print(indexes+" finished downloading")
                    ## current problem didn't save or further processing
                # break
                # df = pdr.get_data_yahoo('0'*(6-len(str(indexes+1)))+str(indexes+1)+ "." + place, start="2019-1-1", end=current_datetime,proxy="202.86.138.18:8080") #proxy="173.244.200.156:64631"
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
                ## I am thinking about resetting the data in this case
                print("Resetting the data")
                # df = yf.download(indexes, period="max", auto_adjust=True, session=session, progress=False, threads=True)
                ticker = yf.Ticker(indexes, session=session)
                df = ticker.history(period='5y', auto_adjust=True)
                if df.empty:
                    print("Something is wrong with the data")
                    print("Skipping this index")
                    print(f"Data for {indexes} is empty after reset.")
                    # raise ValueError(f"Data for {indexes} is empty after reset.")
                    continue
                else:
                    print(indexes+" finished downloading after reset")
                
                if 'Capital Gains' in df.columns:
                    df.drop(['Capital Gains'],axis = 1, inplace=True)
            
                if 'Dividends' in df.columns:
                    df=df.drop(['Dividends'],axis = 1)

                if 'Stock Splits' in df.columns:
                    df=df.drop(['Stock Splits'],axis = 1)

                df = df.reset_index().rename(columns={"index":"Date"})
                # print(df['Date'])
                # print(type(df['Date']))
                for col in df.select_dtypes(['datetimetz']).columns:
                    df[col] = df[col].dt.tz_localize(None)
                # df['Date'].dt.tz_localize(None)  # Convert Date to string
                # print("Date column after timezone localization:")
                # print(df['Date'])
                # df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce').tz_localize(None)  # Remove timezone localization
                # print(df['Date'])
                ## need to remove index 
                # df = pd.DataFrame(df, columns=['Date','Close','High','Low','Open','Volume'])
                # print(df)
                ## Converting it to the format
                # df = df[['Close','High','Low','Open','Volume']]
                # print(df)
                for i in df.index:
                    # print(i)
                    # if i == 'Date':
                        # continue
                    # print(i)
                    # continue
                    for j in df:
                        if j == 'Date':
                            continue
                        if type(df.loc[i,j]) == pd.Timestamp:
                            continue
                        try:
                            df.loc[i,j] = round(float(df.loc[i,j]),2)
                        except Exception:
                            continue
                
                print("After modification")
                print(df)
                df = df.astype(str)
                filename = 'stock_data/'+str(area)+"/"+indexes+'.txt'
                print("Turning to csv")
                df.to_csv('stock_data/'+str(area)+"/"+indexes+'.txt', header = False, quoting=csv.QUOTE_NONNUMERIC,mode="w",index=False)

                continue
            # else:
                
            #     try:

            #         if len(df) != 0:
            #             # print(df)
            #             if 'Dividends' in df.columns:
            #                 df.drop(['Dividends'],axis = 1, inplace=True)
            #             if 'Stock Splits' in df.columns:
            #                 df.drop(['Stock Splits'],axis = 1, inplace=True)

            #             if 'Capital Gains' in df.columns:
            #                 df.drop(['Capital Gains'],axis = 1, inplace=True)
                    
            #             # df = df[['Close','High','Low','Open','Volume']]

            #             for i in df.index:
            #                 for j in df:
            #                     if j == 'Date':
            #                         continue
            #                     if type(df.loc[i,j]) == pd.Timestamp:
            #                         continue
            #                     try:
            #                         df.loc[i,j] = round(float(df.loc[i,j]),2)
            #                     except Exception:
            #                         continue
                        
            #             try:
            #                 df = df.reset_index(drop=True)
            #             except ValueError as e:
            #                 if 'level_0' in str(e) or 'level_0' in df.columns:
            #                     if 'level_0' in df.columns:
            #                         df = df.rename(columns={'level_0': 'level_0_old'})
            #                     df = df.reset_index(drop=True)
            #                 else:
            #                     raise
            #             if 'Date' not in df.columns:
            #                 if 'index' in df.columns:
            #                     df = df.rename(columns={'index': 'Date'})
            #                 else:
            #                     df = df.rename(columns={df.columns[0]: 'Date'})
            #             df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
            #             # df.drop(['Dividends'],axis = 1, inplace=True)
            #             # df.drop(['Stock Splits'],axis = 1, inplace=True)

            #             if 'Capital Gains' in df.columns:
            #                 df.drop(['Capital Gains'],axis = 1, inplace=True)
                    
            #             # print('The original dataframe')
            #             current_dataframe.drop_duplicates(subset='Date',inplace=True, ignore_index=True)
            #             current_dataframe['Date'] = pd.to_datetime(current_dataframe['Date'], format='%Y-%m-%d', errors='coerce')
                        
            #             # print(current_dataframe)
            #             # print("The about to update dataframe")
            #             # print(df)

            #             # for col in current_dataframe.select_dtypes(['datetimetz']).columns:
            #             #     current_dataframe[col] = current_dataframe[col].dt.tz_localize(None)

            #             for col in df.select_dtypes(['datetimetz']).columns:
            #                 df[col] = df[col].dt.tz_localize(None)

            #             # print("Old")
            #             # print(current_dataframe['Date'])
            #             # print("New")
            #             # print(df['Date'])

            #             ## Here cause the trouble afterward

            #             # print(pd.concat([current_dataframe, df],ignore_index=True))
                        
            #             current_dataframe= pd.concat([current_dataframe, df],ignore_index=True)
            #             # current_dataframe['Date']=pd.to_datetime(current_dataframe['Date'], format='%Y-%m-%d', errors='coerce')
            #             print(current_dataframe)
            #             current_dataframe=current_dataframe.drop_duplicates(subset='Date', inplace=False, ignore_index=True,keep='last')
            #             current_dataframe.dropna(inplace=True)
            #             for col in current_dataframe.select_dtypes(['datetimetz']).columns:
            #                 current_dataframe[col] = current_dataframe[col].dt.tz_localize(None)

            #             # print(current_dataframe)

            #             # print("the integrated dataframe")
            #             # print(current_dataframe)
            #             current_dataframe = current_dataframe.astype(str)
            #             df = current_dataframe
            #             # print("The final dataframe")
            #             # print(df)
            #             # break

            #             # print(df)
            #             filename = 'stock_data/'+str(area)+"/"+indexes+'.txt'
            #             if Path(filename).exists() == False or refreshing:
            #                 Path("stock_data").mkdir(parents=True, exist_ok=True)
            #                 Path("stock_data/"+str(area)).mkdir(parents=True, exist_ok=True)
            #                 df.to_csv('stock_data/'+str(area)+"/"+indexes+'.txt', header = False, quoting=csv.QUOTE_NONNUMERIC,index=False)
            #             else:
            #                 df.to_csv('stock_data/'+str(area)+"/"+indexes+'.txt', header = False, quoting=csv.QUOTE_NONNUMERIC,mode="w",index=False)

            #             time.sleep(1)  # Sleep for 1 second to avoid hitting the rate limit
            #         else:
            #             print("This Code Doesn't Exist")
            #     except Exception as e:
            #         print(f"An error occurred while processing {indexes}: {e}")
            #         print("Skipping this index")
            #         continue
            finally:
                print("Loading Next Data")
                time.sleep(1)

    def America(self,full_scale=False):
        f=open(self.stock_price_america,'r',encoding="utf8")
        strings = f.read().split("\n")
        strings=strings[1:-1]
        # print(strings)
        for string in strings:
            string= string.split("|",1)[0]
            # print(string)
            self.list_for_america.append(string)     

        # self.list_for_america = ["AMZN","AMZZ","AMZU"]  
        if full_scale:
            self.write_to_file(self.list_for_america,"America",True)
        else:
            self.write_to_file(self.list_for_america,"America")

    def shenzhen(self):
        f=open(self.stock_price_shenzhen,'r',encoding="utf8")
        strings = f.read().split("\n")
        strings=strings[1:-1]
        # print(strings)
        for string in strings:
            string= string.split(" ",1)[0]
            # print(string)
            self.list_for_shenzhen.append(string+".SZ")
        
        self.write_to_file(self.list_for_shenzhen,"SZ")

    def shanghai(self):

        f=open(self.stock_price_shanghai,'r',encoding="utf8")
        strings = f.read().split("\n")
        strings=strings[1:-1]
        # print(strings)
        for string in strings:
            string= string.split(" ",1)[0]
            # print(string)
            self.list_for_shanghai.append(string+".SS")

        self.write_to_file(self.list_for_shanghai,"SS")

    def China(self):
        self.shanghai()
        self.shenzhen()

def main():
    s = scrapper("China")
    # s.America()
    s.shenzhen()
    s.shanghai()

class scapper_with_thread(scrapper):
    def write_to_file(self,index_list,area,full_scale=False):
        for indexes in index_list:
            try:
                print("Now working on "+indexes)
                session = request.Session(impersonate="chrome")

                filename = 'stock_data/'+str(area)+"/"+indexes+'.txt'
                refreshing = False
                if Path(filename).exists():
                    print("Checking the last updated date")
                    last_updated_date = pd.read_csv(filename, header=None, quoting=csv.QUOTE_NONNUMERIC).iloc[-1, 0]
                    if last_updated_date == "":
                        print("File is empty, return error")
                        raise ValueError(f"File {filename} is empty or has no valid last updated date.")
                    current_dataframe = pd.read_csv(filename, header=None, quoting=csv.QUOTE_NONNUMERIC,names=['Date','Close','High','Low','Open','Volume'],on_bad_lines='skip')
                    if current_dataframe.empty:
                        print("DataFrame is empty, return error")
                        raise ValueError(f"DataFrame for {indexes} is empty or has no valid data.")
                    last_updated_date = pd.to_datetime(last_updated_date.split(" ")[0], format='%Y-%m-%d')
                    if last_updated_date.strftime('%Y-%m-%d') == current_datetime.strftime("%Y-%m-%d"):
                        print("Already updated")
                        continue
                    else:
                        print((current_datetime-last_updated_date).days)
                        df = pd.DataFrame()
                        print(current_datetime)
                        print(last_updated_date)

                        days_difference = (current_datetime - last_updated_date).days
                        if full_scale:
                            days_difference = 9999  # Force refresh for full scale

                        match (days_difference) :
                            # case 0|1:
                            #     print("Already updated and should be ready to use")
                            #     continue
                            case 0|1|2:
                                # if current_datetime.hour < 20 and (current_datetime-last_updated_date).days == 1:
                                #     continue
                                # yf.download
                                df = yf.download(indexes, start=last_updated_date.strftime('%Y-%m-%d'), auto_adjust=True, session=session, progress=False, threads=True,multi_level_index=False )

                                # ticker = yf.Ticker(indexes, session=session)
                                # df = ticker.history(period='1d', auto_adjust=True)
                                print("This is the df")
                                # print(df)
                                print("Updated for 1 day")
                            case 3|4|5:
                                # ticker = yf.Ticker(indexes, session=session)
                                df = yf.download(indexes, start=last_updated_date.strftime('%Y-%m-%d'), auto_adjust=True, session=session, progress=False, threads=True,multi_level_index=False )
                                print("This is the df")

                                # df = ticker.history(period='5d', auto_adjust=True)
                                # df = pd.concat([current_dataframe, df]).drop_duplicates(inplace=True).reset_index(drop=True)
                                # print(df)
                                print("Updated for 5 days")
                            case _:
                                print("Refreshing for 5 years")
                                # ticker = yf.Ticker(indexes, session=session)
                                refreshing = True
                                # df = ticker.history(period='5y', auto_adjust=True)
                                df = yf.download(indexes, period="5y", auto_adjust=True, session=session, progress=False, threads=True,multi_level_index=False )

                                # print(df)

                
                
                # print(df)

                # a= YFinance(indexes)
                # df = YFinance.get_history(a)

                # a= yf.Ticker(indexes) ## oringal one
                # df = a.history(period='5y',auto_adjust=True) ## original one
                # df = df[['Close','High','Low','Open','Volume']]

                # print(indexes)
                # df = yf.download(indexes, start="2020-03-07",auto_adjust=True,progress=False, threads=False)
                # print(df)

                # df = pdr.get_data_yahoo(indexes,start="2019-1-1",end= current_datetime)
                if df.empty:
                    print("Something is wrong")
                    continue
                else:
                    print(indexes+" finished downloading")
                # break
                # df = pdr.get_data_yahoo('0'*(6-len(str(indexes+1)))+str(indexes+1)+ "." + place, start="2019-1-1", end=current_datetime,proxy="202.86.138.18:8080") #proxy="173.244.200.156:64631"
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
                ## I am thinking about resetting the data in this case
                print("Resetting the data")
                # ticker = yf.Ticker(indexes, session=session)
                # df = ticker.history(period='5y', auto_adjust=True)
                df = yf.download(indexes, period="5y", auto_adjust=True, session=session, progress=False, threads=True,multi_level_index=False )
                # df = df[['Close','High','Low','Open','Volume']]
                # print(df)
                if df.empty:
                    print("Something is wrong with the data")
                    print("Skipping this index")
                    print(f"Data for {indexes} is empty after reset.")
                    # raise ValueError(f"Data for {indexes} is empty after reset.")
                    continue
                else:
                    print(indexes+" finished downloading after reset")
                
                # if 'Capital Gains' in df.columns:
                #     df.drop(['Capital Gains'],axis = 1, inplace=True)
            
                # df=df.drop(['Dividends'],axis = 1)
                # df=df.drop(['Stock Splits'],axis = 1)
                # print(df)
                ## Converting it to the format
                # df = df[['Close','High','Low','Open','Volume']]
                # print(df)
                for i in df.index:
                    for j in df:
                        if j == 'Date':
                            continue
                        try:
                            df.loc[i,j] = round(float(df.loc[i,j]),2)
                        except Exception:
                            continue
                
                # print(df)
                df = df.astype(str)
                filename = 'stock_data/'+str(area)+"/"+indexes+'.txt'
                df.to_csv('stock_data/'+str(area)+"/"+indexes+'.txt', header = False, quoting=csv.QUOTE_NONNUMERIC,mode="w")

                continue
            else:
                print("Now creating the files")
                df = yf.download(indexes, period="5y", auto_adjust=True, session=session, progress=False, threads=True,multi_level_index=False )

                try:

                    if len(df) != 0:
                        # print(df)
                        if 'Dividends' in df.columns:
                            df.drop(['Dividends'],axis = 1, inplace=True)
                        # df.drop(['Dividends'],axis = 1)
                        if 'Stock Splits' in df.columns:
                            df.drop(['Stock Splits'],axis = 1, inplace=True)
                        # df.drop(['Stock Splits'],axis = 1)

                        if 'Capital Gains' in df.columns:
                            df.drop(['Capital Gains'],axis = 1, inplace=True)
                    
                        # df = df[['Close','High','Low','Open','Volume']]
                        print(df)

                        for i in df.index:
                            for j in df:
                                if j == 'Date':
                                    continue
                                if type(df.loc[i,j]) == float or type(df.loc[i,j]) == pd.Timestamp:
                                    try:
                                        df.loc[i,j] = round(float(df.loc[i,j]),2)
                                    except Exception:
                                        continue
                        
                        try:
                            df = df.reset_index(drop=True)
                        except ValueError as e:
                            if 'level_0' in str(e) or 'level_0' in df.columns:
                                if 'level_0' in df.columns:
                                    df = df.rename(columns={'level_0': 'level_0_old'})
                                df = df.reset_index()
                            else:
                                raise
                        if 'Date' not in df.columns:
                            if 'index' in df.columns:
                                df = df.rename(columns={'index': 'Date'})
                            else:
                                df = df.rename(columns={df.columns[0]: 'Date'})
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                        
                        # print(df)
                        # df.drop(['Dividends'],axis = 1, inplace=True)
                        # df.drop(['Stock Splits'],axis = 1, inplace=True)

                        # if 'Capital Gains' in df.columns:
                        #     df.drop(['Capital Gains'],axis = 1, inplace=True)
                    
                        # print('The original dataframe')
                        # print(current_dataframe)
                        current_dataframe.drop_duplicates(subset='Date',inplace=True, ignore_index=True)
                        current_dataframe['Date'] = pd.to_datetime(current_dataframe['Date'],format='%Y-%m-%d', errors='coerce')
                        
                        # print(current_dataframe)
                        # print("The about to update dataframe")
                        # print(df)
                        ## Here cause the trouble afterward

                        # print(pd.concat([current_dataframe, df],ignore_index=True))
                        
                        current_dataframe= pd.concat([current_dataframe, df],ignore_index=True)
                        # current_dataframe['Date']=pd.to_datetime(current_dataframe['Date'], format='%Y-%m-%d', errors='coerce')
                        # print(current_dataframe)
                        current_dataframe=current_dataframe.drop_duplicates(subset='Date', inplace=False, ignore_index=True,keep='last')
                        current_dataframe.dropna(inplace=True)
                        # print("the integrated dataframe")
                        # print(current_dataframe)
                        current_dataframe = current_dataframe.astype(str)
                        df = current_dataframe
                        # print(df)

                        filename = 'stock_data/'+str(area)+"/"+indexes+'.txt'
                        if Path(filename).exists() == False or refreshing:
                            Path("stock_data").mkdir(parents=True, exist_ok=True)
                            Path("stock_data/"+str(area)).mkdir(parents=True, exist_ok=True)
                            df.to_csv('stock_data/'+str(area)+"/"+indexes+'.txt', header = False, quoting=csv.QUOTE_NONNUMERIC,index=False)
                        else:
                            df.to_csv('stock_data/'+str(area)+"/"+indexes+'.txt', header = False, quoting=csv.QUOTE_NONNUMERIC,mode="w",index=False)

                        # time.sleep(1)  # Sleep for 1 second to avoid hitting the rate limit
                    else:
                        print("This Code Doesn't Exist")
                # except ValueError as ve:
                #     print(f"ValueError occurred while processing {indexes}: {ve}")
                #     print("Skipping this index")
                #     continue
                finally:
                    print("Loading Next Data")
            finally:
                print("Loading Next Data")


if __name__ == "__main__":
    import sys,time

    # time1 = time.time_ns()
    main()

    # s= scapper_with_thread(area="America")
    # time2 = time.time_ns()
    # print("Time taken: ", (time2 - time1) / 1e9, "seconds")

    # time1 = time.time_ns()
    # if len(sys.argv) == 2:
    #     s = scrapper(sys.argv[1])
    # else:
    #     s = scrapper(sys.argv[1], sys.argv[2])

    # time2 = time.time_ns()
    # print("Time taken: ", (time2 - time1) / 1e9, "seconds")


    # if sys.argv[2] != None:
    #     s = scrapper(sys.argv[1], sys.argv[2])
    # else:
    #     s = scrapper(sys.argv[1])  # Pass the area as a command line argument
    # # s = scrapper("America")




#     # print(int(string[:6]))

# # # print(list_for_shanghai)
# s=open(stock_price_shenzhen,'r',encoding="utf8")
# strings123 = s.read().split("\n")
# strings123=strings123[1:-1]
# for string in strings123:
#     # print(string[:6])
#     list_for_shenzhen.append(int(string[:6]))

# # SS_list= []

# # print("Hello")
# for place in Places:
#     for indexes in list_for_shenzhen:
#         try:
#             a= YFinance('0'*(6-len(str(indexes+1)))+str(indexes+1)+ "." + place)
#             print("Now loading")

#             df = YFinance.get_history(a)
#             print("Complete")


#             # df = pdr.get_data_yahoo('0'*(6-len(str(indexes+1)))+str(indexes+1)+ "." + place, start="2019-1-1", end=current_datetime,proxy="202.86.138.18:8080") #proxy="173.244.200.156:64631"
#         except:
#             pass
#         else:
#             if len(df) != 0:
#                 df.drop(['Dividends'],axis = 1)
#                 df.drop(['Stock Splits'],axis = 1)

#                 df = df[['Close','High','Low','Open','Volume']]

#                 for i in df.index:
#                     for j in df:
#                         df.loc[i,j] = round(df.loc[i,j],2)

#                 df = df.astype(str)
                
#                 df.to_csv('/home/ricky/Documents/Stockid_data/'+'0'*(6-len(str(indexes+1)))+str(indexes+1)+"." + place+'.txt', header = False, quoting=csv.QUOTE_NONNUMERIC)
#             else:
#                 print("This Code Doesn't Exist")
#         finally:
#             print("Loading Next Data")

#         time.sleep(6)

        # for indexes in self.list_for_america:
        #     try:
        #         # a= YFinance(indexes)
        #         # df = YFinance.get_history(a)

        #         a= yf.Ticker(indexes) ## oringal one
        #         df = a.history(period='5y',auto_adjust=True) ## original one

        #         # print(indexes)
        #         # df = yf.download(indexes, start="2020-03-07",auto_adjust=True,session= session)
        #         # print(df)

        #         # df = pdr.get_data_yahoo(indexes,start="2019-1-1",end= current_datetime)
        #         # print(df)
        #         # break
        #         # df = pdr.get_data_yahoo('0'*(6-len(str(indexes+1)))+str(indexes+1)+ "." + place, start="2019-1-1", end=current_datetime,proxy="202.86.138.18:8080") #proxy="173.244.200.156:64631"
        #     except:
        #         pass
        #     else:
        #         if len(df) != 0:
        #             # df.drop(['Dividends'],axis = 1)
        #             # df.drop(['Stock Splits'],axis = 1)

        #             df = df[['Close','High','Low','Open','Volume']]
        #             # break

        #             for i in df.index:
        #                 for j in df:
        #                     df.loc[i,j] = round(df.loc[i,j],2)

        #             df = df.astype(str)
                    
        #             df.to_csv('W:\Trading\Stockid_data\ '+indexes+'.txt', header = False, quoting=csv.QUOTE_NONNUMERIC)
        #         else:
        #             print("This Code Doesn't Exist")
        #     finally:
        #         print("Loading Next Data")

        # # time.sleep(5)

        