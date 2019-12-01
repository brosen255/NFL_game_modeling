import bs4
import requests
import urllib.request
import httplib2
from bs4 import BeautifulSoup
from bs4 import SoupStrainer
import pandas as pd
import numpy as np


def weather_score_scraper(curr_week):
    start_page = 'http://www.nflweather.com/en/week/'
    table_cols = ['away','home','score','forecast','wind']
    all_df = pd.DataFrame(columns = table_cols)

    year = '2019'
    for week in range(2,curr_week+1):
       
        scraping_page = start_page + year + '/week-{}'.format(week)
        print('scraping:',scraping_page)
        resp = urllib.request.urlopen(scraping_page)
        soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'))
        page = requests.get(scraping_page)
        soup = BeautifulSoup(page.content, 'html.parser')
        table = soup.find('table')

        table_rows = table.find_all('tr')

        scraped_data = []

        for tr in table_rows:
            td = tr.find_all('td')
            row = [tr.text for tr in td]
            row1 = [str(i).strip() for i in row if '\n' in i]
            row1_wind = [str(i).strip() for i in row if 'm ' in i]
            if len(row1_wind) > 1:
                row1_wind = list(row1_wind)[-1]
            row1 = row1 + row1_wind

            row1 = [i for i in row1 if i != 'Details']
            row1 = [i for i in row1 if i != '']
            if str(row1) != '[]':
                if len(row1) != 5:
                    break
                else:
                    #print('added {}'.format('week',week))
                    scraped_data.append(row1)

        temp_df = pd.DataFrame(scraped_data,columns = table_cols)
        temp_df['week'] = week
        all_df = pd.concat([all_df,temp_df],axis = 0).reset_index().drop('index',axis =1)

    all_df['temp'] = all_df.forecast.apply(lambda x: x.split(' ')[0])
    all_df['forecast'] = all_df.forecast.apply(lambda x: x.split(' ')[1:]).apply(' '.join)
    all_df.loc[all_df.temp != 'DOME','temp'] = all_df.temp.str.replace('f','')

    all_df.loc[all_df['week'] == curr_week,'score'] = ''


    outpath = '/Users/Ben/Desktop/pwd_test/'
    outfilename = 'weather_score_2019_df.csv'
    all_df.to_csv(outpath + outfilename)

    return all_df












