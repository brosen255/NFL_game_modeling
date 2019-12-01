import bs4
import requests
import urllib.request
import httplib2
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd
import numpy as np

def nfl_scraper():
	start_page = 'https://www.teamrankings.com/nfl/stats/'
	resp = urllib.request.urlopen(start_page)
	soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'))

	table_urls = []
	for link in soup.find_all('a', href=True):
	    if '/nfl/stat/' in link['href']:
	        table_urls.append(link['href'])

	## gets list of dates needed for each stat
	filename = 'spreadspoke_scores'
	ext = '.csv'
	path = '/Users/Ben/Downloads/nfl-scores-and-betting-data/'

	game_data_since_66 = pd.read_csv(path + filename + ext)

	cutoff_year = 2003
	cutoff_year = 2019

	o_three = game_data_since_66[game_data_since_66['schedule_season'] >= cutoff_year]

	thresh_date = o_three[o_three.columns[:3]].drop_duplicates(subset = ['schedule_season','schedule_week'], keep = 'last')
	thresh_date = thresh_date.reset_index().drop('index',axis = 1)
	thresh_date['week_sep'] = pd.to_datetime(thresh_date['schedule_date'])
	thresh_date['week_sep'] = pd.DatetimeIndex(thresh_date.week_sep) + pd.DateOffset(1)
	date_to_week_dict = dict(zip(thresh_date.week_sep,thresh_date.schedule_week))

	print('Grabbing the dates that separate NFL weeks')
	list_of_dates = thresh_date['week_sep'].values
	list_of_dates = [str(i) for i in list_of_dates]
	list_of_dates = [i[:i.index('T')] for i in list_of_dates]

	homebase = 'https://www.teamrankings.com'

	iterative_df = pd.DataFrame()
	iterative_df['team'] = ''
	iterative_df['date'] = ''
	# iterative_df = iterative_df.set_index(['team','date'],inplace = True)

	# table_urls = table_urls[:7]
	# list_of_dates = list_of_dates[:5]

	for stat_url_ext in table_urls:
		end_of_stat_df = pd.DataFrame()
		end_of_stat_df['team'] = ''
		end_of_stat_df['date'] = ''
		
		progress_stats = (table_urls.index(stat_url_ext)/len(table_urls)) * 100
		print(progress_stats,'%')

		for date in list_of_dates:
			progress_stats = (table_urls.index(stat_url_ext)/len(table_urls)) * 100
			#progress_dates = round(list_of_dates.index(date)/len(list_of_dates),2) * 10
			#print('PROGRESS : ' + str(progress_stats + progress_dates) + '%')
			#print(progress_stats + '%')
			#curr_stat_index = table_urls.index(stat_url_ext)

			statistics_page = homebase + stat_url_ext + '?date=' + date

			page = requests.get(statistics_page)
			soup = BeautifulSoup(page.content, 'html.parser')
			table = soup.find('table')
			table_rows1 = table.find_all('th')

			table_cols = []
			for th in table_rows1:
			    th = str(th)
			    end_idx = th.index('</th>')
			    table_cols.append(th[end_idx-6:end_idx])
			table_cols = [i.replace('>','') for i in table_cols]
			quote = '"'
			table_cols = [i.replace(quote,'') for i in table_cols]

			table_rows = table.find_all('tr')
			scraped_data = []
			for tr in table_rows:
			    td = tr.find_all('td')
			    row = [tr.text for tr in td]
			    #ow += str(date)
			    scraped_data.append(row)
			scraped_data = list(filter(None, scraped_data))
			scraped_frame = pd.DataFrame(scraped_data,columns = table_cols)
			scraped_frame['date'] = date
			table_cols.append('date')
			old_names = [s for s in table_cols if s.isdigit()]
			new_names = ['current_year','last_year']
			dd = dict(zip(old_names,new_names))
			updated_col_names = [dd.get(item,item)  for item in table_cols]
			scraped_frame.columns = updated_col_names
			scraped_frame.columns = [i.replace(' ','_').lower() for i in scraped_frame.columns]

			scraped_frame.columns = [i+'_'+str(stat_url_ext) for i in scraped_frame.columns]
			old_team = [i for i in scraped_frame.columns if 'team' in i]
			new_team = [i[:i.index('team')+4] for i in scraped_frame.columns if 'team' in i]
			dict_ = dict(zip(old_team,new_team))
			scraped_frame.columns = [dict_.get(item,item)  for item in scraped_frame.columns]

			old_date = [i for i in scraped_frame.columns if 'date' in i]
			new_date = [i[:i.index('date')+4] for i in scraped_frame.columns if 'date' in i]
			dict_ = dict(zip(old_date,new_date))
			scraped_frame.columns = [dict_.get(item,item)  for item in scraped_frame.columns]
			#scraped_frame['current_week'] = scraped_frame.date.map(date_to_week_dict)
			scraped_frame.drop(scraped_frame.columns[0],axis=1,inplace = True)

			#scraped_frame.set_index(['team','date'],inplace = True)
			end_of_stat_df= pd.concat([end_of_stat_df,scraped_frame], axis = 0)

		if table_urls[0] == stat_url_ext:
			iterative_df = iterative_df.merge(end_of_stat_df,how = 'outer',on = ['team','date'])
			
		if table_urls[0] != stat_url_ext:
			iterative_df = iterative_df.merge(end_of_stat_df,how = 'left',on = ['team','date'])

	iterative_df = iterative_df.set_index(['team','date'])


	old_slash = [i for i in iterative_df.columns if '/nfl/stat/' in i]
	new_slash = [i.replace('/nfl/stat/','') for i in iterative_df.columns if '/nfl/stat/' in i]
	dict_ = dict(zip(old_slash,new_slash))
	iterative_df.columns = [dict_.get(item,item)  for item in iterative_df.columns]

	print("Iterative DF Shape: " + str(iterative_df.shape))
	print('finished creating the dataframe')
	out_path = '/Users/Ben/Desktop/pwd_test/'
	out_filename = 'NFL_statistics_{}_mostrecent'.format(cutoff_year)
	out_ext = '.csv'
	file_locashe = out_path+out_filename+out_ext
	print('File location: ' + str(file_locashe))
	iterative_df.to_csv(file_locashe)
	return iterative_df


