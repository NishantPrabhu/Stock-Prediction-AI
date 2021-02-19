
# Script to scrape news for some company

import os
import time
import json
import pickle
from unidecode import unidecode
from selenium import webdriver
from selenium.webdriver.common.keys import Keys 


class Scraper:

    def __init__(self, base_url, ticker, stop_year):
        self.driver = webdriver.Firefox()
        self.driver.get(base_url)
        self.ticker = ticker
        self.buffer_url = None
        self.visited_url = None
        self.ticker = ticker
        self.stop_year = stop_year
        self.all_data = []

        if not os.path.exists('saved_data/'):
            os.makedirs('saved_data/')

        # if os.path.exists(f'saved_data/{self.ticker}_state.pkl'):
        #     with open(f'saved_data/{self.ticker}_state.pkl', 'rb') as f:
        #         state = pickle.load(f)
        #         self.driver.get(state['url'])
        #         self.visited_url = state['url']
        #         self.all_data = state['data']

        # Close pop ups if any
        self.resolve_popup()

    def get_news_page_(self):
        inp_field = self.driver.find_elements_by_xpath("""//input[contains(@class, 'searchText')]""")[0]
        inp_field.send_keys(self.ticker)
        inp_field.send_keys(Keys.RETURN)
        time.sleep(5)
        links = self.driver.find_elements_by_xpath("""//span""")
        for l in links:
            if l.text == self.ticker:
                l.click()
                break

        news_links = self.driver.find_elements_by_xpath("""//a[contains(@href, 'news')]""")
        news_links = [l for l in news_links if len(l.text) > 0]
        news_links[2].click()
        self.buffer_url = self.driver.current_url

    def save_data_temp_(self):
        state = {
            'url': self.visited_url,
            'data': self.all_data
        }
        with open(f'saved_data/{self.ticker}_state.pkl', 'wb') as f:
            pickle.dump(state, f)

    def resolve_popup(self):
        try:
            popup = self.driver.find_elements_by_xpath("""//i[contains(@class, 'popupCloseIcon)]""")[0]
            popup.click()
            time.sleep(3)
        except Exception as e:
            pass

    def scrape_titles_(self):
        done = False
        count = 2
        self.buffer_url = self.driver.current_url

        while not done:
            try:
                self.resolve_popup()
                titles = self.driver.find_elements_by_xpath("""//a[contains(@class, 'title')]""")
                titles = [t for t in titles if len(t.text) > 0]
                timestamps = self.driver.find_elements_by_xpath("""//span[contains(@class, 'date')]""")
                tstamps = [t for t in timestamps if len(t.text) > 0]

                for i, t in enumerate(titles[:-3]):
                    self.all_data.append({
                        'link': t.get_attribute('href'),
                        'title': t.text,
                        'time': tstamps[i].text
                    })

                for t in tstamps:
                    if str(self.stop_year) in t.text:
                        done = True
                        break
                keys = list(set([d['title'] for d in self.all_data]))
                current_on = tstamps[0].text
                print(f"[{self.ticker}] - [Reached] {current_on} - [Total clips] {len(keys)}")
                self.save_data_temp_()

                self.driver.get(self.buffer_url + f'/{count}')
                count += 1
                time.sleep(3)

            except Exception as e:
                print(e)
                self.save_data_temp_()
                done = True

    def scrape_news_(self):
        print("\n[INFO] Beginning news scrape!\n")
        self.driver.get(self.buffer_url)
        for i in range(len(self.all_data)):
            self.driver.get(self.all_data[i]['link'])
            if ('seekingalpha' in self.driver.current_url):
                continue

            paras = self.driver.find_elements_by_tag_name('p')
            paras = [p for p in paras if len(p.text) > 0]
            text = [p.text for p in paras]
            self.all_data[i]['text'] = unidecode(' '.join(text))
            
            print(f"{i+1} {self.all_data[i]['time']} - {self.all_data[i]['title']}")
            self.save_data_temp_()

    def save_data_(self):
        with open(f"saved_data/{self.ticker}.pkl", "wb") as f:
            pickle.dump(self.all_data, f)

        with open(f"saved_data/{self.ticker}.json", "w") as f:
            json.dump(self.all_data, f, indent=4)

        print("\n[INFO] Successfully saved data!")

    def run(self):
        self.get_news_page_()
        self.scrape_titles_()
        # self.scrape_news_()
        self.save_data_()