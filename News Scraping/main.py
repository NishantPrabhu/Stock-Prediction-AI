
# Main script

import scraper
import argparse


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", '--ticker', required=True, type=str, help='Ticker of the company whose news is to be scraped')
    ap.add_argument('-u', "--url", default="https://www.investing.com/", type=str, help='Site from which news will be scraped')
    ap.add_argument('-s', "--stop-year", default='2015', type=str, help='Year until which news is to be scraped')
    args = vars(ap.parse_args())

    web = scraper.Scraper(args['url'], args['ticker'], args['stop_year'])
    web.run()