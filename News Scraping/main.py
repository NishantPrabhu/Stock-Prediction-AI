
# Main script

import scraper
import argparse
import multiprocessing as mp


def scrape_process(pid, queue, args):
    web = scraper.Scraper(args['url'], args['ticker'], args['stop_year'], args['num_workers'], queue, pid)
    web.run()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", '--ticker', required=True, type=str, help='Ticker of the company whose news is to be scraped')
    ap.add_argument('-u', "--url", default="https://www.investing.com/", type=str, help='Site from which news will be scraped')
    ap.add_argument('-s', "--stop-year", default='2015', type=str, help='Year until which news is to be scraped')
    ap.add_argument('-n', "--num-workers", default=1, type=int, help='Number of cores to use parallely')
    args = vars(ap.parse_args())

    # web = scraper.Scraper(args['url'], args['ticker'], args['stop_year'], args['num_workers'])
    # web.run()

    queue = mp.Queue()
    workers = []

    for i in range(args['num_workers']):
        worker_args = (i, queue, args)
        workers.append(mp.Process(target=scrape_process, args=worker_args))
        workers[-1].start()

    for i in range(args['num_workers']):
        _ = queue.get()