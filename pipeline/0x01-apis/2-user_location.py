#!/usr/bin/env python3
""" Get requests location from Github API"""
import sys
import requests
import time


if __name__ == '__main__':
    url = sys.argv[1]
    headers = {'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print(response.json()['location'])
    elif response.status_code == 404:
        print('Not found')
    elif response.status_code == 403:
        limit = response.headers['X-Ratelimit-Reset']
        start = time.time()
        elapsed = (limit - start) / 60
        print('Reset in {}'.format(elapsed))