#!/usr/bin/env python3
""" Get request from SpaceX"""
import requests


if __name__ == '__main__':

    object = dict()
    url = 'https://api.spacexdata.com/v4/launches'
    launches = requests.get(url).json()
    for launch in launches:
        urls = "https://api.spacexdata.com/v4/rockets/{}"
        rocket_id = launch['rocket']
        rocket_url = urls.format(rocket_id)
        rocket_name = requests.get(rocket_url).json()['name']

        if rocket_name in object.keys():
            object[rocket_name] += 1
        else:
            object[rocket_name] = 1

    keys = sorted(object.items(), key=lambda x: x[0])
    keys = sorted(keys, key=lambda x: x[1], reverse=True)

    for k in keys:
        print("{}: {}".format(k[0], k[1]))
