#!/usr/bin/env python3
""" Get request from SpaceX"""

import requests

if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    response = requests.get(url).json()
    date = [x['date_unix'] for x in response]
    idx = date.index(min(date))
    launch = response[idx]
    launch_name = launch['name']
    date_l = launch['date_local']
    rocket_id = launch['rocket']
    rocket_url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id)
    rocket_name = requests.get(rocket_url).json()['name']
    lpad_id = launch['launchpad']
    lpad_url = "https://api.spacexdata.com/v4/launchpads/{}".\
        format(lpad_id)
    lpad_req = requests.get(lpad_url).json()
    lpad_name = lpad_req['name']
    lpad_loc = lpad_req['locality']

    upcoming_launch = "{} ({}) {} - {} ({})".format(launch_name, date_l,
                                                    rocket_name, lpad_name,
                                                    lpad_loc)

    print(upcoming_launch)
