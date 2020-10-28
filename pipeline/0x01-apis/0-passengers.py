#!/usr/bin/env python3
import requests

def availableShips(passengerCount):
    starships = []
    url = 'https://swapi-api.hbtn.io/api/starships/'
    while url is not None:
        response = requests.get(url, headers={'Accept': 'application/json'}, params={"term":'starships'})
        for ship in response.json()['results']:
            passenger = ship['passengers']
            if passenger.isnumeric() and int(passenger) >= passengerCount:
                starships.append(ship['name'])
        url = response.json()['next']
    return starships
