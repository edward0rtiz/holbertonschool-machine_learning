#!/usr/bin/env python3
""" Get requests v2"""
import requests


def sentientPlanets():
    """
    Function to get request from names of planets
    Args:
        passengerCount: number of passangers
    Returns: List of starships
    """
    planets = []
    url = 'https://swapi-api.hbtn.io/api/species/'
    while url is not None:
        response = requests.get(url,
                                headers={'Accept': 'application/json'},
                                params={"term": 'specie'})
        for specie in response.json()['results']:
            if specie['classification'] == 'sentient' or \
                    specie['designation'] == 'sentient':
                if specie['homeworld'] is not None:
                    homeworld = requests.get(specie['homeworld'])
                    planets.append(homeworld.json()['name'])

        url = response.json()['next']
    return planets
