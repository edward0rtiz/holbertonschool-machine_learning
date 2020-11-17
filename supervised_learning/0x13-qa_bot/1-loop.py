#!/usr/bin/env python3
"""prompt of the loop"""

words = ['bye', 'goodbye', 'quit', 'exit', 'BYE']
while True:
    request = input("Q: ")

    if request in words:
        print('A: Goodbye')
        break
    else:
        print('A: ')
