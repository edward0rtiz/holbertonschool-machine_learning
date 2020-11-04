#!/usr/bin/env python3
""" insert doc"""


def insert_school(mongo_collection, **kwargs):
    """
    insert a document in a collection based on kwargs
    Args:
        mongo_collection: pymongo collection object
        **kwargs: entry value
    Returns: new id
    """
    return mongo_collection.insert_one(kwargs).inserted_id
