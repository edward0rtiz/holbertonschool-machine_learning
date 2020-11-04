#!/usr/bin/env python3
"""update values with pymongo"""


def update_topics(mongo_collection, name, topics):
    """
    update values in a collection
    Args:
        mongo_collection: pymongo collection object.
        name: type str name to be updated.
        topics: type list of topics approached in the school.
    """
    query = {"name": name}
    topic = {"$set": {"topics": topics}}
    mongo_collection.update_many(query, topic)
