#!/usr/bin/env python3
"""School by topic"""


def schools_by_topic(mongo_collection, topic):
    """
    query specific topic by school
    Args:
        mongo_collection: pymongo collection object
        topic: will be topic searched
    Returns: query
    """
    school_chosen = []
    results = mongo_collection.find({"topics": {"$all": [topic]}})
    for result in results:
        school_chosen.append(result)
    return school_chosen
