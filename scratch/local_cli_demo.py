def getUserData(user_id):
    result = fetch_from_db(user_id)
    return result.name
