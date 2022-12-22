def get_at(param, index):
    """
    Function to ease the access to parameters.
    """
    if type(param) == list:
        return param[index]
    else:
        return param