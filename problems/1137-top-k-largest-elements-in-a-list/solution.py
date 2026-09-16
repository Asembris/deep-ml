def top_three_largest(values):
    # values: list of numbers
    # return the three largest values in descending order
    res=sorted(values,reverse=True)
    return res[:3]