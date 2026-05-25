def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    # Write code here
    left_idx = 0 
    right_idx = len(values) - 1
    if left_idx == right_idx:
        return values
    left_not_none, right_not_none = [], []
    for i in range(len(values)):
        if values[i] is not None:
            left_idx = i
        left_not_none.append(left_idx)
    
    for i in range(len(values) - 1, -1, -1):
        if values[i] is not None:
            right_idx = i
        right_not_none.append(right_idx)
    right_not_none = right_not_none[::-1]
    new_values = []
    for j in range(len(values)):
        if values[j] is None:
            l = left_not_none[j]
            r = right_not_none[j]
            mul = ((j - l)/(r - l))
            new_values.append(values[l] + mul * (values[r] - values[l]))
        else:
            new_values.append(values[j])
    return new_values        
