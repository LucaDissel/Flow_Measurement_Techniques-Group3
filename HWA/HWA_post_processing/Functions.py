import re

#  Check whether the it's positive or negative from the 
def file_sorter(file_name):
    match = re.search(r'(?P<sign>neg|pos)(?P<number>\d+)', file_name)
    if match:
        sign = -1 if match.group('sign') == 'neg' else 1
        return sign * int(match.group('number'))
    return 0  # return a default value if no match