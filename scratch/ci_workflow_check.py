def calculateTotal(items):
    total = 0
    for item in items:
        total = total + item.price
    return total
