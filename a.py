def fn(x, y):
    return 108.0 - (815.0 - 1500.0 / y) / x


x = 4
y = 4.25
while True:
    z = y
    y = fn(y, x)
    x = z
    print(y)
    if x == 100:
        break
