hi = [20.14, 90.00, 164.25, 237.53, 380.14]
e = [0.8, 0.7, 1.0, 1.2, 0.8]
Hi = [0, 100, 200, 300, 400]

h_list = [8.25, 92.86, 115.0, 245.56]
i_list = [3, 1, 1, 3]
i_dict = {0: 'red', 1:'yellow', 2:'green', 3:'blue', 4:'red'}
for i, h in zip(i_list, h_list):
    if h < 20.14:
        h_prime = h + 360
    else: h_prime = h
    Pi = (h_prime - hi[i]) / e[i]
    Pi1 = (hi[i+1] - h_prime) / e[i+1]
    H = Hi[i] + 100 * Pi / (Pi + Pi1)
    Hci = Pi / (Pi + Pi1) * 100
    Hci1 = Pi1 / (Pi + Pi1) * 100
    print(f"in hue {h} degree, Hue quadrature is {H:.1f} and {i_dict[i]} composition percentage is: {Hci1:.1f}%, and {i_dict[i+1]} composition percentage is : {Hci:.1f}%")

# in hue 8.25 degree, Hue quadrature is 388.0 and blue composition percentage is: 12.0%, and red composition percentage is : 88.0%
# in hue 92.86 degree, Hue quadrature is 105.4 and yellow composition percentage is: 94.6%, and green composition percentage is : 5.4%
# in hue 115.0 degree, Hue quadrature is 142.0 and yellow composition percentage is: 58.0%, and green composition percentage is : 42.0%
# in hue 245.56 degree, Hue quadrature is 303.8 and blue composition percentage is: 96.2%, and red composition percentage is : 3.8%