import csv

# Define the header
header = ['caption']

# Define the data
data = [
    ['Hill Country Castle by R Del Angel'],
    ['Captain Marvel Exclusive Ccxp Poster Released Online By Marvel'],
    ['the colors of the mountain fall color colorado highway 145 in the san juan mountains'],
    ['85th Annual Academy Awards – Arrivals'],
    ['Portrait of Tiger in black and white by Lukas Holas'],
    ['Rosie Huntington-Whiteley short hair (2015 Vanity Fair Oscar Party) (Venturelli, photographer for Getty Images)'],
    ['VAN GOGH CAFE TERASSE copy.jpg'],
    ['US presidential election certified results'],
    ['Golden Globes best fashion on the red carpet , CNN Style'],
    ['Sony Boss Confirms Bloodborne Expansion is Coming'],
    ['New Orleans House Galaxy Case'],
    ['Adidas Equipment Support ADV white-white side'],
    ['<i>The Long Dark</i> Gets First Trailer, Steam Early Access'],
    ['A painting with letter M written on it Canvas Wall Art Print']
]

# Specify the file name
filename = '../../data/coco/subset-overfit.csv'

# Writing to csv file
with open(filename, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csvwriter = csv.writer(csvfile)

    # Write the header
    csvwriter.writerow(header)

    # Write the data
    csvwriter.writerows(data)

print(f"{filename} has been created with the header and data.")
