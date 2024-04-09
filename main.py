# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import geopy.distance
from gamspy import *
import sys

# Data Preprocessing
df = pd.read_csv("https://raw.githubusercontent.com/dudegladiator/Logistic-optimization-for-industrial-expansion/main/data/Preprocessed_Russia_Cities_Database.csv")

total = 150  # Number of cities to consider
rang = 400  # Range of distance
var_cost = 12  # Cost of each km
fix_cost = 75  # Base price
capacity = 5  # Capacity of the truck

cit = df['city'][:total].tolist()
x_cor = df['lat'][:total].tolist()
y_cor = df['lng'][:total].tolist()
population = df['population'][:total].tolist()
cost = df['cost'][:total].tolist()

usable = [1 if i < 120 else 0 for i in range(total)]

# Calculate distances between cities
dist = [[0 for i in range(total)] for j in range(total)]
for i in range(total):
    for j in range(total):
        dist[i][j] = geopy.distance.geodesic((x_cor[i], y_cor[i]), (x_cor[j], y_cor[j])).km

# Visualization
fig = plt.figure()
fig.set_size_inches(12, 8)
plot = fig.add_subplot()
plot.plot(x_cor[0], y_cor[0], c='r', marker='s')  # Depot

usableCX = {i: x_cor[i] for i in range(total) if usable[i] == 1}
usableCY = {i: y_cor[i] for i in range(total) if usable[i] == 1}
nonUsableCX = {i: x_cor[i] for i in range(total) if usable[i] == 0}
nonUsableCY = {i: y_cor[i] for i in range(total) if usable[i] == 0}

plot.scatter(list(usableCX.values()), list(usableCY.values()), c='g')
plot.scatter(list(nonUsableCX.values()), list(nonUsableCY.values()), c='b')

# Assignment Optimization
cont = Container()

city = Set(container=cont, name='city', records=cit)

x = Variable(container=cont, name='x', domain=city, type='binary', description='exist or not')

for i, j in enumerate(cit):
    if usable[i] == 0:
        Equation(container=cont, name=f'c{i}')[...] = x[j] == 0
    Equation(container=cont, name=f'co{i}')[...] = sum(x[q] for p, q in enumerate(cit) if dist[i][p] <= rang) >= 1

Equation(container=cont, name='alwys')[...] = x[cit[0]] == 1

obj = Variable(container=cont, name='obj')
obj = sum(x[j] * cost[i] for i, j in enumerate(cit))

optimal = Model(container=cont, name='optimal', equations=cont.getEquations(), problem=Problem.MIP, sense=Sense.MIN, objective=obj)

optimal.solve(output=sys.stdout, options=Options(solver='CPLEX', iteration_limit=10000, job_time_limit=1000, time_limit=1000))

# Extracting coordinates of stores
buildingCosts = optimal.objective_value * 1000
storesCoords = {i: (x_cor[i], y_cor[i]) for i, j in enumerate(x.records['level']) if j == 1}

# Clark Wright Savings Algorithm
def calculate_distance_savings():
    stores_indices = storesCoords.keys()
    distance_savings = [(dist[0][i] + dist[0][j] - dist[i][j], i + 1, j + 1) for i in stores_indices for j in stores_indices if i != j and i > j]
    distance_savings.sort(reverse=True)
    return distance_savings

distance_savings = calculate_distance_savings()
routes = [[1, i + 1, 1] for i in storesCoords.keys() if i > 0]

def find_routes_passing_through(location1, location2):
    route1 = None
    route2 = None
    for route in routes:
        if route1 is not None and route2 is not None:
            break
        if route[1] == location1:
            route.reverse()
            route1 = route
            continue
        elif route[-2] == location1:
            route1 = route
            continue
        if route[1] == location2:
            route2 = route
            continue
        elif route[-2] == location2:
            route.reverse()
            route2 = route
            continue
    return route1, route2

def merge_routes(route1, intermediate_locations, route2):
    route1_copy = list(route1)
    route2_copy = list(route2)
    del route1_copy[-2:]
    del route2_copy[:2]
    return route1_copy + intermediate_locations + route2_copy

def clarke_wright():
    while len(distance_savings) > 0:
        current_saving = distance_savings.pop(0)
        r1, r2 = find_routes_passing_through(current_saving[1], current_saving[2])
        if r1 is not None and r2 is not None:
            new_route = merge_routes(r1, [current_saving[1], current_saving[2]], r2)
            if len(new_route) - 2 <= capacity:  # excluding the store at location 1 (depot)
                routes.remove(r1)
                routes.remove(r2)
                routes.insert(0, new_route)

clarke_wright()

# Solution
colors = iter(cm.rainbow(np.linspace(0, 1, len(routes))))
for route in routes:
    color = next(colors)
    for i, j in zip(route, route[1:]):
        x_val = [storesCoords[i - 1][0], storesCoords[j - 1][0]]
        y_val = [storesCoords[i - 1][1], storesCoords[j - 1][1]]
        plt.plot(x_val, y_val, lw=1, color=color)

totalKm = sum(dist[i - 1][j - 1] for route in routes for i, j in zip(route, route[1:]))
drivingCosts = totalKm * var_cost
drivingFees = len(routes) * fix_cost
totalRefCosts = drivingCosts + drivingFees + buildingCosts

print("Driving Costs: ", drivingCosts + drivingFees)
print("Total Costs:", totalRefCosts)


