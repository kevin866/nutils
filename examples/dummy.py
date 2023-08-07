import numpy as np
array = np.array([np.linspace(0, 1, 4 + 1)] * 3)
inner_radius = 5
outer_radius = 10
height = 10
cps = np.array([[inner_radius, 0, 0], [outer_radius, 0, 0],
                [inner_radius, 0, height], [outer_radius, 0, height],
                [inner_radius, -inner_radius, 0],
                [outer_radius, -outer_radius, 0],
                [inner_radius, -inner_radius, height],
                [outer_radius, -outer_radius,
                    height], [0, -inner_radius, 0], [0, -outer_radius, 0],
                [0, -inner_radius, height], [0, -outer_radius, height],
                [-inner_radius, -inner_radius, 0],
                [-outer_radius, -outer_radius, 0],
                [-inner_radius, -inner_radius, height],
                [-outer_radius, -outer_radius, height],
                [-inner_radius, 0, 0], [-outer_radius, 0, 0],
                [-inner_radius, 0, height], [-outer_radius, 0, height],
                [-inner_radius, inner_radius, 0],
                [-outer_radius, outer_radius, 0],
                [-inner_radius, inner_radius, height],
                [-outer_radius, outer_radius, height],
                [0, inner_radius, 0], [0, outer_radius, 0],
                [0, inner_radius, height], [0, outer_radius, height],
                [inner_radius, inner_radius, 0],
                [outer_radius, outer_radius, 0],
                [inner_radius, inner_radius, height],
                [outer_radius, outer_radius, height]])

# print(array.shape)
print(cps.shape)