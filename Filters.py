import numpy as np

def Make_Flat_Laplacian_Filter(inner_radius=0, external_radius=1.5):
    
    side = 2 * int(external_radius) + 1
    res = np.zeros((side,side))
    center = np.array(res.shape) // 2
    
    external_radius_sq = external_radius ** 2
    inner_radius_sq = inner_radius ** 2
    
    external_counter = 0
    inner_counter = 0
    
    for x in range(side):
        for y in range(side):
            dist_sq = sum((center - (x,y)) ** 2)
            
            if dist_sq <= external_radius_sq:
                res[x,y] = -1
                external_counter  += 1
                
            if dist_sq <= inner_radius_sq:
                inner_counter  += 1
                
    inner_value = (external_counter - inner_counter) / inner_counter
    
    for x in range(side):
        for y in range(side):
            dist_sq = sum((center - (x,y)) ** 2)
            if dist_sq <= inner_radius_sq:
                res[x,y] = inner_value
                
    return res

def Make_Gausian_Filter(radius=3, sigma=1, round_crop=True):
    x, y = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
    
    if round_crop:
        radius_squared = radius ** 2
        for i in range(2*radius + 1):
            i_sq = (radius - i) ** 2
            for j in range(2*radius + 1):
                if i_sq + (radius - j) ** 2 > radius_squared:
                    g[i][j] = 0
    return g

def Make_Difference_of_Gaussian(radius=5, sigma_1 = 1, sigma_2 = 3, round_crop=True):
    G1 = Make_Gausian_Filter(radius=radius, sigma=sigma_1, round_crop=round_crop)
    G2 = Make_Gausian_Filter(radius=radius, sigma=sigma_2, round_crop=round_crop)
    sum1 = sum(sum(G1))
    sum2 = sum(sum(G2))
    G2 = sum1 / sum2 * G2
    DoG = G1 - G2
    return DoG