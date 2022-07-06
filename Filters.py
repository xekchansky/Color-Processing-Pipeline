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