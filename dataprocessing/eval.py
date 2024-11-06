from mapUtil import (
    getTotalCost
)
from scipy.stats import kendalltau

def get_correlation(path1, path2):
    """
    Evaluate the correlation between two ordered lists using Kendall Tau correlation.
    
    Args:
        path1 (list): The first ordered list of labels.
        path2 (list): The second ordered list of labels, same length and elements as path1.
    
    Returns:
        float: Kendall Tau correlation coefficient between path1 and path2.
    """
    # Ensure path1 and path2 are of the same length
    if len(path1) != len(path2):
        raise ValueError("The two lists must have the same length.")
    
    # Calculate Kendall Tau correlation
    correlation, _ = kendalltau(path1, path2)
    return correlation

def get_distance(path, cityMap):
    """
    Calculate the total distance or cost of a given path through a map of cities.

    Args:
        path (list): A list of city identifiers representing the order in which cities are visited.
        cityMap (dict): A dictionary representing the map of cities, where keys are city pairs or
                        identifiers and values are the distances or costs between those cities.
    
    Returns:
        float: The total distance or cost associated with traversing the given path.

    This function uses the helper function getTotalCost to compute the sum of distances or costs
    between consecutive cities in the path based on the cityMap.
    """
    return getTotalCost(path, cityMap)
