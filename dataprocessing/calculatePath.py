import json
import os
from typing import List, Optional
import searchUtil
import argparse
from tqdm import tqdm
from mapUtil import (
    CityMap,
    createMap
)
from search import (
    generateRandomPath,
    WaypointsShortestPathProblem
)


def extractPath(startLocation: str, search: searchUtil.SearchAlgorithm) -> List[str]:
    return [startLocation] + search.actions


def saveResultPath(
    path: List[str],
    waypointTags: List[str],
    cityMap: CityMap,
    outPath: Optional[str],
    startLocation,
    endTag
):  
    doneWaypointTags = []
    # print(waypointTags)
    # print(path)
    for location in path:
        for tag in cityMap.tags[location]:
            if tag in waypointTags:
                doneWaypointTags.append(tag.split('=')[1])

    if outPath is not None:
        with open(outPath, "w") as f:
            data = {"start": startLocation,
                    "waypointTags": waypointTags, 
                    "end": endTag.split('=')[1]
                    }
            json.dump(data, f, indent=2)


def calculatePath(minNum, maxNum, saveId, savePath):
    """Given custom WaypointsShortestPathProblem, find the minimun path and prepare visualization."""
    cityMap = createMap("../data/stanford.pbf")
    startLocation, waypointTags, endTag = generateRandomPath(cityMap, minWayPoints=minNum, maxWayPoints=maxNum)
    problem = WaypointsShortestPathProblem(startLocation, tuple(sorted(waypointTags)), str(endTag), cityMap)

    ucs = searchUtil.UniformCostSearch(verbose=0)
    ucs.solve(problem)
    path = extractPath(problem.startLocation, ucs)
    saveResultPath(path=path, waypointTags=problem.waypointTags, cityMap=cityMap, outPath=f"{savePath}/{saveId}.json", startLocation=startLocation, endTag=endTag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add data generation hyper parameters")

    # Add arguments
    parser.add_argument(
        "--minWayPoints", 
        type=int, 
        default=6, 
        help="Minimum number of way points"
    )
    parser.add_argument(
        "--maxWayPoints", 
        type=int, 
        default=8, 
        help="Maximum number of way points"
    )
    parser.add_argument(
        "--numPath", 
        type=int, 
        default=1000, 
        help="Number of data points to generate"
    )
    parser.add_argument(
        "--savePath", 
        type=str, 
        default="out", 
        help="Number of data points to generate"
    )

    args = parser.parse_args()
    os.makedirs(args.savePath, exist_ok=True)
    for num in tqdm(range(args.numPath)):
        calculatePath(args.minWayPoints, args.maxWayPoints, num, args.savePath)
