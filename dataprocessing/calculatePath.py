import json
import os
from typing import List, Optional
import searchUtil
import argparse
from tqdm import tqdm
from mapUtil import (
    CityMap,
    createMap,
    createGridMap
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
        tag = f"label={location}"
        if tag in waypointTags:
            doneWaypointTags.append(tag)

    if outPath is not None:
        with open(outPath, "w") as f:
            data = {"start": startLocation,
                    "waypointTags": doneWaypointTags, 
                    "end": endTag.split('=')[1]
                    }
            json.dump(data, f, indent=2)

def printPath(
    path: List[str],
    waypointTags: List[str],
    cityMap: CityMap,
    outPath: Optional[str] = "path.json",
):
    doneWaypointTags = set()
    for location in path:
        for tag in cityMap.tags[location]:
            if tag in waypointTags:
                doneWaypointTags.add(tag)
        tagsStr = " ".join(cityMap.tags[location])
        doneTagsStr = " ".join(sorted(doneWaypointTags))
        print(f"Location {location} tags:[{tagsStr}]; done:[{doneTagsStr}]")
    # print(f"Total distance: {getTotalCost(path, cityMap)}")
    # (Optional) Write path to file, for use with `visualize.py`
    if outPath is not None:
        with open(outPath, "w") as f:
            data = {"waypointTags": waypointTags, "path": path}
            json.dump(data, f, indent=2)


def calculatePath(minNum, maxNum, saveId, savePath):
    """Given custom WaypointsShortestPathProblem, find the minimun path and prepare visualization."""
    # cityMap = createMap("../data/stanford.pbf")
    cityMap = createGridMap(10, 10)
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
        default=6, 
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
        default="outSmall", 
        help="Number of data points to generate"
    )

    args = parser.parse_args()
    os.makedirs(args.savePath, exist_ok=True)
    for num in tqdm(range(args.numPath)):
        calculatePath(args.minWayPoints, args.maxWayPoints, num, args.savePath)
