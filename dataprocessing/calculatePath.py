import json
from typing import List, Optional
import searchUtil
from mapUtil import (
    CityMap,
    createStanfordMap,
    getTotalCost,
    createMap
)
from search import (
    generateRandomPath,
    WaypointsShortestPathProblem
)


def extractPath(startLocation: str, search: searchUtil.SearchAlgorithm) -> List[str]:
    return [startLocation] + search.actions


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
        # tagsStr = " ".join(cityMap.tags[location])
        # doneTagsStr = " ".join(sorted(doneWaypointTags))
        # print(f"Location {location} tags:[{tagsStr}]; done:[{doneTagsStr}]")
    # print(f"Total distance: {getTotalCost(path, cityMap)}")
    if outPath is not None:
        with open(outPath, "w") as f:
            data = {"waypointTags": waypointTags, "path": path}
            json.dump(data, f, indent=2)


# Load Map
stanfordMap = createStanfordMap()


def calculatePath():
    """Given custom WaypointsShortestPathProblem, find the minimun path and prepare visualization."""
    cityMap = createMap("stanford.pbf")
    startLocation, waypointTags, endTag = generateRandomPath(cityMap, minWayPoints=2, maxWayPoints=3)
    problem = WaypointsShortestPathProblem(startLocation, tuple(sorted(waypointTags)), str(endTag), cityMap)

    ucs = searchUtil.UniformCostSearch(verbose=0)
    ucs.solve(problem)
    path = extractPath(problem.startLocation, ucs)
    printPath(path=path, waypointTags=problem.waypointTags, cityMap=stanfordMap)

if __name__ == "__main__":
    calculatePath()
