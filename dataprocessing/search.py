from typing import List, Tuple
from mapUtil import (
    CityMap,
    createMap
)
from searchUtil import SearchProblem, State
import random

def generateRandomPath(cityMap, minWayPoints=2, maxWayPoints=3):
    all_nodes = list(cityMap.tags.keys())
    startLocation = random.choice(all_nodes)
    random_idx = random.sample(range(0, len(cityMap.tags)), random.randint(minWayPoints, maxWayPoints))
    waypointTags = [cityMap.tags[all_nodes[idx]][0] for idx in random_idx]
    endLocation = random.choice(all_nodes)
    endTag = cityMap.tags[endLocation][0]

    return startLocation, waypointTags, endTag

class WaypointsShortestPathProblem(SearchProblem):
    
    def __init__(
        self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: CityMap
    ):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap
        self.waypointTags = tuple(sorted(waypointTags))
        self.visited = {}

    def startState(self) -> State:
        currMemory = [False] * len(self.waypointTags)
        for i in range(len(self.waypointTags)):
            if self.waypointTags[i] in self.cityMap.tags[self.startLocation]:
                currMemory[i] = True
        return State(location=self.startLocation, memory=tuple(currMemory))

    def isEnd(self, state: State) -> bool:
        if self.endTag in self.cityMap.tags[state.location]:
            for temp in state.memory:
                if temp == False:
                    return False
            return True
        else:
            return False

    def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        result = []
        currLocation = state.location
        nextLocationCostDict = self.cityMap.distances[currLocation]
        for nextLocation in nextLocationCostDict.keys():
            cost = nextLocationCostDict[nextLocation]
            nextMemory = list(state.memory)
            for i in range(len(self.waypointTags)):
                if self.waypointTags[i] in self.cityMap.tags[nextLocation]: 
                    nextMemory[i] = True
            nextMemory = tuple(nextMemory)
            nextState = State(location=nextLocation, memory=nextMemory)
            result.append((nextLocation, nextState, cost))
        return result