import heapq
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

########################################################################################
# Abstract Interfaces for State, Search Problems, and Search Algorithms.

@dataclass(frozen=True, order=True)
class State:
    location: str
    memory: Optional[Hashable] = None


class SearchProblem:
    # Return the start state.
    def startState(self) -> State:
        raise NotImplementedError("Override me")

    # Return whether `state` is an end state or not.
    def isEnd(self, state: State) -> bool:
        raise NotImplementedError("Override me")

    # the various edges coming out of `state`. Note: it is valid for action 
    # to be equivalent to location of a successor state.
    def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        raise NotImplementedError("Override me")


class SearchAlgorithm:
    def __init__(self):
        self.actions: List[str] = None
        self.pathCost: float = None
        self.numStatesExplored: int = 0
        self.pastCosts: Dict[str, float] = {}

    def solve(self, problem: SearchProblem) -> None:
        raise NotImplementedError("Override me")

########################################################################################
# Uniform Cost Search (Dijkstra's algorithm)

class UniformCostSearch(SearchAlgorithm):
    def __init__(self, verbose: int = 0):
        super().__init__()
        self.verbose = verbose

    def solve(self, problem: SearchProblem) -> None:
        self.actions: List[str] = []
        self.pathCost: float = None
        self.numStatesExplored: int = 0
        self.pastCosts: Dict[str, float] = {}

        # Initialize data structures
        frontier = PriorityQueue()  # Explored states are maintained by the frontier.
        backpointers = {}           # Map state -> previous state.

        # Add the start state
        startState = problem.startState()
        frontier.update(startState, 0.0)

        while True:
            # Remove the state from the queue with the lowest pastCost (priority).
            state, pastCost = frontier.removeMin()
            if state is None and pastCost is None:
                if self.verbose >= 1:
                    print("Searched the entire search space!")
                return

            # Update tracking variables
            self.pastCosts[state.location] = pastCost
            self.numStatesExplored += 1
            if self.verbose >= 2:
                print(f"Exploring {state} with pastCost {pastCost}")

            # Check if we've reached an end state; if so, extract solution.
            if problem.isEnd(state):
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.pathCost = pastCost
                if self.verbose >= 1:
                    print(f"numStatesExplored = {self.numStatesExplored}")
                    print(f"pathCost = {self.pathCost}")
                    print(f"actions = {self.actions}")
                return

            # Expand from `state`, updating the frontier with each `newState`
            for action, newState, cost in problem.actionSuccessorsAndCosts(state):
                if self.verbose >= 3:
                    print(f"\t{state} => {newState} (Cost: {pastCost} + {cost})")

                if frontier.update(newState, pastCost + cost):
                    # We found better way to go to `newState` --> update backpointer!
                    backpointers[newState] = (action, state)


# Data structure for supporting uniform cost search.
class PriorityQueue:
    def __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    def update(self, state: State, newPriority: float) -> bool:
        oldPriority = self.priorities.get(state)
        if oldPriority is None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    # Returns (state with minimum priority, priority) or (None, None) if empty.
    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE:
                # Outdated priority, skip
                continue
            self.priorities[state] = self.DONE
            return state, priority

        return None, None
