import argparse
import json
from typing import List

import plotly.express as px
import plotly.graph_objects as go

from mapUtil import CityMap, readMap, createMap, createGridMap
from calculatePath import extractPath
from searchUtil import ShortestPathProblem, UniformCostSearch

def getTotalCost(path: List[str], cityMap: CityMap) -> float:
    """
    Return the length of a given path
    """
    cost = 0.0
    for i in range(len(path) - 1):
        cost += cityMap.distances[path[i]][path[i + 1]]
    return cost


def plotMap(cityMap: CityMap, path: List[str], waypointTags: List[str], mapName: str):
    """
    Plot the full map, highlighting the provided path.
    """
    lat, lon = [], []

    # Convert `cityMap.distances` to a list of (source, target) tuples...
    connections = [
        (source, target)
        for source in cityMap.distances
        for target in cityMap.distances[source]
    ]
    for source, target in connections:
        lat.append(cityMap.geoLocations[source].latitude)
        lat.append(cityMap.geoLocations[target].latitude)
        lat.append(None)
        lon.append(cityMap.geoLocations[source].longitude)
        lon.append(cityMap.geoLocations[target].longitude)
        lon.append(None)

    # Plot all states & connections
    fig = px.line_geo(lat=lat, lon=lon)

    # Plot path (represented by connections in `path`)
    if len(path) > 0:
        solutionLat, solutionLon = [], []

        # Get and convert `path` to (source, target) tuples to append to lat, lon lists
        connections = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        for connection in connections:
            source, target = connection
            solutionLat.append(cityMap.geoLocations[source].latitude)
            solutionLat.append(cityMap.geoLocations[target].latitude)
            solutionLat.append(None)
            solutionLon.append(cityMap.geoLocations[source].longitude)
            solutionLon.append(cityMap.geoLocations[target].longitude)
            solutionLon.append(None)

        # Visualize path by adding a trace
        fig.add_trace(
            go.Scattergeo(
                lat=solutionLat,
                lon=solutionLon,
                mode="lines",
                line=dict(width=5, color="blue"),
                name="solution",
            )
        )

        # Plot the points
        for i, location in enumerate(path):
            tags = set(cityMap.tags[location]).intersection(set(waypointTags))
            if i == 0 or i == len(path) - 1 or len(tags) > 0:
                for tag in cityMap.tags[location]:
                    if tag.startswith("landmark="):
                        tags.add(tag)
            if len(tags) == 0:
                continue

            # Add descriptions as annotations for each point
            description = " ".join(sorted(tags))

            # Color the start node green, the end node red, intermediate gray
            if i == 0:
                color = "red"
            elif i == len(path) - 1:
                color = "green"
            else:
                color = "gray"

            waypointLat = [cityMap.geoLocations[location].latitude]
            waypointLon = [cityMap.geoLocations[location].longitude]

            fig.add_trace(
                go.Scattergeo(
                    lat=waypointLat,
                    lon=waypointLon,
                    mode="markers",
                    marker=dict(size=20, color=color),
                    name=description,
                )
            )

    # Final scaling, centering, and figure title
    midIdx = len(lat) // 2
    fig.update_layout(title=mapName, title_x=0.5)
    fig.update_layout(
        geo=dict(projection_scale=20000, center=dict(lat=lat[midIdx], lon=lon[midIdx]))
    )
    fig.show()


def plotEmptyMap(cityMap: CityMap, mapName: str):
    """
    Plot the full map, highlighting the provided path.
    """
    lat, lon = [], []

    # Convert `cityMap.distances` to a list of (source, target) tuples...
    connections = [
        (source, target)
        for source in cityMap.distances
        for target in cityMap.distances[source]
    ]
    for source, target in connections:
        lat.append(cityMap.geoLocations[source].latitude)
        lat.append(cityMap.geoLocations[target].latitude)
        lat.append(None)
        lon.append(cityMap.geoLocations[source].longitude)
        lon.append(cityMap.geoLocations[target].longitude)
        lon.append(None)

    # Plot all states & connections
    fig = px.line_geo(lat=lat, lon=lon)

    # Final scaling, centering, and figure title
    midIdx = len(lat) // 2
    fig.update_layout(title=mapName, title_x=0.5)
    fig.update_layout(
        geo=dict(projection_scale=20000, center=dict(lat=lat[midIdx], lon=lon[midIdx]))
    )
    fig.show()

def getCompletePath(start, wayPoints, end, stanfordCalMap):

    resultPath = []
    usc = UniformCostSearch(verbose=0)
    currStart = start
    for wayPoint in wayPoints:
        currEnd = wayPoint
        problem = ShortestPathProblem(startLocation=currStart, endTag=currEnd, cityMap=stanfordCalMap)
        usc.solve(problem)
        currPath = extractPath(problem.startLocation, usc)
        
        # Here is an example
        currPathCost = getTotalCost(currPath, stanfordCalMap)

        resultPath += currPath[:-1]
        currStart = currEnd.split('=')[1]
    
    # End node
    problem = ShortestPathProblem(startLocation=currStart, endTag=end, cityMap=stanfordCalMap)
    usc.solve(problem)
    currPath = extractPath(problem.startLocation, usc)
    resultPath += currPath
    return resultPath
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map-file", type=str, default="../data/stanford.pbf", help="Map (.pbf)"
    )
    parser.add_argument(
        "--path-file",
        type=str,
        default="out/10.json",
        help="Path to visualize (.json), path should correspond to some map file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="plotempty",
        help="Select visualization mode",
    )
    args = parser.parse_args()


    stanfordMapName = args.map_file.split("/")[-1].split("_")[0]
    stanfordCityMap = readMap(args.map_file)
    stanfordCalMap = createMap(args.map_file)
    # stanfordCityMap = createGridMap(10, 10)

    if args.mode == "plot_empty":
        plotEmptyMap(
            cityMap=stanfordCityMap,
            mapName=stanfordMapName,
        )
    
    else:

        if args.path_file != 'None':
            with open(args.path_file) as f:
                data = json.load(f)
                parsedWaypointTags = data["waypointTags"]
                parsedPath = getCompletePath(data['start'], data['waypointTags'], f"label={data['end']}", stanfordCalMap)
        else:
            parsedPath = []
            parsedWaypointTags = []

        print(parsedWaypointTags)

        plotMap(
            cityMap=stanfordCityMap,
            path=parsedPath,
            waypointTags=parsedWaypointTags,
            mapName=stanfordMapName,
        )
