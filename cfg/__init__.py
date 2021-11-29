# from cfg import pcnet_comap, cia_ssd_comap, cia_ssd_kitti

LABEL_COLORS = {
    'Unlabeled': (0, 0, 0),  # 0 Unlabeled
    'Buildings': (70, 70, 70),  # 1 Buildings
    'Fences': (100, 40, 40),  # 2 Fences
    'Other': (55, 90, 80),  # 3 Other
    'Pedestrians': (220, 20, 60),  # 4 Pedestrians
    'Poles': (153, 153, 153),  # 5 Poles
    'RoadLines': (157, 234, 50),  # 6 RoadLines
    'Roads': (128, 64, 128),  # 7 Roads
    'Sidewalks': (244, 35, 232),  # 8 Sidewalks
    'Vegetation': (107, 142, 35),  # 9 Vegetation
    'Vehicles': (0, 0, 142),  # 10 Vehicles
    'Walls': (102, 102, 156),  # 11 Walls
    'TrafficSign': (220, 220, 0),  # 12 TrafficSign
    'Sky': (70, 130, 180),  # 13 Sky
    'Ground': (81, 0, 81),  # 14 Ground
    'Bridge': (150, 100, 100),  # 15 Bridge
    'Railtrack': (230, 150, 140),  # 16 Railtrack
    'GuardRail': (180, 165, 180),  # 17 GuardRail
    'TrafficLight': (250, 170, 30),  # 18 TrafficLight
    'Static': (110, 190, 160),  # 19 Static
    'Dynamic': (170, 120, 50),  # 20 Dynamic
    'Water': (45, 60, 150),  # 21 Water
    'Terrain': (145, 170, 100)  # 22 Terrain
}