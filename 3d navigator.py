from ursina import *

app = Ursina()

mountains = Terrain(heightmap='definitelymountains',skip=4)


terrain = Entity(model=mountains, scale=(10,2,10),texture='definitelymountainstexture')

EditorCamera()

app.run()