import os
from stl import mesh

assets_path = 'assets'
for filename in os.listdir(assets_path):
    if filename.endswith('.stl'):
        path = os.path.join(assets_path, filename)
        m = mesh.Mesh.from_file(path)
        m.save(path, mode=mesh.Mode.BINARY)
        print(f"Converted: {filename}")