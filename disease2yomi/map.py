import folium


def mesh2sq(meshcode, dim):
    meshcode = str(meshcode)
    if dim == 1:
        p = meshcode[0:2]
        u = meshcode[2:4]
        return p, u
    elif dim == 2:
        p = meshcode[0:2]
        u = meshcode[2:4]
        q = meshcode[4]
        v = meshcode[5]
        return (
            p,
            q,
            u,
            v,
        )
    elif dim == 3:
        p = meshcode[0:2]
        u = meshcode[2:4]
        q = meshcode[4]
        v = meshcode[5]
        r = meshcode[6]
        w = meshcode[7]
        return p, q, r, u, v, w
    else:
        p = meshcode[0:2]
        u = meshcode[2:4]
        q = meshcode[4]
        v = meshcode[5]
        r = meshcode[6]
        w = meshcode[7]
        m = meshcode[8]
        n = meshcode[9]
        return p, q, r, u, v, w, m, n


def first(meshcode):
    p, u = mesh2sq(meshcode, 1)
    p = float(p)
    u = float(u)
    lat = p / 1.5 * 3600
    lon = (u + 100) * 3600
    south = lat
    west = lon
    north = lat + 40 * 60
    east = lon + 1 * 3600
    return south / 3600, west / 3600, north / 3600, east / 3600


def second(meshcode):
    p, q, u, v = mesh2sq(meshcode, 2)
    p = float(p)
    u = float(u)
    q = float(q)
    v = float(v)
    lat = p / 1.5 * 3600 + q * 5 * 60
    lon = (u + 100) * 3600 + v * 7.5 * 60
    south = lat
    west = lon
    north = lat + 5 * 60
    east = lon + 7.5 * 60
    return south / 3600, west / 3600, north / 3600, east / 3600


def third(meshcode):
    p, q, r, u, v, w = mesh2sq(meshcode, 3)
    p = float(p)
    u = float(u)
    q = float(q)
    v = float(v)
    r = float(r)
    w = float(w)
    lat = p / 1.5 * 3600 + q * 5 * 60 + r * 30
    lon = (u + 100) * 3600 + v * 7.5 * 60 + w * 45
    south = lat
    west = lon
    north = lat + 30
    east = lon + 45
    return south / 3600, west / 3600, north / 3600, east / 3600


def view(meshcode, region):
    if region == "東京":
        south, west, north, east = second(meshcode)
    elif region == "全国":
        south, west, north, east = first(meshcode)
    start_coords = ((south + north) / 2, (west + east) / 2)
    folium_map = folium.Map(location=start_coords, zoom_start=10)
    sw = (south, west)
    se = (south, east)
    ne = (north, east)
    nw = (north, west)
    folium.Polygon(
        locations=[sw, se, ne, nw],  # 多角形の頂点
        color="red",  # 線の色
        weight=10,  # 線の太さ
        fill=True,  # 塗りつぶす
        fill_opacity=0.5,  # 透明度（1=不透明）
    ).add_to(folium_map)
    return folium_map


def polygon(meshcode, region="全国"):
    if region == "全国" or region == None:
        south, west, north, east = first(meshcode)
    else:
        south, west, north, east = second(meshcode)
    start_coords = [(south + north) / 2, (west + east) / 2]
    sw = [south, west]
    se = [south, east]
    ne = [north, east]
    nw = [north, west]
    return start_coords, sw, se, ne, nw
