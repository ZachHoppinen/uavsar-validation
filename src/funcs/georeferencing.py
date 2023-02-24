from geopy.distance import geodesic
def resolution_to_meters(resolution_deg, ul_lat):
    """
    Calculation resolution in meters from a resolution in degrees/pixel
    """

    from geopy.distance import geodesic

    resolution_deg = float(resolution_deg)
    ul_lat = float(ul_lat)

    origin = (ul_lat, 0)  # (latitude, longitude) don't confuse
    dist = (ul_lat + 1, 0)

    ms_per_deg = geodesic(origin, dist).meters

    resolution_m = resolution_deg * ms_per_deg

    return resolution_m