from datetime import datetime
from metloom.pointdata import SnotelPointData
from conversions import imperial_to_metric

def get_snotel_data(name, site_id, dates):
    snotel_point = SnotelPointData(site_id, name)

    df = snotel_point.get_daily_data(
        dates[0], dates[1],
        [snotel_point.ALLOWED_VARIABLES.SWE, snotel_point.ALLOWED_VARIABLES.SNOWDEPTH, \
            snotel_point.ALLOWED_VARIABLES.TEMPAVG])
    
    df['SWE'] = imperial_to_metric(df['SWE'], 'inch')
    df['SD'] = imperial_to_metric(df['SNOWDEPTH'], 'inch')
    df['temp'] = imperial_to_metric(df['AVG AIR TEMP'], 'fahrenheit')
    df = df.drop(['SWE_units', 'SNOWDEPTH_units','SNOWDEPTH', 'AVG AIR TEMP_units', 'AVG AIR TEMP'], axis = 1)
    df['site_name'] = name

    return df