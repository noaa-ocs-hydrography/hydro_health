
import ssl
import time
from xmlrpc.client import ServerProxy


def send_record(record, hostname='hibase.noaa.gov', table='bluetopo_test', repeat=1, sleep=0.1):
    """
    Insert a new row into a NOAA database

    :param dict[] record: Dictionary containing column:value options
    :param str hostname: Host name of the database
    :param str table: Name of table in database
    :param int repeat: Max tries to repeat the insert request
    :param float sleep: A pause between consecutive requests
    :return None|int: Insert result or None if failure
    """
    
    rtn = None
    count = 0
    while (rtn is None and count < repeat):
        count += 1
        try:
            server = ServerProxy(f'https://{hostname}/pydro/default/call/xmlrpc', context=ssl._create_unverified_context())
            if table == 'app_usage':
                rtn = server.insert_app_usage_rpc(record)
            elif table == 'bluetopo':
                rtn = server.insert_bluetopo_rpc(record)
            elif table == 'bluetopo_test':
                rtn = server.insert_bluetopo_test_rpc(record)
            else:
                rtn = None
        except:
            rtn = None
        if rtn is None and count < repeat:
            time.sleep(sleep)
    return rtn


def sql_query(sql, hostname='hibase.noaa.gov'):
    """
    Query table with raw SQL

    :param str sql: SQL query string
    :param str hostname: Host name of the database
    :return None|int: Insert result or None if failure    
    """

    try:
        server = ServerProxy(f'https://{hostname}/pydro/default/call/xmlrpc', context=ssl._create_unverified_context())
        return server.sql_rpc(sql)
    except:
        return None
