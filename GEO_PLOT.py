"""
functions used for get and plot geo data
"""

__version__ = f'Version 1.0  \nTime-stamp: <2019-02-21>'
__author__ = "ChaoTANG@univ-reunion.fr"

import pandas as pd


def query_data(mysql_query: str):
    """
    select data from DataBase
    :return: DataFrame
    """

    from sqlalchemy import create_engine
    import pymysql
    pymysql.install_as_MySQLdb()

    db_connection_str = 'mysql+pymysql://pepapig:123456@localhost/SWIO'
    db_connection = create_engine(db_connection_str)

    df: pd.DataFrame = pd.read_sql(sql=mysql_query, con=db_connection)

    df = df.set_index('DateTime')
    return df


