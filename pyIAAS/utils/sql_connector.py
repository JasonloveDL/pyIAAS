import os
import sqlite3

from .logger import get_logger

sql_create_table = '''CREATE TABLE MODEL(
   ID INT PRIMARY KEY NOT NULL,
   STRUCTURE VARCHAR NOT NULL,
   TRAIN_TIME INT,
   LOSS DOUBLE,
   PREV_INDEX INT,
   IN_POOL INT
   )
'''


def get_or_create_connection(cfg):
    SQL_FILE = cfg.SQL_FILE
    if os.path.exists(SQL_FILE):
        return sqlite3.connect(SQL_FILE)
    conn = sqlite3.connect(SQL_FILE)
    conn.execute(sql_create_table)
    conn.commit()
    return conn


def get_total_model_count(cfg):
    sql = '''
    SELECT COUNT(ID) FROM MODEL
    '''
    conn = get_or_create_connection(cfg)
    r = conn.execute(sql).fetchall()
    conn.close()
    r = r[0][0]
    return r


def insert_new_model_config(cfg, id: int, structure: str, prev_index: int):
    sql = f'''
    INSERT INTO MODEL (ID, STRUCTURE, TRAIN_TIME,PREV_INDEX, IN_POOL) VALUES ({id}, \"{structure}\", 0, {prev_index}, -1);
    '''
    conn = get_or_create_connection(cfg)
    conn.execute(sql)
    conn.commit()
    conn.close()
    # logger.info(sql)


def get_prev_record(cfg, structure: str):
    sql = f'''
    SELECT * FROM MODEL WHERE STRUCTURE = \'{structure}\'
    '''
    conn = get_or_create_connection(cfg)
    id, structure, train_time, loss, prev_index = conn.execute(sql).fetchall()[0]
    return id, structure, train_time, loss, prev_index


def update_new_train_result(cfg, id: int, train_time: int, loss: float):
    sql = f'''
    UPDATE MODEL SET LOSS = {loss}, TRAIN_TIME = {train_time} WHERE ID = {id}
    '''
    try:
        conn = get_or_create_connection(cfg)
        conn.execute(sql)
        conn.commit()
        conn.close()
    except:
        logger = get_logger('SQL', cfg.LOG_FILE)
        logger.fatal(sql)


def update_new_pool_state(cfg, id: int, pool_state: int):
    sql = f'''
    UPDATE MODEL SET IN_POOL = {pool_state} WHERE ID = {id}
    '''
    try:
        conn = get_or_create_connection(cfg)
        conn.execute(sql)
        conn.commit()
        conn.close()
    except:
        logger = get_logger('SQL', cfg.LOG_FILE)
        logger.fatal(sql)
