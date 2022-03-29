# ETL Pipeline - MLB

## Motivation

This script grabs the necessary MLB data from the SportsData API and places it in the appropriate table in an Amazon RDS \
Postgres database. Tables were created previously from the data dictionary kindly provided by SportsData. The python file \
described in this markdown file can also be found in this directory. A similar script for handling PGA data can also be found\
there. 

## Overview
* Imports and Definitions
* Grab Data Method
* Read/Write Method
* Closing Thoughts

## Imports and Definitions

This section imports the necessary libraries and defines some of the information we will be reusing in the script, mainly \
the API endpoints I will be grabbing data from and the parameters necessary for those endpoints.

I use SQLAlchemy to grab the tables in my MLB database, and then execute the grab_data method to begin the ETL process.

```python
import datetime as dt

import pandas as pd
import requests
from sqlalchemy import create_engine, MetaData
from sqlalchemy.dialects.postgresql import insert

api = 'https://api.sportsdata.io/v3/mlb/{}/json/{}'

schema = dict(dfsslate=('daily', 'projections', ['DfsSlatesByDate']),
              dfsslategame=('daily', 'projections', ['DfsSlatesByDate']),
              dfsslateplayer=('daily', 'projections', ['DfsSlatesByDate']), game=('daily', 'scores', ['GamesByDate']),
              gameinfo=('daily', 'odds', ['GameOddsByDate']), gameodd=('daily', 'odds', ['GameOddsByDate']),
              inning=('daily', 'stats', ['BoxScores']), player=('static', 'scores', ['Players', 'FreeAgents']),
              playergame=('daily', 'stats', ['PlayerGameStatsByDate']),
              playergameprojection=('daily', 'projections', ['PlayerGameProjectionStatsByDate']),
              playerprop=('daily', 'odds', ['PlayerPropsByDate']),
              playerseason=('season', 'stats', ['PlayerSeasonStats']),
              playerseasonprojection=('season', 'projections', ['PlayerSeasonProjectionStats']),
              season=('static', 'scores', ['CurrentSeason']), stadium=('static', 'scores', ['Stadiums']),
              standing=('season', 'scores', ['Standings']), team=('static', 'scores', ['AllTeams']),
              teamgame=('daily', 'scores', ['TeamGameStatsByDate']),
              teamseason=('season', 'scores', ['TeamSeasonStats']))

keys = dict(stats={'key': ''},
            projections={'key': ''},
            odds={'key': ''},
            scores={'key': ''})

# tables
nested = ['dfsslate', 'dfsslategame', 'dfsslateplayer', 'game', 'gameinfo', 'gameodd', 'inning']

stats = ['playergame', 'playergameprojection', 'teamgame', 'game', 'inning']
dfs = ['dfsslate', 'dfsslategame', 'dfsslateplayer']
vegas = ['gameodd', 'gameinfo', 'playerprop']
season = ['playerseason', 'playerseasonprojection', 'standing', 'teamseason']
static = ['player', 'team', 'stadium']

daily = stats + dfs + vegas

engine = create_engine('postgresql://user:password@mlb-postgresql.cxmbk6ooy1lu.us-east-1.rds.amazonaws.com/mlb',
                       use_batch_mode=True)

update = True
tables = daily
first = '2020-08-12'
last = '2020-09-15'
season = 2020

# startup sqlalchemy
metadata = MetaData(bind=engine)
metadata.reflect(bind=engine)

now = dt.datetime.now().time()
print('Connecting to mlb PostgreSQL database...             {}:{}:{:.0f}'.format(now.hour, now.minute, now.second))

# grab data from api and insert into tables
if type(tables) == list:
    [grab_data(table) for table in metadata.sorted_tables if table.name in tables]
elif type(tables) == str:
    [grab_data(table) for table in metadata.sorted_tables if table.name == tables]

print('mlb data successfully inserted into mlb PostgreSQL database!')
```

## Grab Data method

Below details the grab_data method, which houses the logic for grabbing data alongside the nested method rw for doing the \
leg work of grabbing the appropriate data.

Here we are used for loops to iterate through the desired tournaments and endpoints to grab data. Text formatting plays \
a large role here in defining the specific endpoints to pass into the rw method. See the definitions above to understand \
how endpoint urls are constructed. 

```python
def grab_data(table):
    def rw(table, url):...

    name = table.name
    info = schema[name]
    timeframe = info[0]
    keyname = info[1]
    endpoints = info[2]
    key = keys[keyname]
    
    print('Grabbing data for {} table...'.format(name))
    
    for endpoint in endpoints:
        if timeframe == 'static':
            url = api.format(keyname, endpoint)
            rw(table, url, key)
        elif timeframe == 'season':
            endpt = endpoint + '/' + str(season)
            url = api.format(keyname, endpt)
            rw(table, url, key)
        elif timeframe == 'daily':
            for date in pd.date_range(first, last):
                endpt = endpoint + '/' + str(date.date())
                url = api.format(keyname, endpt)
                rw(table, url, key)
```

## Read/Write method

Here we have the main method for grabbing the data and placing it in the appropriate table. For this API, responses are \
formatted in JSON format. Multiple conditionals are put in place to handle the variety of data processing needs for the \
different tables found in my PGA database.

We also have some except handling at the bottom to handle data that already exists in the database but needs to be updated.

```python
def rw(table, url, key):
    name = table.name
    conn = engine.connect()
    r = requests.get(url, key)  # get data from api
    json_list = r.json()
    if json_list:
        if name == 'season':
            json_list = {k.lower(): v for k, v in json_list.items()}
        else:
            type_check = sum([(type(item) == dict) * 1 for item in json_list])
            if len(json_list) == type_check:
                if name == 'dfsslate':
                    for item in json_list:
                        del (item['DfsSlateGames'])
                        del (item['DfsSlatePlayers'])
                if name == 'dfsslategame':
                    l = []
                    for item in json_list:
                        dfsslategames = item['DfsSlateGames']
                        for games in dfsslategames:
                            del (games['Game'])
                            l.append(games)
                    json_list = l
                if name == 'dfsslateplayer':
                    l = []
                    for item in json_list:
                        dfsslateplayers = item['DfsSlatePlayers']
                        for player in dfsslateplayers:
                            l.append(player)
                    json_list = l
                if name == 'game':
                    for item in json_list:
                        del (item['Innings'])
                if name == 'gameinfo':
                    for item in json_list:
                        del (item['PregameOdds'])
                        del (item['LiveOdds'])
                if name == 'gameodd':
                    l = []
                    for item in json_list:
                        pregameodds = item['PregameOdds']
                        for odds in pregameodds:
                            l.append(odds)
                    json_list = l
                if name == 'inning':
                    l = []
                    for item in json_list:
                        innings = item['Innings']
                        for inning in innings:
                            l.append(inning)
                    json_list = l
                json_list = [{k.lower(): v for k, v in entry.items()} for entry in json_list]
        if len(json_list) > 0:
            try:
                conn.execute(table.insert(), json_list)
                now = dt.datetime.now().time()
                print(
                    '{} data inserted into {} table         {}:{}:{:.0f}'.format(url, table.name, now.hour,
                                                                                 now.minute,
                                                                                 now.second))
            except:
                if update:
                    now = dt.datetime.now().time()
                    print('{} data exists, updating entries...        {}:{}:{:.0f}'.format(url, now.hour,
                                                                                           now.minute,
                                                                                           now.second))
                    for item in json_list:
                        insert_stmt = insert(table).values(item)
                        do_update_stmt = insert_stmt.on_conflict_do_update(constraint=table.primary_key,
                                                                           set_=item)
                        conn.execute(do_update_stmt)
                        now = dt.datetime.now().time()
                        print('Entry updated        {}:{}:{:.0f}'.format(now.hour, now.minute, now.second))
                else:
                    now = dt.datetime.now().time()
                    print('{} data exists        {}:{}:{:.0f}'.format(url, now.hour, now.minute, now.second))
                pass
        conn.close()
```

## Closing Thoughts

I utilized nested methods in this script, so I changed up a few things in this document to make it more readable. If you \
are interested to see how the script is structured in practice please take a look at the script of the same name in the \
scripts folder.

Areas of improvement:
* Better comment strings
* Better handling of sensitive login information, embed in path variables instead