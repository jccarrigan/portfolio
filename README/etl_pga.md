# ETL Pipeline - PGA

## Motivation 

This script grabs the necessary PGA data from the SportsData API and places it in the appropriate table in an Amazon RDS \
Postgres database. Tables were created previously from the data dictionary kindly provided by SportsData. The python file \
described in this markdown file can also be found in this directory. A similar script for handling MLB data can also be found\
there.

## Overview
* Imports and Definitions
* Grab Data Method
* Read/Write Method
* Closing Thoughts

## Imports and Definitions

This section imports the necessary libraries and defines some of the information we will be reusing in the script, mainly \
the API endpoints I will be grabbing data from and the parameters necessary for those endpoints.

I use SQLAlchemy to grab the tables in my PGA database, and then execute the grab_data method to begin the ETL process.

```python
import datetime as dt

import requests
from sqlalchemy import create_engine, MetaData
from sqlalchemy.dialects.postgresql import insert

api = 'https://api.sportsdata.io/golf/v2/json/{}'

key = '?key='
endpoints = dict(dfsslate='dfsslatesbytournament/{tournamentid}', dfsslateplayer='dfsslatesbytournament/{tournamentid}',
                 dfsslatetournament='dfsslatesbytournament/{tournamentid}', injury='injuries', player='players',
                 playerhole='leaderboard/{tournamentid}',
                 playerround='leaderboard/{tournamentid}', playerseason='playerseasonstats/{season}',
                 playertournament='leaderboard/{tournamentid}',
                 playertournamentprojection='playertournamentprojectionstats/{tournamentid}',
                 playertournamentodds=['tournamentoddslinemovement/{tournamentid}',
                                       'inplaytournamentoddslinemovement/{tournamentid}'], round='tournaments',
                 tournament='tournaments')
static = ['injury', 'player', 'round', 'tournament']
season = ['playerseason']
tournament = ['dfsslate', 'dfsslateplayer', 'dfsslatetournament', 'playerhole', 'playerround', 'playertournament',
              'playertournamentprojection', 'playertournamentodds']
db = ['dfsslate', 'dfsslateplayer', 'dfsslatetournament', 'injury', 'player', 'playerhole', 'playerround',
      'playerseason', 'playertournament', 'playertournamentodds', 'playertournamentprojection', 'round', 'tournament']
engine = create_engine('postgresql://user:password@pga-postgresql.cxmbk6ooy1lu.us-east-1.rds.amazonaws.com/pga')

update = True
tables = tournament
tournament_start = 392
tournament_stop = 392

season = 2020
   
# startup sqlalchemy
metadata = MetaData(bind=engine)
metadata.reflect(bind=engine)

now = dt.datetime.now().time()
print('Connecting to pga PostgreSQL database...             {}:{}:{:.0f}'.format(now.hour, now.minute, now.second))

# grab data from api and insert into tables
if type(tables) == list:
    [grab_data(table) for table in metadata.sorted_tables if table.name in tables]
elif type(tables) == str:
    [grab_data(table) for table in metadata.sorted_tables if table.name == tables]

print('PGA data successfully inserted into pga PostgreSQL database!')

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
    endpoint = endpoints[name]
    print('Grabbing data for {} table...'.format(name))
    if name == 'playertournamentodds':
        api2 = 'https://api.sportsdata.io/v3/golf/odds/json/{}'
        for tournamentid in range(tournament_start, tournament_stop + 1):
            for endpt in endpoint:
                url = api2.format(endpt.format(tournamentid=tournamentid) + key)
                rw(table, url)
    elif '{tournamentid}' in endpoint:
        for tournamentid in range(tournament_start, tournament_stop + 1):
            url = api.format(endpoint.format(tournamentid=tournamentid) + key)
            rw(table, url)
    elif '{season}' in endpoint:
        url = api.format(endpoint.format(season=season) + key)
        rw(table, url)
    else:
        url = api.format(endpoint + key)
        rw(table, url)
```

## Read/Write method

Here we have the main method for grabbing the data and placing it in the appropriate table. For this API, responses are \
formatted in JSON format. Multiple conditionals are put in place to handle the variety of data processing needs for the \
different tables found in my PGA database. 

We also have some except handling at the bottom to handle data that already exists in the database but needs to be updated. 

```python
def rw(table, url):
    name = table.name
    conn = engine.connect()
    try:
        r = requests.get(url)  # get data from api
        json_list = r.json()
        if json_list:
            if name == 'dfsslate':
                for item in json_list:
                    del (item['DfsSlateTournaments'])
                    del (item['DfsSlatePlayers'])
            if name == 'dfsslateplayer':
                l = []
                for item in json_list:
                    dfsslateplayers = item['DfsSlatePlayers']
                    for player in dfsslateplayers:
                        l.append(player)
                json_list = l
            if name == 'dfsslatetournament':
                l = []
                for item in json_list:
                    dfsslatetournament = item['DfsSlateTournaments']
                    for tournament in dfsslatetournament:
                        del (tournament['Tournament'])
                        l.append(tournament)
                json_list = l
            if name == 'playerhole':
                l = []
                players = json_list['Players']
                for player in players:
                    rounds = player['Rounds']
                    for round in rounds:
                        holes = round['Holes']
                        for hole in holes:
                            l.append(hole)
                json_list = l
            if name == 'playerround':
                l = []
                players = json_list['Players']
                for player in players:
                    rounds = player['Rounds']
                    for round in rounds:
                        del (round['Holes'])
                        l.append(round)
                json_list = l
            if name == 'playertournament':
                l = []
                players = json_list['Players']
                for player in players:
                    del (player['Rounds'])
                    l.append(player)
                json_list = l
            if name == 'playertournamentprojection':
                for item in json_list:
                    del (item['Rounds'])
            if name == 'playertournamentodds':
                l = []
                odds = json_list['PlayerTournamentOdds']
                for odd in odds:
                    odd['tournamentid'] = tournamentid
                    l.append(odd)
                json_list = l
            if name == 'round':
                l = []
                for item in json_list:
                    rounds = item['Rounds']
                    for round in rounds:
                        l.append(round)
                json_list = l
            json_list = [{k.lower(): v for k, v in entry.items()} for entry in json_list]
            if len(json_list) > 0:
                try:
                    conn.execute(table.insert(), json_list)
                    now = dt.datetime.now().time()
                    print(
                        '{} data inserted into {} table         {}:{}:{:.0f}'.format(url, table.name, now.hour,
                                                                                     now.minute, now.second))
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
                        print(
                            '{} data exists        {}:{}:{:.0f}'.format(url, now.hour, now.minute, now.second))
                    pass
    except:
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