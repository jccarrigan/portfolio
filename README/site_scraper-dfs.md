# Web Scraping using Selenium 



## Motivation

Web scraping is a great way to create novel datasets. In this particular case, I was interested in acquiring historical \
data from DraftKings to inform a backtesting algorithm. I was able to scrape this data off of Awesemo.com through a \
combination of Python and Selenium. Selenium is typically used for website testing, and it allows us to seamlessly \
simulate a user experience on a site and capture the data that is typically rendered in browser. Below demonstrates how \
I went about logging in to the site, navigating to the proper page, grabbing the appropriate data, and then placing it in \
a Postgres database residing in the cloud using Amazon RDS. The python file described in this markdown file can also \
be found in this directory. 


## Overview

* Finding Appropriate Contests
* Navigating Website using Selenium 
* Parsing the Data
* Placement in Amazon RDS Postgres Database
* Closing Thoughts 

## Finding appropriate contests

The first step in the process was to query the existing contest database to determine which contests were already in \
the database and which needed to be scraped. 

First we start with the necessary imported libraries. 

```python
import json
import time
from json.decoder import JSONDecodeError

import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, TimeoutException
from sqlalchemy import create_engine
from dask import delayed, compute

```

We then connect to our contest database and grab all associated contestid's for the date of interest, in this example \
being June 11th, 2020. These are obtained through a different ETL process and will not be covered in detail here. 

```python 
engine = create_engine('postgresql://user:password@pga-postgresql.cxmbk6ooy1lu.us-east-1.rds.amazonaws.com/pga')
conn = engine.connect()
contests = conn.execute(
    "SELECT distinct (contestid) FROM contest WHERE site = 'draftkings' and (period = '2020-06-11')").fetchall()
contests = [c[0] for c in contests]
```

 We then grab the contestids from our contestpayout database that have already been populated with detailed contest data.

```python
db = conn.execute(
    "SELECT DISTINCT (cp.contestid) FROM contestpayout cp INNER JOIN (SELECT distinct (contestid) as contestid FROM \
    contest WHERE site = 'draftkings' and (period = '2020-06-11')) c on cp.contestid=c.contestid").fetchall()

db = [x[0] for x in db]
conn.close()
```

We now find the contests that are in the contest database but do not have detailed contest info by utilizing python's set \
datatype to find the difference betweeen these two lists of contests. 

```python
contests = set(contests)
db = set(db)
remaining_contests = list(contests - db)
```
An advanced feature I've implemented for this script is to take advantage of dask to distribute the scraping across multiple \
Amazon EC2 instances for faster scraping. I split the list into the desired number of instances, and run the scripts \
concurrently in PyCharm. Below is an example of launching the final instance of contests. I've also implemented some basic \
data testing to determine if the data was captured successfully.

```python 
contest_list = np.array_split(remaining_contests, 10, axis=0)
output = dask_output(contest_list[9])
fails = sum([x[1] for x in output])
```
REMINDER: Be careful, this can lead to overloading the website and can alert the website administrator that someone is \
scraping their site through unusual behavior. Make sure to always be considerate of the website host whenever \
developing scraping procedures.

We will delve into the dask_output function next.

```python
def dask_output(contests):
    output = []
    for contest in contests:
        output.append(compute(grab_contests(contest)))
    return output
```
 This function primarily handles dask's distributed functionality, and returns some basic information about whether \
scraping was successful.

## Navigating the website using Selenium

The section below includes the function grab_contests, which handles site navigation and data processing for the contest \
of interest. We use the delayed wrapper to signify to Dask we want this process distributed to our different processors.

Here we have the basic url info for navigating our site, along with the necessary steps to login and navigate to the page \
of interest. 

```python
@delayed 
def grab_contests(contest):
    awesemo_login = 'https://www.awesemo.com/login2/'
    user = ''
    pwd = ''
    
        contest_info = "https://awesemo.fantasycruncher.com/lineup-study/draftkings/PGA"
        get_data = "https://awesemo.fantasycruncher.com/funcs/tournament-analyzer/get-contest-data.php?data=init&contest_id={}"
        get_lineups = "https://awesemo.fantasycruncher.com/funcs/tournament-analyzer/get-contest-data.php?data=lineups" \
                      "&contest_id={}&offset=0&limit=500000"
    
        engine = create_engine('postgresql://@pga-postgresql.cxmbk6ooy1lu.us-east-1.rds.amazonaws.com/pga')
        conn = engine.connect()
    
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.headless = True
    
        browser = webdriver.Chrome(executable_path='/usr/bin/chromedriver',
                                   desired_capabilities=webdriver.DesiredCapabilities.CHROME, options=chrome_options)
    
        d = {}
        exception = False
        try:
            browser.get(awesemo_login)
            time.sleep(60)
            button_xpath = '/html/body/div[11]/div[1]/div/button'
            button = browser.find_element_by_xpath(button_xpath)
            browser.execute_script("arguments[0].click();", button)
            time.sleep(1)
            username = browser.find_element_by_id("user_login")
            username.send_keys(user)
            time.sleep(1)
            password = browser.find_element_by_id("user_pass")
            password.send_keys(pwd)
            time.sleep(1)
            browser.find_element_by_name("submit").click()
            time.sleep(1)
            browser.get(contest_info)
            time.sleep(1)
            browser.get(get_data.format(contest))
            time.sleep(1)
```

The best way to determine how to approach scraping a site is to use your browser to inspect the website's html \
code. This is how I was able to determine elements needed to emulate a user's experience logging in and grab the proper data. 

## Parsing the data and placing in Amazon RDS Postgres database
Here we now process the data of interest and place it in the appropriate database. The json response we receive from the site \
includes the data for the payout, users, and players data. We also see the conclusion of some exception handling. As mentioned \
before, this method returns a tag regarding scraping success for each contest.
```python
        contest_data = browser.find_element_by_tag_name('body').text
        d = json.loads(contest_data)
        d = d['data']

        payout = d['payouts']
        payout = pd.DataFrame(payout).reset_index(level=0)
        payout.rename({'index': 'rank'}, axis=1, inplace=True)
        payout['contestid'] = contest
        if len(payout) > 0:
            payout.to_sql('contestpayout', index=False, if_exists='append', con=conn)

        user = d['users']
        user = [{k.lower(): v for k, v in entry.items()} for entry in user]
        user = pd.DataFrame(user)
        if len(user) > 0:
            user['players_used'] = [str(x) for x in user['players_used'].values]
            user['team_stacks_used'] = [str(x) for x in user['team_stacks_used'].values]
        user['contestid'] = contest
        if len(user) > 0:
            user.to_sql('contestuser', index=False, if_exists='append', con=conn)

        player = d['players']
        player = [x for x in player.values()]
        player = [{k.lower(): v for k, v in entry.items()} for entry in player]
        player = pd.DataFrame(player)
        player['contestid'] = contest
        if len(player) > 0:
            player.to_sql('contestplayer', index=False, if_exists='append', con=conn)

        time.sleep(1)
        browser.get(get_lineups.format(contest))
        time.sleep(1)
        lineup_data = browser.find_element_by_tag_name('body').text
        d = json.loads(lineup_data)
        d = d['data']

        lineup = d['lineups']
        lineup = [{k.lower(): v for k, v in entry.items()} for entry in lineup]
        lineup = pd.DataFrame(lineup)
        lineup['contestid'] = contest
        if len(lineup) > 0:
            lineup['meta_data'] = [str(x) for x in lineup['meta_data'].values]
            lineup.to_sql('contestlineup', index=False, if_exists='append', con=conn)
    except (KeyError, NoSuchElementException, ElementClickInterceptedException, TimeoutException, AttributeError,
            JSONDecodeError) as e:
        exception = True
        pass
    conn.close()
    browser.quit()

    return [contest, exception]
```


## Closing Thoughts 

This script has worked well for scraping Awesemo.com for DraftKings outcome information relating to contest payouts, users \
players, and lineups.

Some areas of improvement include:
* comment strings to elucidate effort of code
* better exception handling
* building concurrency into script
* clearer variable naming
* handling login info in a more sensitive manner, embed in path variables