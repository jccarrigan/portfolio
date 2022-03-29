# %% set defaults

import json
import time
from json.decoder import JSONDecodeError

import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, TimeoutException
from sqlalchemy import create_engine


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


# %% dask output

def dask_output(contests):
    output = []
    for contest in contests:
        output.append(grab_contests(contest))
    return output


# %%

engine = create_engine('postgresql://jc:M()$@l@h!1@pga-postgresql.cxmbk6ooy1lu.us-east-1.rds.amazonaws.com/pga')
conn = engine.connect()

contests = conn.execute(
    "SELECT distinct (contestid) FROM contest WHERE site = 'draftkings' and (period = '2020-06-11')").fetchall()
contests = [c[0] for c in contests]
contests = set(contests)
# contests = np.array([x for x in contests])

db = conn.execute(
    "SELECT DISTINCT (cp.contestid) FROM contestpayout cp INNER JOIN (SELECT distinct (contestid) as contestid FROM \
    contest WHERE site = 'draftkings' and (period = '2020-06-11')) c on cp.contestid=c.contestid").fetchall()

db = [x[0] for x in db]
db = set(db)
conn.close()

remaining_contests = list(contests - db)
contest_list = np.array_split(remaining_contests, 10, axis=0)
output = dask_output(contest_list[9])
fails = sum([x[1] for x in output])

# %%
