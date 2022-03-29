# %% import packages

import warnings

import numpy as np
import pandas as pd
from autokeras import StructuredDataRegressor
from dask import delayed, compute
from dask.distributed import Client
from autogluon import TabularPrediction as task
from ngboost import NGBRegressor
from pydfs_lineup_optimizer import get_optimizer, Site, Sport, Player, LineupOptimizerException
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MaxAbsScaler
from sqlalchemy import create_engine
from xgboost import XGBRegressor

# %% PGA Simulator

class PgaSimulator(object):
    def __init__(self, model='ngb', limit_features=False, transform_target=False,
                 add_projections=False, advanced_stats=True, advanced_only=False, roll=False, window=4, diff=False,
                 start_tournament=381, entry_tournament=392, training_tournaments=4, site='DraftKings', mode='Classic',
                 predictions_only=True, n_lineups=10, exposure= .2, min_salary=True, salary=49800, unique=True, uniques=2, random=False,
                 dev=False, min_dev=0, max_dev=.3, min_cost=4, max_cost=10, min_entries=10, max_entries=150,
                 min_payout=5000, contest_name=None):

        self.model = model
        self.limit_features = limit_features
        self.transform_target = transform_target
        self.add_projections = add_projections
        self.advanced_stats = advanced_stats
        self.advanced_only = advanced_only
        self.roll = roll
        self.window = window
        self.diff = diff

        self.start_tournament = start_tournament
        self.entry_tournament = entry_tournament
        self.training_tournaments = training_tournaments
        self.site = site
        self.mode = mode

        self.predictions_only = predictions_only
        self.n_lineups = n_lineups
        self.exposure = exposure
        self.min_salary = min_salary
        self.salary = salary
        self.unique = unique
        self.uniques = uniques
        self.random = random
        self.dev = dev
        self.min_dev = min_dev
        self.max_dev = max_dev

        self.min_cost = min_cost
        self.max_cost = max_cost
        self.min_entries = min_entries
        self.max_entries = max_entries
        self.min_payout = min_payout
        self.contest_name = contest_name

        self.ids = ['playerid', 'tournamentid']
        self.slate_info = ['slateid', 'playertournamentprojectionid', 'operatorslateplayerid', 'operatorposition',
                           'operatorsalary']
        self.optimizer_info = ['TeamAbbrev', 'ID', 'Name', 'Position', 'Salary', 'AvgPointsPercontest', 'projection',
                               'actual']
        self.target = ['fantasypoints', 'fantasypointsdraftkings', 'fantasypointsfanduel', 'fantasypointsyahoo',
                       'fantasypointsfantasydraft']
        self.salaries = ['draftkingssalary', 'fanduelsalary', 'yahoosalary', 'fantasydraftsalary']

        self.tournaments = None
        self.tourneys = None
        self.t = {}

        self.start_date = None
        self.entry_date = None

        self.players = None
        self.predictions = None
        self.lineup_results = None
        self.contest_results = None
        self.contests = None
        self.cost = None
        self.winnings = None

    def simulate(self, verbose=False, tracking=False):

        engine = create_engine(
            'postgresql://username:password@pga-postgresql.cxmbk6ooy1lu.us-east-1.rds.amazonaws.com/pga')
        conn = engine.connect()
        self.tournaments = pd.read_sql("SELECT tournamentid FROM tournament", conn).values[::-1]

        for x in range(len(self.tournaments)):
            self.t[int(self.tournaments[x])] = x

        self.tourneys = [int(x) for x in
                         self.tournaments[int(np.where(self.tournaments == self.start_tournament)[0]):int(
                             np.where(self.tournaments == self.entry_tournament)[0]) + 1]]

        self.start_date = \
            conn.execute(f'SELECT startdate FROM tournament WHERE tournamentid = {self.start_tournament}').fetchone()[0]
        self.entry_date = \
            conn.execute(f'SELECT startdate FROM tournament WHERE tournamentid = {self.entry_tournament}').fetchone()[0]
        conn.close()

        warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning, DeprecationWarning))
        np.random.seed(seed=0)

        self._clean_data()
        self._run_historical_simulation()

        if not self.predictions_only:
            self._backtest_contest_results()
            if verbose:
                self._print_results()

    def _clean_data(self):
        def _datagolf():
            datagolf_query = "SELECT t.tournamentid, p.playerid, d.sg_app as approach, d.sg_arg as short_contest, d.sg_ott \
                            as driving, d.sg_putt as putting, d.sg_t2g as tee_to_green, d.sg_total FROM datagolf d \
                                INNER JOIN player p on d.player_num = p.pgatourplayerid \
                                INNER JOIN tournament t on d.tournamentid = t.tournamentid \
                                    WHERE startdate >= '2018-10-04'"

            engine = create_engine(
                'postgresql://username:password@pga-postgresql.cxmbk6ooy1lu.us-east-1.rds.amazonaws.com/pga')
            conn = engine.connect()
            datagolf_players = pd.read_sql(datagolf_query, conn)
            conn.close()

            return datagolf_players

        engine = create_engine(
            'postgresql://username:password@pga-postgresql.cxmbk6ooy1lu.us-east-1.rds.amazonaws.com/pga')
        conn = engine.connect()
        players_query = "SELECT pt.* FROM playertournament pt INNER JOIN playertournamentprojection ptj on \
        pt.playertournamentid = ptj.playertournamentid WHERE (pt.teetime >= '2018-10-04' or pt.teetime is null) \
        and ptj.iswithdrawn is False"
        print('Grabbing data......')
        self.players = pd.read_sql(players_query, conn)
        conn.close()

        if self.advanced_stats:
            datagolf_players = _datagolf()
            self.players = self.players.merge(datagolf_players, on=['tournamentid', 'playerid'])
            dg_cols = ['approach', 'short_contest', 'driving', 'putting', 'tee_to_green', 'sg_total']
            self.players[dg_cols] = self.players[dg_cols].fillna(0)

        self.players['fpdk'] = self.players.fantasypointsdraftkings

        cols = ['country', 'totalthrough', 'tournamentstatus', 'isalternate', 'oddstowin', 'fantasydraftsalary',
                'madecutdidnotfinish', 'oddstowindescription', 'iswithdrawn', 'teetime']
        self.players.drop(cols, axis=1, inplace=True)

        if self.advanced_only:
            self.players = self.players[
                dg_cols + self.target + self.ids + self.salaries[:2] + ['playertournamentid', 'fpdk', 'name']].copy()

    def _transform_data(self, slate_players):

        df = slate_players.groupby(['playerid', 'tourney']).mean()

        d = []
        if self.roll:
            rolling = df.drop(self.target + self.salaries[:2], axis=1).rolling(
                window=self.window)
            d.append(rolling.mean())
            lagged_stats = pd.concat(d, axis=1)
            new = lagged_stats.groupby(level=0).shift(1)
        else:
            new = df.drop(self.target + self.salaries[:2], axis=1).groupby(level=0).shift(1)

        df = df[self.target + self.salaries[:2]].merge(new, left_index=True, right_index=True)

        df['y_diff'] = pd.concat(
            [z.diff(1) for z in [y for x, y in df.fantasypointsdraftkings.groupby(level=0)]],
            axis=0)
        if not self.advanced_only:
            df.dropna(axis=0, subset=['totalstrokes'], inplace=True)
        df.fillna(0, inplace=True)

        return df

    def _grab_slate_data(self, tournament):

        engine = create_engine(
            'postgresql://username:password@pga-postgresql.cxmbk6ooy1lu.us-east-1.rds.amazonaws.com/pga')
        conn = engine.connect()
        slate_info = [('projection', 'projection') if tournament == self.entry_tournament else ('actual', '')][0]

        slate = f"SELECT t.slateid, \
                    t.tournamentid, \
                    a.operatorslateplayerid, \
                    a.playertournamentprojectionid, \
                    p.playerid, \
                    a.operatorplayername, \
                    a.operatorposition, \
                    a.operatorsalary, \
                    p.fantasypoints{self.site.lower()} as {slate_info[0]} \
                FROM dfsslate s \
                INNER JOIN dfsslatetournament t on s.slateid = t.slateid \
                INNER JOIN dfsslateplayer a on s.slateid = a.slateid \
                INNER JOIN playertournament{slate_info[1]} p on a.playerid = p.playerid AND p.tournamentid = t.tournamentid \
                WHERE t.tournamentid = {tournament} AND s.operator = '{self.site}' AND s.operatorgametype = '{self.mode}'"

        slate_data = pd.read_sql(slate, conn)
        slate_data.sort_values(['operatorplayername', 'operatorsalary'], ascending=False).reset_index(
            inplace=True, drop=True)
        conn.close()

        player_data = slate_data.drop(
            self.slate_info + ['actual' if tournament != self.entry_tournament else 'projection'],
            axis=1).drop_duplicates()

        slate_players = player_data.merge(self.players, how='outer')
        slate_players['tourney'] = [self.t[x] for x in slate_players.tournamentid]
        slate_players.drop(['tournamentid', 'playertournamentid'], axis=1, inplace=True)
        slate_players.operatorplayername.fillna(slate_players.name, inplace=True)

        if self.limit_features:
            feature_cols = ['earnings', 'sg_total', 'tee_to_green', 'pars', 'fedexpoints',
                            'streaksofthreebirdiesorbetter']
            slate_players = slate_players[feature_cols + ['playerid', 'tourney'] + self.target + self.salaries[:2]]

        return slate_data, slate_players

    def _train_predict(self, slate_data, player_data, tournament):

        np.random.seed(0)
        df = player_data

        if self.add_projections:
            engine = create_engine(
                'postgresql://username:password@pga-postgresql.cxmbk6ooy1lu.us-east-1.rds.amazonaws.com/pga')
            conn = engine.connect()

            projections = pd.read_sql("SELECT tournamentid, \"Player\", \"Own\", \"FC Proj\"  FROM \"cruncher-rewind\"",
                                      conn)
            players = pd.read_sql('SELECT playerid, CONCAT(firstname, \' \', lastname) as \"Player\" FROM player', conn)
            projections = projections.merge(players, on='Player').drop('Player', axis=1)
            projections['tourney'] = [self.t[x] for x in projections.tournamentid]
            projections = projections.groupby(['playerid', 'tourney']).mean().dropna(
                axis=0)
            df = df.join(projections, rsuffix='proj')
            conn.close()

        if len(slate_data) > 0:
            tourney = int(np.where(self.tournaments == tournament)[0])

            if self.training_tournaments == 0:
                df_train = df.loc(axis=0)[:, :(tourney - 1)].copy()
            else:
                df_train = df.loc(axis=0)[:, (tourney - self.training_tournaments):(tourney - 1)].copy()

            for i in range(1, 20):
                if len(df_train) == 0:
                    df_train = df.loc(axis=0)[:, (tourney - self.training_tournaments - i):(tourney - 1)].copy()
                else:
                    break

            df_train.dropna(inplace=True)
            X_train = df_train.drop(self.target + ['fpdk', 'y_diff'], axis=1)

            if self.site == 'DraftKings':
                y_train = df_train.fantasypointsdraftkings
            elif self.site == 'FanDuel':
                y_train = df_train.fantasypointsfanduel
            if self.diff:
                y_train = df_train.y_diff

            df_pred = df.loc(axis=0)[:, tourney].copy()
            X_pred = df_pred.drop(self.target + ['fpdk', 'y_diff'], axis=1)
            X_pred.dropna(inplace=True)
            y_pred = pd.Series()

            if len(X_pred) > 0:

                ss = MaxAbsScaler()
                ys = MaxAbsScaler()

                X_trains = ss.fit_transform(X_train)
                X_preds = ss.fit_transform(X_pred)

                if self.transform_target:
                    y_trains = ys.fit_transform(y_train.values.reshape(-1, 1))
                    y_trains = [y[0] for y in y_trains]
                    y_trains = pd.Series(y_trains)
                else:
                    y_trains = y_train

                np.random.seed(seed=0)
                if self.model == 'autogluon':
                    df_train.drop(['fpdk', 'y_diff'], axis=1).to_csv(f'/mnt/efs/{tourney}-df_train.csv', index=False)
                    X_pred.to_csv(f'/mnt/efs/{tourney}-X_pred.csv', index=False)
                    model = task.fit(train_data=task.Dataset(file_path=f'/mnt/efs/{tourney}-df_train.csv'),
                                     label='fantasypointsdraftkings', output_directory='/mnt/efs/', auto_stack=True)
                if self.model == 'xgb':
                    model = XGBRegressor(verbose=False)
                    model.fit(X_trains, y_trains)
                    y_preds = model.predict(X_preds)
                if self.model == 'ngb':
                    model = NGBRegressor(verbose=False)
                    model.fit(X_trains, y_trains)
                    y_preds = model.predict(X_preds)
                if self.model == 'autokeras':
                    model = StructuredDataRegressor(max_trials=100, seed=0)
                    model.fit(x=X_trains, y=y_trains, epochs=10)
                    y_preds = model.predict(X_preds)
                    y_preds = [y[0] for y in y_preds]
                if self.model == 'autogluon':
                    y_preds = model.predict(dataset=task.Dataset(file_path=f'/mnt/efs/{tourney}-X_pred.csv'))
                    y_preds = [y[0] for y in y_preds]

                if self.transform_target:
                    y_preds = pd.DataFrame(y_preds)
                    y_preds = ys.inverse_transform(y_preds)
                    y_preds = [y[0] for y in y_preds]
                    y_preds = pd.Series(y_preds, index=X_pred.index, name='AvgPointsPerGame')
                    y_preds.fillna(0, inplace=True)

                y_pred = pd.Series(y_preds, index=X_pred.index, name='AvgPointsPerGame')

                if self.diff:
                    y_pred += df_pred.fpdk

            return y_pred.reset_index()
        else:
            return pd.Series()

    def _generate_lineups(self, slate_data, predictions, tournament):
        def _optimize(predictions, slate_data, slate):

            slate_projections = predictions[predictions.slateid == slate].copy()
            slate_projections.sort_values(by=['operatorplayername', 'operatorsalary'], ascending=True).reset_index(
                drop=True, inplace=True)

            if tournament == self.entry_tournament:
                export = []
                fantasy_points = 'projection'
            else:
                fantasy_points = 'actual'

            optimizer_projections = slate_projections.drop(
                ['slateid', 'tournamentid', 'tourney', 'playertournamentprojectionid', 'playerid', fantasy_points],
                axis=1).copy()

            optimizer_projections.columns = self.optimizer_info[1:6]

            players = []
            for row in optimizer_projections.iterrows():
                row = row[1]
                players.append(
                    Player(str(row.ID), row.Name.split(' ')[0], row.Name.split(' ')[1], [row.Position], '',
                           row.Salary,
                           row.AvgPointsPercontest))

            if self.site == 'DraftKings':
                operator_site = Site.DRAFTKINGS
            elif self.site == 'FanDuel':
                operator_site = Site.FANDUEL

            optimizer = get_optimizer(operator_site, Sport.GOLF)
            if self.site == 'FanDuel':
                optimizer.settings.min_teams = None
            optimizer.load_players(players)

            lineups, points, pts_summary = [], [], []
            optimizer_lineups = optimizer.optimize(n=self.n_lineups, max_exposure=self.exposure, randomness=self.random)

            if self.dev:
                optimizer.set_deviation(min_deviation=self.min_dev, max_deviation=self.max_dev)
            if self.unique:
                optimizer.set_max_repeating_players(6 - self.uniques)
            if self.min_salary:
                optimizer.set_min_salary_cap(self.salary)

            try:
                for lineup in optimizer_lineups:
                    p_id = []
                    [p_id.append(player._player.id) for player in lineup.lineup]
                    lineups.append(
                        optimizer_projections[optimizer_projections.ID.isin(p_id)][['Name', 'ID']].values)
                    if tournament == self.entry_tournament:
                        export.append(p_id)
            except LineupOptimizerException:
                pass

            if tournament != self.entry_tournament:
                [points.append(slate_data[slate_data.operatorslateplayerid.isin([x[1] for x in lineup])].actual.sum())
                 for
                 lineup in lineups]
                try:
                    pts_summary.append((float(max(points)), float(sum(points) / len(points))))
                except ValueError:
                    pass

            if tournament == self.entry_tournament:
                exports = pd.DataFrame(export, columns=['G'] * 6)
                exports.to_csv('/mnt/efs/' + str(tournament) + '-' + str(slate) + 'lineupexports.csv', index=False)

            lineup_result = [tournament, slate, lineups, points, pts_summary]
            return lineup_result

        lineup_results = []
        if len(predictions) > 0:
            predictions = slate_data.merge(predictions)

            if tournament != self.entry_tournament:
                rmse = np.sqrt(mean_squared_error(predictions.actual, predictions.AvgPointsPerGame))
                mae = mean_absolute_error(predictions.actual, predictions.AvgPointsPerGame)
                metrics = rmse, mae
            else:
                metrics = 0, 0

            slates = slate_data.slateid.unique()
            for slate in slates:
                results = _optimize(predictions, slate_data, slate)
                results.append(metrics)
                lineup_results.append(results)

        return lineup_results, predictions

    @delayed
    def _tournament_simulation(self, tournament):

        slate_data, slate_players = self._grab_slate_data(tournament)
        player_data = self._transform_data(slate_players)

        prediction = []
        predictions = self._train_predict(slate_data, player_data, tournament)
        if not predictions.empty:
            prediction.append(predictions)
        predictions = pd.concat(prediction, axis=0)
        predictions = slate_data.merge(predictions)

        if tournament != self.entry_tournament:
            rmse = np.sqrt(mean_squared_error(predictions.actual, predictions.AvgPointsPerGame))
            mae = mean_absolute_error(predictions.actual, predictions.AvgPointsPerGame)
            metrics = rmse, mae
        else:
            metrics = 0, 0

        if self.predictions_only:
            results = [predictions, metrics]
        else:
            results = self._generate_lineups(slate_data, predictions, tournament)
        return results

    def _simulations_wrapper(self):
        np.random.seed(0)
        results = []
        print('Simulating tournaments...')
        for tournament in self.tourneys:
            result = self._tournament_simulation(tournament)
            results.append(result)
        return results

    def _run_historical_simulation(self):

        np.random.seed(0)
        results = compute(self._simulations_wrapper())

        if self.predictions_only:
            self.lineup_results = results
        else:
            self.lineup_results = results[0]
            self.lineup_results = [(x, y) for x, y in self.lineup_results if x]
            self.predictions = [x[1].sort_values(by='AvgPointsPerGame', axis=0, ascending=False) for x in
                                self.lineup_results]
            self.lineup_results = [x[0] for x in [x[0] for x in self.lineup_results]]

    @delayed
    def _tally_winnings(self, contestid):
        engine = create_engine(
            'postgresql://username:password@pga-postgresql.cxmbk6ooy1lu.us-east-1.rds.amazonaws.com/pga')
        conn = engine.connect()
        contest_info = f"SELECT l.contestid, \
                    l.rank as place, \
                    l.points, \
                    l.cash_amt, \
                    t.tournamentid, \
                    t.slateid, \
                    c.period, \
                    c.cost, \
                    c.max_entries, \
                    c.prizepool, \
                    c.winning_score, \
                    c.mincash_score, \
                    c.winning_payout, \
                    c.mincash_payout \
                        FROM contest c \
                            INNER JOIN dfsslate s on c.slate = s.operatorslateid \
                            INNER JOIN dfsslatetournament t on s.slateid = t.slateid \
                            INNER JOIN contestlineup l on c.contestid = l.contestid \
                                WHERE l.contestid = {contestid} and l.cash_amt > 0"

        contest = pd.read_sql(contest_info, conn).sort_values(by='place')
        payout = pd.read_sql(
            f'SELECT rank as place, by_rank, ranges, last_paid_rank, contestid FROM contestpayout WHERE contestid = {contestid}',
            conn)
        conn.close()

        scores = []
        try:
            scores = \
                [x[3] for x in self.lineup_results if x[0] == contest.tournamentid.unique()
                 and x[1] == contest.slateid.unique()][0]
        except IndexError:
            pass

        costs, winnings = 0, 0
        if not payout.empty and not contest.empty and scores:
            payout['place'] = [int(x) for x in payout['place'].values]
            payout = payout.sort_values(by='place').fillna(method='ffill').drop(
                ['last_paid_rank', 'contestid'], axis=1)
            payout[['ranges', 'by_rank']] = payout[['ranges', 'by_rank']].apply(lambda x: x / 100)

            ranks = payout['place'].values
            places = [i for i in range(1, ranks[-1] + 1)]

            ranks = pd.Series([x if x in ranks else None for x in places]).fillna(method='ffill')
            ranks = [x for x in ranks.values]

            ranges = [[int(x) for x in payout[payout['place'] == i]['ranges'].values] for i in ranks]
            ranges = [int(r[0]) for r in ranges]

            wins_or_ties = []
            contest_points = contest[['place', 'points']].copy()
            n_entries = int(contest.max_entries.unique())

            scores = scores[:n_entries]
            scores = sorted(scores, reverse=True)
            for score in scores:
                cost = float(contest.cost.unique())
                if cost == 0:
                    cost = .25
                costs += cost
                for row in contest_points.iterrows():
                    row = row[1]
                    if score > row.points:
                        wins_or_ties.append((row.place + len(wins_or_ties), score, 0))
                        break
                    elif score == row.points:
                        wins_or_ties.append((row.place + len(wins_or_ties), score, 1))
                        break

            for place in wins_or_ties:
                if place[2] == 0:
                    contest_points[contest_points.place >= place[0]].place.apply(lambda x: x + 1)
                    contest_points = contest_points.append({'place': place[0], 'points': place[1]},
                                                           ignore_index=True)
                if place[2] == 1:
                    contest_points[contest_points.place > place[1]].place.apply(lambda x: x + 1)
                    contest_points = contest_points.append({'place': place[0], 'points': place[1]},
                                                           ignore_index=True)

            contest_points = contest_points[contest_points.place <= contest.place.values[-1]].copy()

            new_rankings = places, ranges, [x for x in contest_points['place'].values]
            new_rankings = pd.DataFrame(new_rankings).T
            new_rankings = new_rankings.rename(columns={0: 'place', 1: 'ranges', 2: 'rank'}) \
                .sort_values(by='place').dropna(axis=0)

            new_payout = []
            for rank in [x for x in new_rankings['rank'].unique()]:
                ties = new_rankings[new_rankings['rank'] == rank].copy()
                tie_amount = len(ties)
                [new_payout.append(sum(ties.ranges.values) / tie_amount) for _ in range(tie_amount)]
            new_rankings['payout'] = new_payout

            placements = wins_or_ties
            if placements and not new_rankings.empty:
                for placement in placements:
                    try:
                        winnings += new_rankings[new_rankings.place == placement[0]].payout.values[0]
                    except IndexError:
                        pass

        return costs, winnings

    def _run_historical_winnings_calculation(self):
        contests, cost, winnings = [], [], []
        print('Simulating contests...')
        for contest in [x for x in self.contests.contestid.unique()]:
            contests.append(contest)
            tally = self._tally_winnings(contest)
            cost.append(tally[0])
            winnings.append(tally[1])

        contest_results = [contests, cost, winnings]
        cost = sum(cost)
        winnings = sum(winnings)

        return [contest_results, cost, winnings]

    def _backtest_contest_results(self):
        engine = create_engine(
            'postgresql://username:password@pga-postgresql.cxmbk6ooy1lu.us-east-1.rds.amazonaws.com/pga')
        conn = engine.connect()
        contests = f"SELECT DISTINCT(l.contestid), \
                            t.tournamentid, \
                            c.name as contest_name\
                        FROM contest c \
                            INNER JOIN dfsslate s on c.slate = s.operatorslateid \
                            INNER JOIN dfsslatetournament t on s.slateid = t.slateid \
                            INNER JOIN contestlineup l on c.contestid = l.contestid \
                                WHERE c.site = '{self.site.lower()}' and s.operatorgametype = '{self.mode}' \
                                and c.cost >= {self.min_cost} and c.cost <= {self.max_cost} \
                                and c.winning_payout >={self.min_payout} and c.max_entries >= {self.min_entries} \
                                and c.max_entries <={self.max_entries} and c.period >= '{self.start_date}' \
                                and c.period < '{self.entry_date}'"
        self.contests = pd.read_sql(contests, conn)
        conn.close()

        if self.contest_name:
            self.contests = self.contests[self.contests.contest_name.str.contains(f'{self.contest_name}')]

        contest_results = compute(self._run_historical_winnings_calculation())
        contest_results = contest_results[0]

        self.contest_results = contest_results[0]
        self.cost = contest_results[1]
        self.winnings = contest_results[2]

    def _print_results(self):
        try:
            print(
                '\n+++++++++++++++++++++++++++++++++++++++++++++CONTEST '
                'RESULTS++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Tournaments: {}'.format(int(self.contests.tournamentid.nunique())), end='            ')
            print('Contests: {}'.format(int(self.contests.contestid.nunique())), end='            ')
            print('Cost: ${:.0f}'.format(self.cost), end='            ')
            print('Winnings: ${:.0f}'.format(self.winnings), end='          ')
            print('Profit: ${:.0f}'.format(self.winnings - self.cost))
            print('\nCost/tournament: ${:.0f}/tournament'.format(self.cost / int(self.contests.tournamentid.nunique())),
                  end='                ')
            print('Profit/tournament: ${:.0f}/tournament'.format(
                (self.winnings - self.cost) / int(self.contests.tournamentid.nunique())))
            print('Cost/contest: ${:.0f}/contest'.format(self.cost / int(self.contests.contestid.nunique())),
                  end='                    ')
            print('Profit/contest: ${:.0f}/contest'.format(
                (self.winnings - self.cost) / int(self.contests.contestid.nunique())))
        except ZeroDivisionError:
            print('No contests to calculate!')


# %%

if __name__ == '__main__':
    client = Client()
    pga = PgaSimulator(training_tournaments=4)
    pga.simulate()

