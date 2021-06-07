

#!/usr/bin/env python
import sys
import argparse 

import pandas as pd 

import cmd2

from . import api, dashboard

list_parser = argparse.ArgumentParser()
list_parser.add_argument('year', type=int, help='words to say')

speak_parser = argparse.ArgumentParser()
speak_parser.add_argument('-p', '--piglatin', action='store_true', help='atinLay')
speak_parser.add_argument('-s', '--shout', action='store_true', help='N00B EMULATION MODE')
speak_parser.add_argument('-r', '--repeat', type=int, help='output [n] times')
speak_parser.add_argument('words', nargs='+', help='words to say')



class RowingApp(cmd2.Cmd):
    """A simple cmd2 application."""
    
    def __init__(self):
        super().__init__()

        # Make maxrepeats settable at runtime
        self.current_race = api.get_last_race_started().name
        self.current_competition = api.get_most_recent_competition().name

        self.add_settable(
            cmd2.Settable('current_race', int, 'id of current race')
        )
        self.add_settable(
            cmd2.Settable('current_competition', int, 'id of current competition')
        )

    def select_from_dataframe(self, df, column='DisplayName', name='row'):
        selected_id = self.select(
            list(df[column].items()),
            f"Select which {name} you want "    
        )
        return df.loc[selected_id]

    @cmd2.with_argparser(list_parser)
    def do_select_race(self, args):
        race, event, competition = self.select_race(args.year)
        self.current_race = race.name
        self.current_event = competition.name
        self.current_competition = competition.name
        self.poutput(race)

    @cmd2.with_argparser(list_parser)
    def do_pgmts(self, args):
        competition = self.select_competition(args.year)
        pgmts = api.get_competition_pgmts(competition.name)
        group_boat_pgmts = pgmts.groupby('BoatClass')
        boat_pgmts = group_boat_pgmts\
            .first()\
            .sort_values('PGMT', ascending=False)
        self.poutput(
            f"loaded PGMTS for {len(pgmts)} results"
        )
        while True:
            mode = self.select(
                [
                    'by result', 
                    'by boat class',
                    'by final',
                    'stop showing'
                ],
                'How to display PGMTs?'
            ) 
            if mode == 'by result':
                self.poutput(pgmts.to_string())
            elif mode == 'by boat class':
                self.poutput(boat_pgmts.to_string())
            elif mode == 'by final':
                final_pgmts = api.get_competition_pgmts(
                    competition.name, finals_only=True)
                self.poutput(final_pgmts.to_string())
            elif mode == 'stop showing':
                break



    def select_competition(self, year):
        return self.select_from_dataframe(
            api.get_competitions(year), name='competition'
        )

    def select_race(self, year):
        competitions = api.get_competitions(year)
        competition = self.select_from_dataframe(
            competitions, name='competition'
        )
        events = api.get_competition_events(competition.name)
        if len(events) == 0:
            self.poutput("no races found for {competition.DisplayName}")
            return 

        event = self.select_from_dataframe(
            events, name='event'
        )
        races = api.get_competition_races(competition.name)
        if len(races) == 0:
            self.poutput("no races found for {event.DisplayName}")
            return 

        race = self.select_from_dataframe(
            races.loc[races.eventId == event.name], 
            name='race'
        )
        return race, event, competition
        
    
def main():
    sys.exit(RowingApp().cmdloop())


if __name__ == '__main__':
    main()